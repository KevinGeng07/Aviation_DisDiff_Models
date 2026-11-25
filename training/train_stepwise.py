"""
Stepwise distillation for CFM + SWI_DiT (pipeline stage 5): fewer sampling steps, same
architecture. Two methods (--method), trained from the same base model so they compare directly:

  progressive  (Salimans & Ho, 2022)  halve the step count round by round: 32 → 16 → ... → 1
  consistency  (Song et al., 2023)    one stage that trains the model to jump straight to the
                                      end of the teacher's ODE path from any point on it

── Progressive distillation ──────────────────────────────────────────────────

The architecture stays the same; what shrinks is the number of Euler steps the
sampler needs. Starting from any trained model (--base, default 4_256), each
round trains a student to cover in ONE Euler step what the frozen teacher
covers in TWO, halving the step count:

    round 1:  teacher @ 32 steps → student @ 16 steps     (teacher = base model)
    round 2:  teacher @ 16 steps → student @  8 steps     (teacher = round-1 student)
    ...       down to --min_steps (default 1)

The student of every round is initialized from its teacher's weights (so the
two share an architecture), and the next round's teacher is this round's best
EMA student.

Per-round loss, on the student's time grid t ∈ {0, 1/N, ..., (N-1)/N}
(N = student steps, dt = 1/(2N) = teacher step):

    x_t      = (1 - t)·noise + t·fut               (same interpolation as CFM training)
    v1       = teacher(obs, x_t,         t,      t_rel)
    v2       = teacher(obs, x_t + v1·dt, t + dt, t_rel)
    v_tgt    = (v1 + v2) / 2                       (= (x_{t+2dt} - x_t) / 2dt)
    L        = weighted_mse(student(obs, x_t, t, t_rel), v_tgt)

so one student Euler step of size 2dt lands exactly where two teacher steps do.
Loss weighting, as in training/train_distill.py (training/loss_weighting.py), applies to both
methods: --turn_weight / --turn_mode / --turn_threshold and --z_weight. Both methods match
the teacher, not the real trajectories, so weighting here only decides where the student
matches its teacher most closely; it cannot make it turn better than the base model does.

── Consistency distillation ──────────────────────────────────────────────────

The consistency function f(x_t, t) predicts the trajectory at t = 1 (the data end; t = 0 is
pure noise) from any point x_t. It is built on the same velocity network:

    f(x_t, t) = x_t + (1 - t)·v_θ(x_t, t)          (= where ONE Euler step from t lands)

so f(x, 1) = x holds by construction (the boundary condition), and eval/evaluate.py's Euler
sampler is the consistency sampler: --n_steps 1 is exactly x = f(noise, 0); with more steps
each Euler step moves along the straight line towards the current f prediction.

Training (Song et al. 2023, Algorithm 2), on a grid t_n = n/N, n = 0..N (N = --cd_n):

    n ~ U{0, ..., N-1},  x_{t_n} = (1 - t_n)·noise + t_n·fut
    x̂_{t_n+1} = one teacher ODE step from x_{t_n}  (--cd_solver heun | euler, step 1/N)
    target    = f_θ⁻(x̂_{t_n+1}, t_n+1)              (θ⁻ = slow EMA "target" copy, no grad)
    L         = d(f_θ(x_{t_n}, t_n), target)          (--cd_loss l2 | huber, horizon-weighted)

At t_n+1 = 1 the target is the teacher's step itself (f(x, 1) = x), which anchors the chain;
consistency between neighbouring grid points then propagates that back to t = 0. The student
starts from the base model's weights. One stage of --cd_epochs epochs; the best epoch is chosen
by val minFDE@5 with --cd_eval_steps Euler steps.

Output: <output_dir>/{best.pt, last.pt, metrics.csv, student_config.json, done}. The checkpoint
stores no fixed step count, so evaluate it at the step counts you want:
    python eval/evaluate.py <run folder> --n_steps 1
    python eval/benchmark.py <run folder> --n_steps 1 2 4 8 16

── Selecting a model ─────────────────────────────────────────────────────────

(--base, and --teacher for an optional different teacher; progressive: round-1 teacher)

    4_256                            shorthand for finished_ckpt/2_architecture/distill_4_256
                                       (falls back to the older finished_ckpt/SWi_DiT/ layout)
    teacher                            the default teacher (configs/teacher_dit_cfm_20M.json)
    finished_ckpt/extra_rope/rope_ts_4L_256     any run folder; best.pt (--ckpt_name) is found inside,
                                       split / compressed checkpoints are joined
    path/to/some.pt                    a checkpoint file

The architecture comes from --base_config / --teacher_config if given, else
student_config.json next to the checkpoint, else the config stored inside it.

Progressive output layout (default <output_dir> = finished_ckpt/5_stepwise_distillation/progressive_<base>/model;
each round folder stores its step count, which eval/ uses automatically):

    <output_dir>/steps_16/{best.pt, last.pt, metrics.csv, student_config.json, done}
    <output_dir>/steps_08/...

Resume: finished rounds (with a `done` file) are skipped and their best.pt is
loaded as the next teacher; an unfinished round resumes from its last.pt.
"""

import argparse
import csv
import json
import os
import re
import time
import torch
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from DataLoaders.ADSBdataset import get_dataloaders
from models.registry import build_model, model_weights, maybe_amp
from models.cfm import forward_cfm, euler_sample
from eval.evaluate import load_checkpoint
from training.loss_weighting import turn_weights, feature_weights, feature_stats, TURN_MODES
from training.train_distill import (
    EMA, set_seed, resolve_path, load_config, weighted_mse, time_weights,
)

CHECKPOINT_DIR = ROOT / "training" / "checkpoints"
TEACHER_CONFIG = "configs/teacher_dit_cfm_20M.json"
# Where the original teacher lives locally and on Savio

METRIC_FIELDS = [
    "epoch", "lr", "epoch_time_s", "train_loss", "val_loss", "val_minfde",
]


# ── Model selection ───────────────────────────────────────────────────────

def run_label(spec):
    """Short name for a --base spec, used in the default output folder."""
    if spec == "teacher":
        return load_config(TEACHER_CONFIG).get("name", "teacher")
    path = Path(spec)
    if path.name.endswith(".pt"):
        path = path.parent
    if path.name == "model":                      # .../<run>/model → <run>
        path = path.parent
    name = path.name.removesuffix("_SWiDiT").removeprefix("distill_")
    if re.fullmatch(r"\d+_\d+_\d+", name):        # older teacher_layers_width shorthand
        name = name.split("_", 1)[1]
    return name


def resolve_shorthand(spec):
    """4_256 (layers_width) → finished_ckpt/2_architecture/distill_4_256. The older
    teacher_layers_width form (6_4_256) is accepted too: the leading teacher depth is dropped."""
    if spec.count("_") == 2:
        spec = spec.split("_", 1)[1]
    for cand in (f"finished_ckpt/2_architecture/distill_{spec}",):
        if resolve_path(cand).exists():
            return cand
    sys.exit(f"Shorthand {spec}: no finished_ckpt/2_architecture/distill_{spec} found")


def load_model_spec(spec, config_path, ckpt_name, device):
    """
    Resolve a model spec (see module docstring) to a frozen TrajectoryDiT with EMA
    weights. Returns (model, config dict, checkpoint path).
    """
    if spec == "teacher":
        config_path = config_path or TEACHER_CONFIG
        ckpt_path   = resolve_path(load_config(config_path).get("checkpoint", ""))
        if not ckpt_path.is_file():
            sys.exit(f"Teacher checkpoint not found: {ckpt_path} (named in {config_path})")
        ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        if re.fullmatch(r"\d+_\d+(_\d+)?", spec):           # e.g. 4_256
            spec = resolve_shorthand(spec)
        path = resolve_path(spec)
        if path.is_file():
            ckpt_path = path
            ckpt      = torch.load(path, map_location=device, weights_only=False)
        elif path.is_dir():
            ckpt, ckpt_path = load_checkpoint(path, ckpt_name, device)
        else:
            sys.exit(f"Model not found: {spec}")

    # Architecture: explicit config > student_config.json beside it > copy in the checkpoint
    if config_path:
        cfg = load_config(config_path)
    elif (ckpt_path.parent / "student_config.json").exists():
        cfg = load_config(ckpt_path.parent / "student_config.json")
    elif ckpt.get("student_config"):
        cfg = ckpt["student_config"]
    else:
        sys.exit(f"No architecture for {ckpt_path}: pass a config JSON")

    model = build_model(cfg).to(device)
    model.load_state_dict(model_weights(ckpt))
    model.eval()
    model.requires_grad_(False)
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch')}, val_fde {ckpt.get('val_fde')}) | "
          f"{cfg['model']} | {sum(p.numel() for p in model.parameters()):,} params")
    return model, cfg, ckpt_path


def frozen_copy(cfg, state, device):
    model = build_model(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    return model


# ── Distillation pieces ───────────────────────────────────────────────────

def round_schedule(start_steps, min_steps):
    """
    (teacher steps, student steps) per round, e.g. 32 → [(32, 16), (16, 8), ..., (2, 1)].
    Stops at an odd count (20 → 10 → 5), since a student step must cover exactly two teacher steps.
    """
    rounds = []
    steps  = start_steps
    while steps % 2 == 0 and steps // 2 >= min_steps:
        rounds.append((steps, steps // 2))
        steps //= 2
    return rounds


def two_step_target(teacher, obs, x_t, t, t_rel, dt):
    """Velocity that moves x_t to where two teacher Euler steps of size dt end up."""
    v1   = teacher(obs, x_t, t, t_rel)
    x_md = x_t + v1 * dt
    v2   = teacher(obs, x_md, t + dt, t_rel)
    return 0.5 * (v1 + v2)


def sample_grid_time(B, n_steps, device):
    """t uniformly from the student's Euler grid {0, 1/N, ..., (N-1)/N}."""
    k = torch.randint(0, n_steps, (B,), device=device)
    return k.float() / n_steps


def validate_fde(model, val_loader, device, feat_std, seed, n_steps, max_batches=20):
    """minFDE@5 in metres with an n_steps Euler sampler."""
    model.eval()
    torch.manual_seed(seed)
    feat_std_xy = torch.tensor(feat_std[:2], device=device, dtype=torch.float32)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= max_batches:
                break
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)
            preds = euler_sample(model, obs, t_rel, n_samples=5,
                                 n_steps=n_steps, device=str(device))
            fde   = torch.norm(preds[:, :, -1, :2] * feat_std_xy
                               - fut[None, :, -1, :2] * feat_std_xy, dim=-1)
            total += fde.min(dim=0).values.mean().item()
            n     += 1
    return total / n


def validate_loss(student, teacher, val_loader, device, seed, n_steps, max_batches=20):
    student.eval()
    torch.manual_seed(seed)
    dt = 0.5 / n_steps
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= max_batches:
                break
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)
            t         = sample_grid_time(obs.shape[0], n_steps, device)
            x_t, _, _ = forward_cfm(fut, t)
            v_tgt     = two_step_target(teacher, obs, x_t, t, t_rel, dt)
            total    += weighted_mse(student(obs, x_t, t, t_rel), v_tgt).item()
            n        += 1
    return total / n


# ── Consistency distillation pieces ──────────────────────────────────────

CD_METRIC_FIELDS = [
    "epoch", "lr", "epoch_time_s", "train_loss", "val_loss", "val_minfde",
]


def consistency_fn(model, obs, x_t, t, t_rel):
    """f(x_t, t) = x_t + (1 - t)·v(x_t, t): the t = 1 endpoint one Euler step from t lands on."""
    return x_t + (1.0 - t).view(-1, 1, 1) * model(obs, x_t, t, t_rel)


def teacher_ode_step(teacher, obs, x_t, t, t_rel, dt, solver):
    """One teacher ODE step of size dt from (x_t, t): Euler, or Heun (2nd order, 2 calls)."""
    v1 = teacher(obs, x_t, t, t_rel)
    x1 = x_t + v1 * dt
    if solver == "euler":
        return x1
    v2 = teacher(obs, x1, t + dt, t_rel)
    return x_t + 0.5 * (v1 + v2) * dt


def consistency_distance(pred, target, sample_w, kind, huber_c, feat_w=None):
    """Distance between predicted endpoints, both with the standard horizon weighting
    (last quarter ×2). l2 = weighted MSE; huber = pseudo-Huber per window,
    sqrt(||w·(pred - target)²||₁ + c²) - c, as in improved consistency training."""
    if kind == "l2":
        return weighted_mse(pred, target, sample_w, feat_w=feat_w)
    w    = time_weights(pred.shape[1], "step", pred.device)[None, :, None]
    if feat_w is not None:
        w = w * feat_w[None, None, :]
    sq   = ((pred - target) ** 2 * w).sum(dim=(1, 2))                      # (B,)
    loss = torch.sqrt(sq + huber_c ** 2) - huber_c
    if sample_w is not None:
        loss = loss * sample_w
    return loss.mean()


def cd_batch_loss(student, target_model, teacher, obs, fut, t_rel, args, sample_w=None, feat_w=None):
    """Consistency-distillation loss for one batch (see module docstring)."""
    N     = args.cd_n
    n     = torch.randint(0, N, (obs.shape[0],), device=obs.device)
    t     = n.float() / N
    t_nxt = (n + 1).float() / N
    x_t, _, _ = forward_cfm(fut, t)
    with torch.no_grad():
        x_nxt  = teacher_ode_step(teacher, obs, x_t, t, t_rel, 1.0 / N, args.cd_solver)
        target = consistency_fn(target_model, obs, x_nxt, t_nxt, t_rel)
    pred = consistency_fn(student, obs, x_t, t, t_rel)
    return consistency_distance(pred, target, sample_w, args.cd_loss, args.cd_huber_c, feat_w)


def validate_cd_loss(student, target_model, teacher, val_loader, device, args, max_batches=20):
    student.eval()
    torch.manual_seed(args.seed)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= max_batches:
                break
            total += cd_batch_loss(student, target_model, teacher, batch["obs"].to(device),
                                   batch["fut"].to(device), batch["t_rel"].to(device), args).item()
            n += 1
    return total / n


def train_consistency(teacher, student_cfg, init_state, train_loader, val_loader, feat_std,
                      out_dir, device, args, run_info):
    """One consistency-distillation stage. Returns the path of the best checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "student_config.json", "w") as f:
        json.dump(student_cfg, f, indent=2)
    metrics_path = out_dir / "metrics.csv"

    model = build_model(student_cfg).to(device)
    model.load_state_dict(init_state)                     # starts as a copy of the base model
    ema    = EMA(model, decay=args.ema_decay)             # weights that get evaluated / saved
    target = EMA(model, decay=args.cd_target_decay)       # θ⁻: slow copy that produces targets

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.cd_epochs - 1),
                                  eta_min=args.lr * args.min_lr_frac)
    feat_mean_t, feat_std_t = feature_stats(train_loader, device)

    start_epoch, global_step, best_fde = 0, 0, float("inf")
    last_ckpt = out_dir / "last.pt"
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        target.shadow.load_state_dict(ckpt["target_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_fde    = ckpt["best_fde"]
        print(f"  resumed at epoch {start_epoch}")

    k = args.cd_eval_steps
    teacher_fde = validate_fde(teacher, val_loader, device, feat_std, args.seed, args.start_steps)
    init_fde    = validate_fde(model, val_loader, device, feat_std, args.seed, k)
    print(f"  teacher @ {args.start_steps} steps: {teacher_fde:.1f}m | "
          f"student @ {k} step(s) before this run: {init_fde:.1f}m", flush=True)

    for epoch in range(start_epoch, args.cd_epochs):
        torch.manual_seed(args.seed + 7919 + epoch)
        epoch_start = time.time()
        epoch_lr    = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)
            sample_w = turn_weights(obs, fut, feat_mean_t, feat_std_t, args.turn_weight,
                                    mode=args.turn_mode, threshold_deg=args.turn_threshold)
            loss     = cd_batch_loss(model, target.shadow, teacher, obs, fut, t_rel, args,
                                     sample_w, feature_weights(args.z_weight, device=device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if global_step < args.warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * (global_step + 1) / args.warmup_steps

            ema.update(model)
            target.update(model)
            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1
            if args.log_every and n_batches % args.log_every == 0:
                elapsed = time.time() - epoch_start
                print(f"    epoch {epoch+1:03d} | batch {n_batches:,}/{len(train_loader):,} | "
                      f"loss {total_loss / n_batches:.5f} | {n_batches / elapsed:.1f} it/s",
                      flush=True)

        scheduler.step()

        val_loss = validate_cd_loss(ema.shadow, target.shadow, teacher, val_loader, device, args)
        val_fde  = validate_fde(ema.shadow, val_loader, device, feat_std, args.seed, k)
        metrics  = {
            "epoch":        epoch + 1,
            "lr":           f"{epoch_lr:.3e}",
            "train_loss":   f"{total_loss / n_batches:.6f}",
            "val_loss":     f"{val_loss:.6f}",
            "val_minfde":   f"{val_fde:.1f}",
            "epoch_time_s": f"{time.time() - epoch_start:.1f}",
        }
        append_metrics(metrics_path, metrics, CD_METRIC_FIELDS)

        checkpoint = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "ema_state":       ema.shadow.state_dict(),
            "target_state":    target.shadow.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step":     global_step,
            "best_fde":        best_fde,
            "val_fde":         val_fde,
            "method":          "consistency",
            "n_steps":         None,          # any step count works; eval picks it (--n_steps)
            "val_steps":       k,
            "teacher_val_fde": teacher_fde,
            "init_val_fde":    init_fde,
            "student_config":  student_cfg,
            "train_config":    {**vars(args), **run_info},
            "metrics":         metrics,
        }
        mark = ""
        if val_fde < best_fde:
            best_fde = checkpoint["best_fde"] = val_fde
            torch.save(checkpoint, out_dir / "best.pt")
            mark = "  ✓ best"
        torch.save(checkpoint, last_ckpt)

        print(f"  [consistency] epoch {epoch+1:03d} | train {metrics['train_loss']} | "
              f"val {metrics['val_loss']} | minFDE@{k}step {val_fde:.1f}m "
              f"(teacher@{args.start_steps} {teacher_fde:.1f}m, before {init_fde:.1f}m){mark}",
              flush=True)

    (out_dir / "done").write_text(f"{best_fde:.1f}\n")
    return out_dir / "best.pt"


def append_metrics(path, row, fields=None):
    fields   = fields or METRIC_FIELDS
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def train_round(teacher, student_cfg, init_state, teacher_steps, student_steps,
                train_loader, val_loader, feat_std, round_dir, device, args, run_info):
    """One halving round. Returns the path of this round's best checkpoint."""
    round_dir.mkdir(parents=True, exist_ok=True)
    with open(round_dir / "student_config.json", "w") as f:
        json.dump(student_cfg, f, indent=2)
    metrics_path = round_dir / "metrics.csv"

    model = build_model(student_cfg).to(device)
    model.load_state_dict(init_state)              # student starts as a copy of the teacher
    ema   = EMA(model, decay=args.ema_decay)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # cosine over the round, decaying to min_lr_frac · lr instead of 0. It steps at the end
    # of each epoch, so T_max = epochs - 1 makes the last epoch train exactly at the floor.
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_per_round - 1),
                                  eta_min=args.lr * args.min_lr_frac)
    dt        = 0.5 / student_steps                # teacher step size
    feat_mean_t, feat_std_t = feature_stats(train_loader, device)   # for turn weights

    start_epoch, global_step, best_fde = 0, 0, float("inf")
    last_ckpt = round_dir / "last.pt"
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema.shadow.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_fde    = ckpt["best_fde"]
        print(f"  resumed round at epoch {start_epoch}")

    # References: teacher at its own step count, and the undistilled init at the student's
    teacher_fde = validate_fde(teacher, val_loader, device, feat_std, args.seed, teacher_steps)
    init_model  = frozen_copy(student_cfg, init_state, device)
    init_fde    = validate_fde(init_model, val_loader, device, feat_std, args.seed, student_steps)
    del init_model
    print(f"  teacher @ {teacher_steps} steps: {teacher_fde:.1f}m | "
          f"student init @ {student_steps} steps (no distillation): {init_fde:.1f}m", flush=True)

    for epoch in range(start_epoch, args.epochs_per_round):
        torch.manual_seed(args.seed + 1000 * student_steps + epoch)
        epoch_start = time.time()
        epoch_lr    = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            obs   = batch["obs"].to(device)
            fut   = batch["fut"].to(device)
            t_rel = batch["t_rel"].to(device)

            t         = sample_grid_time(obs.shape[0], student_steps, device)
            x_t, _, _ = forward_cfm(fut, t)
            with torch.no_grad():
                v_tgt = two_step_target(teacher, obs, x_t, t, t_rel, dt)

            sample_w = turn_weights(obs, fut, feat_mean_t, feat_std_t, args.turn_weight,
                                    mode=args.turn_mode, threshold_deg=args.turn_threshold)
            loss     = weighted_mse(model(obs, x_t, t, t_rel), v_tgt, sample_w,
                                    feat_w=feature_weights(args.z_weight, device=device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if global_step < args.warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * (global_step + 1) / args.warmup_steps

            ema.update(model)
            total_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if args.log_every and n_batches % args.log_every == 0:
                elapsed = time.time() - epoch_start
                print(f"    epoch {epoch+1:03d} | batch {n_batches:,}/{len(train_loader):,} | "
                      f"loss {total_loss / n_batches:.5f} | {n_batches / elapsed:.1f} it/s",
                      flush=True)

        scheduler.step()

        val_loss = validate_loss(ema.shadow, teacher, val_loader, device, args.seed, student_steps)
        val_fde  = validate_fde(ema.shadow, val_loader, device, feat_std, args.seed, student_steps)
        metrics  = {
            "epoch":        epoch + 1,
            "lr":           f"{epoch_lr:.3e}",
            "train_loss":   f"{total_loss / n_batches:.6f}",
            "val_loss":     f"{val_loss:.6f}",
            "val_minfde":   f"{val_fde:.1f}",
            "epoch_time_s": f"{time.time() - epoch_start:.1f}",
        }
        append_metrics(metrics_path, metrics)

        checkpoint = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "ema_state":       ema.shadow.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "global_step":     global_step,
            "best_fde":        best_fde,
            "val_fde":         val_fde,
            "n_steps":         student_steps,
            "teacher_steps":   teacher_steps,
            "teacher_val_fde": teacher_fde,
            "init_val_fde":    init_fde,
            "student_config":  student_cfg,
            "train_config":    {**vars(args), **run_info},
            "metrics":         metrics,
        }
        mark = ""
        if val_fde < best_fde:
            best_fde = checkpoint["best_fde"] = val_fde
            torch.save(checkpoint, round_dir / "best.pt")
            mark = "  ✓ best"
        torch.save(checkpoint, last_ckpt)

        print(f"  [{student_steps} steps] epoch {epoch+1:03d} | train {metrics['train_loss']} | "
              f"val {metrics['val_loss']} | minFDE {val_fde:.1f}m "
              f"(teacher@{teacher_steps} {teacher_fde:.1f}m, init {init_fde:.1f}m){mark}",
              flush=True)

    (round_dir / "done").write_text(f"{best_fde:.1f}\n")
    return round_dir / "best.pt"


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir or
                      ROOT / "finished_ckpt" / "5_stepwise_distillation" / f"{args.method}_{run_label(args.base)}" / "model")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    train_loader, val_loader, _ = get_dataloaders(
        args.nc_path, batch_size=args.batch_size, seed=args.split_seed,
        subset=args.subset, num_workers=num_workers,
    )
    import netCDF4 as nc
    ds       = nc.Dataset(args.nc_path, "r")
    feat_std = np.array(ds.feature_std)
    ds.close()

    print(f"{args.method.capitalize()} distillation on {device} → {output_dir}")
    print("Base (student init):", end=" ")
    base_model, base_cfg, base_ckpt = load_model_spec(
        args.base, args.base_config, args.ckpt_name, device)
    init_state = base_model.state_dict()

    # Teacher (progressive: round 1): the base model unless --teacher picks another one
    if args.teacher:
        print("Teacher:", end=" ")
        teacher, teacher_cfg, teacher_ckpt = load_model_spec(
            args.teacher, args.teacher_config, args.ckpt_name, device)
        teacher = maybe_amp(teacher, teacher_cfg)      # e.g. the 20M DiT: fp16 autocast on CUDA
    else:
        teacher, teacher_ckpt = base_model, base_ckpt
        print("Teacher: the base model")
    run_info = {"base_ckpt": str(base_ckpt), "round1_teacher_ckpt": str(teacher_ckpt),
                "turn_weight": args.turn_weight}

    if args.method == "consistency":
        if (output_dir / "done").exists():
            print(f"Already done (best minFDE {(output_dir / 'done').read_text().strip()}m)")
            return
        print(f"\n=== Consistency distillation: N={args.cd_n}, {args.cd_solver} teacher steps, "
              f"{args.cd_loss} loss, target decay {args.cd_target_decay} ===", flush=True)
        train_consistency(teacher, base_cfg, init_state, train_loader, val_loader, feat_std,
                          output_dir, device, args, run_info)
        return

    for teacher_steps, student_steps in round_schedule(args.start_steps, args.min_steps):
        round_dir = output_dir / f"steps_{student_steps:02d}"
        print(f"\n=== Round: {teacher_steps} → {student_steps} steps ({round_dir}) ===", flush=True)

        if (round_dir / "done").exists():
            best_path = round_dir / "best.pt"
            print(f"  already done (best minFDE {(round_dir / 'done').read_text().strip()}m), skipping")
        else:
            best_path = train_round(teacher, base_cfg, init_state, teacher_steps, student_steps,
                                    train_loader, val_loader, feat_std, round_dir, device,
                                    args, run_info)

        # Best EMA student becomes the next teacher and the next student's init
        best       = torch.load(best_path, map_location=device, weights_only=False)
        teacher    = frozen_copy(base_cfg, best["ema_state"], device)
        init_state = teacher.state_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_path", type=str, required=True,
                        help="Path to the .nc trajectory dataset")
    parser.add_argument("--method", choices=("progressive", "consistency"), default="progressive",
                        help="progressive: halve the step count per round; consistency: one stage "
                             "trained to jump to the end of the teacher's path from any point")
    parser.add_argument("--base", type=str, default="4_256",
                        help="Model whose sampler is distilled: shorthand (4_256), 'teacher', "
                             "a run folder or a .pt file")
    parser.add_argument("--base_config", type=str, default=None,
                        help="Architecture JSON for --base (default: found next to / inside the checkpoint)")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Optional different round-1 teacher (same spec format as --base); "
                             "default: the base model itself")
    parser.add_argument("--teacher_config", type=str, default=None,
                        help="Architecture JSON for --teacher")
    parser.add_argument("--ckpt_name", type=str, default="best.pt",
                        help="Checkpoint to use inside run folders (best.pt or last.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: finished_ckpt/5_stepwise_distillation/<method>_<base>/model")
    parser.add_argument("--start_steps", type=int, default=32,
                        help="Euler steps of the round-1 teacher; halved every round "
                             "(consistency: only the teacher's reference minFDE uses it)")
    parser.add_argument("--min_steps", type=int, default=1,
                        help="Stop once the student reaches this many steps")
    parser.add_argument("--epochs_per_round", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak LR of every round (after warmup)")
    parser.add_argument("--min_lr_frac", type=float, default=0.2,
                        help="Floor of the per-round cosine as a fraction of --lr (0.2 → decays to 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=250,
                        help="Linear LR warmup at the start of every round")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--turn_weight", type=float, default=0.0,
                        help="Extra loss weight S for turning windows (0 = off); see --turn_mode")
    parser.add_argument("--turn_mode", choices=TURN_MODES, default="continuous",
                        help="continuous: 1 + S*min(turn/90deg, 1); step: 1 + S past --turn_threshold")
    parser.add_argument("--turn_threshold", type=float, default=45.0,
                        help="Degrees of turn that count as turning for --turn_mode step")
    parser.add_argument("--z_weight", type=float, default=1.0,
                        help="Loss weight of the altitude features (z, vz) relative to the others (1 = off)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds the training randomness (batch order, noise, flow times)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seeds the aircraft-level train/val/test split; must match the base "
                             "model's (42) so its test aircraft stay unseen")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers; default = $SLURM_CPUS_PER_TASK, or 4")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit to N samples; omit for the full dataset")
    parser.add_argument("--device", type=str, default="cuda")
    cd = parser.add_argument_group("consistency distillation (--method consistency)")
    cd.add_argument("--cd_epochs", type=int, default=20,
                    help="Epochs of the single consistency stage")
    cd.add_argument("--cd_n", type=int, default=32,
                    help="Grid intervals N on t in [0, 1]: each training pair is one teacher step "
                         "of 1/N apart. Larger N = smaller, more accurate teacher steps but a longer "
                         "chain for consistency to propagate along")
    cd.add_argument("--cd_solver", choices=("heun", "euler"), default="heun",
                    help="Teacher ODE step between neighbouring grid points: heun (2nd order, 2 "
                         "teacher calls) or euler (1 call, cheaper, less accurate)")
    cd.add_argument("--cd_target_decay", type=float, default=0.95,
                    help="EMA decay mu of the target network theta- that produces the targets. "
                         "Higher = steadier targets but slower learning; 0 = the student itself "
                         "(stop-gradient), as in improved consistency training")
    cd.add_argument("--cd_loss", choices=("l2", "huber"), default="l2",
                    help="Distance between predicted endpoints: l2 (horizon-weighted MSE, as the "
                         "other losses) or huber (pseudo-Huber per window, less hurt by outliers)")
    cd.add_argument("--cd_huber_c", type=float, default=0.00054 * (43 * 6) ** 0.5,
                    help="Pseudo-Huber c (default 0.00054*sqrt(dims) ≈ 0.0087, as in iCT)")
    cd.add_argument("--cd_eval_steps", type=int, default=1,
                    help="Euler steps used for the per-epoch val minFDE that picks best.pt")
    main(parser.parse_args())
