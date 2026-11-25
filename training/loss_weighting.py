"""
Loss weighting (pipeline stage 4, path_weighting): give turning windows and/or the altitude axis a larger
share of the training loss. All weights are rescaled to mean 1, so the loss (and so the
effective learning rate) stays the same size; strength 0 turns a weighting off.

── Turn weighting (per window) ───────────────────────────────────────────────

Turning windows are a small fraction of the data (most flight is straight), so a plain MSE
is dominated by straight segments and the models learn to under-turn. How much a window
turns is measured from the velocity features (vx = east, vy = north):

    heading_k = atan2(vx_k, vy_k)                       (k = last observed step, future steps)
    turn      = max_k |wrap(heading_k - heading_last_obs)|   in degrees, 0..180

The max deviation from the last observed heading (not the sum of step-to-step changes) so
heading jitter doesn't add up on straight flights. Steps slower than min_speed (taxiing /
on the ground) don't count, since their heading is noise.

    continuous (default)  w = 1 + S · min(turn / ref_deg, 1)       ramps up to ref_deg (90°)
    step                  w = 1 + S · [turn >= threshold_deg]       on/off at the threshold

then divided by the batch mean. With S = 2, a window turning >= 90° (continuous) or past
the threshold (step) counts 3× a straight one.

── Vertical-rate weighting (per window, added to the turn weight) ────────────

The vertical counterpart of a turn: how far the vertical rate vz (m/s) moves away from its
last observed value in the true future (a level-off, start of descent, go-around):

    dvz = max_k |vz_k - vz_last_obs|                               m/s, k = future steps
    v   = clamp((dvz - deadband) / (ref - deadband), 0, 1)         0..1

deadband (default 1 m/s ≈ 200 ft/min) ignores ADS-B rate jitter on level flight; ref
(default 10 m/s ≈ 2000 ft/min, a full level-off from a normal climb) is where the weight
saturates. Same moving-only rule as the turn angle. Combined with the turn weight:

    w = 1 + S · h + S_v · v      h = min(turn / 90°, 1) (continuous) or [turn >= threshold] (step)

then divided by the batch mean. S_v = 0 (default) turns the vertical part off.

── Z weighting (per feature) ─────────────────────────────────────────────────

Altitude is a small part of the error in metres but matters operationally. --z_weight W
weights the altitude features (z and vz) W× the horizontal ones (x, y, vx, vy), with the
six weights rescaled to mean 1. W = 1 is off.
"""
import torch

VX, VY, VZ = 3, 4, 5           # feature order: x, y, z, vx, vy, vz
Z_FEATURES = (2, 5)            # z, vz
TURN_MODES = ("continuous", "step")


def turn_angle(obs, fut, feat_mean, feat_std, min_speed=20.0):
    """Degrees the true future turns away from the last observed heading: (B,)."""
    vel = torch.cat([obs[:, -1:, VX:VY + 1], fut[:, :, VX:VY + 1]], dim=1)   # normalized
    vel = vel * feat_std[VX:VY + 1] + feat_mean[VX:VY + 1]                    # m/s
    heading = torch.atan2(vel[..., 0], vel[..., 1])                         # (B, 1 + 43)
    diff    = heading[:, 1:] - heading[:, :1]
    diff    = torch.atan2(torch.sin(diff), torch.cos(diff)).abs()           # wrap to [0, pi]
    moving  = (vel.norm(dim=-1) >= min_speed)
    diff    = diff * (moving[:, 1:] & moving[:, :1])
    return diff.amax(dim=1).rad2deg()


def vrate_change(obs, fut, feat_mean, feat_std, min_speed=20.0):
    """m/s the true future vertical rate moves away from the last observed one: (B,)."""
    feats = torch.cat([obs[:, -1:], fut], dim=1)                             # normalized
    vel   = feats[..., VX:VZ + 1] * feat_std[VX:VZ + 1] + feat_mean[VX:VZ + 1]   # m/s
    dvz   = (vel[:, 1:, 2] - vel[:, :1, 2]).abs()
    moving = vel[..., :2].norm(dim=-1) >= min_speed                         # horizontal speed
    dvz   = dvz * (moving[:, 1:] & moving[:, :1])
    return dvz.amax(dim=1)


def turn_weights(obs, fut, feat_mean, feat_std, strength, mode="continuous",
                 ref_deg=90.0, threshold_deg=45.0,
                 vrate_strength=0.0, vrate_ref=10.0, vrate_deadband=1.0):
    """Per-window loss weights with batch mean 1, or None if both strengths are 0.
    w = 1 + strength * h(turn) + vrate_strength * v(vertical-rate change)."""
    if not strength and not vrate_strength:
        return None
    w = torch.ones(obs.shape[0], device=obs.device)
    if strength:
        turn = turn_angle(obs, fut, feat_mean, feat_std)
        if mode == "continuous":
            w = w + strength * (turn / ref_deg).clamp(max=1.0)
        elif mode == "step":
            w = w + strength * (turn >= threshold_deg).float()
        else:
            raise ValueError(f"unknown turn mode {mode!r}; known: {', '.join(TURN_MODES)}")
    if vrate_strength:
        dvz = vrate_change(obs, fut, feat_mean, feat_std)
        w = w + vrate_strength * ((dvz - vrate_deadband) / (vrate_ref - vrate_deadband)).clamp(0.0, 1.0)
    return w / w.mean()


def feature_weights(z_weight, in_dim=6, device=None):
    """Per-feature loss weights (in_dim,) with mean 1, or None if z_weight is 1 (off)."""
    if z_weight is None or z_weight == 1.0:
        return None
    w = torch.ones(in_dim, device=device)
    w[list(Z_FEATURES)] = z_weight
    return w / w.mean()


def feature_stats(loader, device):
    """feature_mean, feature_std of the dataset behind a DataLoader, as tensors on device."""
    stats = loader.dataset.stats
    return (torch.tensor(stats["feature_mean"], device=device, dtype=torch.float32),
            torch.tensor(stats["feature_std"],  device=device, dtype=torch.float32))
