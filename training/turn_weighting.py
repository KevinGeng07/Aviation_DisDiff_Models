"""
Turn weighting: give windows whose true future turns a larger share of the loss.

Turning windows are a small fraction of the data (most flight is straight), so a
plain MSE is dominated by straight segments and the models learn to under-turn.

How much a window turns is measured from the velocity features (vx = east, vy = north):

    heading_k = atan2(vx_k, vy_k)                       (k = last observed step, future steps)
    turn      = max_k |wrap(heading_k - heading_last_obs)|   in degrees, 0..180

The max deviation from the last observed heading (not the sum of step-to-step
changes) so heading jitter doesn't add up on straight flights. Steps slower than
min_speed (taxiing / on the ground) don't count, since their heading is noise.

Per-window loss weight:

    w = 1 + strength · min(turn / ref_deg, 1)          then divided by the batch mean

so with strength 2, a window turning >= 90° counts 3× a straight one. Dividing by
the batch mean keeps the loss (and so the effective learning rate) the same size.
strength 0 turns this off.
"""
import torch

VX, VY = 3, 4   # feature order: x, y, z, vx, vy, vz


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


def turn_weights(obs, fut, feat_mean, feat_std, strength, ref_deg=90.0):
    """Per-window loss weights with batch mean 1, or None if strength is 0."""
    if not strength:
        return None
    turn = turn_angle(obs, fut, feat_mean, feat_std)
    w    = 1.0 + strength * (turn / ref_deg).clamp(max=1.0)
    return w / w.mean()


def feature_stats(loader, device):
    """feature_mean, feature_std of the dataset behind a DataLoader, as tensors on device."""
    stats = loader.dataset.stats
    return (torch.tensor(stats["feature_mean"], device=device, dtype=torch.float32),
            torch.tensor(stats["feature_std"],  device=device, dtype=torch.float32))
