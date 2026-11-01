"""Train the pose tokenizer with detector confidence baked into each token.

This variant keeps the legacy coordinate representation but changes the model
feature layout from ``(y, x)`` per joint to ``(y, x, confidence)`` per joint:

    input_features = NUM_JOINTS * 3 = 165

Coordinate reconstruction and velocity losses are computed only on the first
two channels. The confidence channel is supervised separately, so decoded tokens
carry visibility/observation-quality information for downstream conditioning.

Run pattern:
    accelerate launch train_confidence.py --config configs/final_8cb_confidence.yaml
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import PoseTokenizerConfig, TrainConfig
from pose_tokenizer.data.kinematic import NUM_JOINTS
from pose_tokenizer.data.rendering import render_comparison_video
from pose_tokenizer.training import (
    bootstrap_run,
    build_dataloaders,
    build_wandb_config,
    load_checkpoint,
    load_model_weights,
    log_train_step,
    parse_args,
    resolve_config_path,
    save_checkpoint,
)

torch.manual_seed(42)

CONF_FEATURES = NUM_JOINTS * 3
COORD_DIMS = 2

_LEFT_HAND = list(range(13, 34))
_RIGHT_HAND = list(range(34, 55))
_CORRUPTION_GROUPS: dict[str, list[int]] = {
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "both_shoulders": [5, 6],
    "left_shoulder_chain": [5, 7, 9],
    "right_shoulder_chain": [6, 8, 10],
    "left_hand": [9, *_LEFT_HAND],
    "right_hand": [10, *_RIGHT_HAND],
    "left_forearm_hand": [7, 9, *_LEFT_HAND],
    "right_forearm_hand": [8, 10, *_RIGHT_HAND],
    "both_hands": [9, 10, *_LEFT_HAND, *_RIGHT_HAND],
    "upper_body": list(range(5, 13)),
}


def _model_cfg(model: torch.nn.Module) -> PoseTokenizerConfig:
    return model.module.config if hasattr(model, "module") else model.config


def _split_features(features: torch.Tensor, n_joints: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(xy, conf)`` from interleaved ``[y, x, conf]`` features."""
    shaped = features.view(features.shape[0], features.shape[1], n_joints, 3)
    return shaped[..., :2], shaped[..., 2]


def _join_features(xy: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    return torch.cat([xy, conf.unsqueeze(-1)], dim=-1).flatten(start_dim=2)


def _randint_inclusive(low: int, high: int, *, device: torch.device) -> int:
    if high <= low:
        return low
    return int(torch.randint(low, high + 1, (), device=device).item())


def _sample_corruption_group(
    gt_conf_b: torch.Tensor,
    *,
    min_conf: float,
    device: torch.device,
) -> torch.Tensor | None:
    names = list(_CORRUPTION_GROUPS)
    for _ in range(8):
        name = names[int(torch.randint(len(names), (), device=device).item())]
        joints = torch.tensor(_CORRUPTION_GROUPS[name], device=device, dtype=torch.long)
        if (gt_conf_b[:, joints] >= min_conf).any():
            return joints
    return None

def _apply_pose_corruption(
    gt_xy: torch.Tensor,
    gt_conf: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Corrupt model inputs while keeping clean targets for denoising training."""
    if cfg.pose_corruption_prob <= 0.0:
        zero = torch.zeros((), device=gt_xy.device)
        return gt_xy, gt_conf, {"corrupt_joint_frac": zero, "corrupt_clip_frac": zero}

    device = gt_xy.device
    input_xy = gt_xy.clone()
    input_conf = gt_conf.clone()
    mask = torch.zeros_like(gt_conf, dtype=torch.bool)
    clip_mask = torch.zeros(gt_conf.shape[0], device=device, dtype=torch.bool)
    modes = list(cfg.pose_corruption_modes) or ["temporal_joint_conf_drop"]
    allowed_modes = {
        "temporal_joint_conf_drop",
        "part_conf_drop",
        "temporal_joint_flicker",
        "part_flicker",
    }
    unknown_modes = sorted(set(modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(
            f"Unknown pose_corruption_modes={unknown_modes}; expected one or more of {sorted(allowed_modes)}"
        )
    min_conf = float(cfg.pose_corruption_min_conf)

    batch_size, n_frames, n_joints = gt_conf.shape
    for b in range(batch_size):
        if torch.rand((), device=device) >= cfg.pose_corruption_prob:
            continue

        mode = modes[int(torch.randint(len(modes), (), device=device).item())]
        is_part_mode = mode.startswith("part_")
        if is_part_mode:
            joints = _sample_corruption_group(gt_conf[b], min_conf=min_conf, device=device)
            if joints is None:
                continue
        else:
            candidates = torch.nonzero((gt_conf[b] >= min_conf).any(dim=0), as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            n_pick = _randint_inclusive(
                cfg.pose_corruption_joints_min,
                cfg.pose_corruption_joints_max,
                device=device,
            )
            n_pick = min(n_pick, int(candidates.numel()))
            perm = torch.randperm(candidates.numel(), device=device)[:n_pick]
            joints = candidates[perm]

        span = _randint_inclusive(
            cfg.pose_corruption_span_min,
            cfg.pose_corruption_span_max,
            device=device,
        )
        span = max(1, min(span, n_frames))
        start = _randint_inclusive(0, n_frames - span, device=device)
        end = start + span

        high_conf = gt_conf[b, start:end, joints] >= min_conf
        if not high_conf.any():
            continue
        mask[b, start:end, joints] |= high_conf
        clip_mask[b] = True

    if not mask.any():
        zero = torch.zeros((), device=device)
        return input_xy, input_conf, {"corrupt_joint_frac": zero, "corrupt_clip_frac": zero}

    conf_min = float(cfg.pose_corruption_conf_min)
    conf_max = max(conf_min, float(cfg.pose_corruption_conf_max))
    input_conf[mask] = torch.empty_like(input_conf[mask]).uniform_(conf_min, conf_max)

    noise_std = float(cfg.pose_corruption_xy_noise_std)
    outlier_prob = float(cfg.pose_corruption_xy_outlier_prob)
    outlier_std = float(cfg.pose_corruption_xy_outlier_std)
    if noise_std > 0.0:
        input_xy[mask] = input_xy[mask] + torch.randn_like(input_xy[mask]) * noise_std
    if outlier_prob > 0.0 and outlier_std > 0.0:
        outlier_mask = mask & (torch.rand_like(gt_conf) < outlier_prob)
        if outlier_mask.any():
            input_xy[outlier_mask] = (
                input_xy[outlier_mask] + torch.randn_like(input_xy[outlier_mask]) * outlier_std
            )
    input_xy = input_xy.clamp(-10.0, 10.0)

    stats = {
        "corrupt_joint_frac": mask.float().mean().detach(),
        "corrupt_clip_frac": clip_mask.float().mean().detach(),
    }
    return input_xy, input_conf, stats


def _sigmoid_if_needed(raw_conf: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if cfg.confidence_loss_type.lower() == "bce":
        return raw_conf.sigmoid()
    return raw_conf.clamp(0.0, 1.0)


def build_joint_weights(
    model_cfg: PoseTokenizerConfig,
    hand_loss_weight: float,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Per-joint weights normalised to mean=1 over joints."""
    n_body_kp = model_cfg.num_keypoints - 42
    w = torch.ones(model_cfg.num_keypoints, device=device)
    w[n_body_kp:] = hand_loss_weight
    return w * (w.numel() / w.sum())


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "codebook" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def apply_freeze_config(model: torch.nn.Module, cfg: TrainConfig, accelerator) -> None:
    """Freeze selected model submodules before optimizer construction."""
    frozen: list[str] = []
    if cfg.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        frozen.append("encoder")
    if cfg.freeze_quantizer and model.quantizer is not None:
        for p in model.quantizer.parameters():
            p.requires_grad_(False)
        frozen.append("quantizer")
    if cfg.freeze_decoder:
        for p in model.decoder.parameters():
            p.requires_grad_(False)
        frozen.append("decoder")
    if frozen:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        accelerator.print(
            f"Frozen modules: {', '.join(frozen)}; "
            f"trainable parameters: {trainable:,}/{total:,} ({100.0 * trainable / total:.1f}%)"
        )


def build_scheduler(optimizer, cfg: TrainConfig, *, start_step: int):
    if cfg.lr_schedule == "constant":
        return None
    if cfg.max_steps <= 0:
        raise ValueError(
            f"lr_schedule={cfg.lr_schedule!r} requires max_steps > 0 to define the horizon."
        )
    last_epoch = start_step - 1
    if last_epoch >= 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    if cfg.lr_schedule == "cosine":
        warmup = max(0, cfg.warmup_steps)
        if warmup >= cfg.max_steps:
            raise ValueError(f"warmup_steps={warmup} must be < max_steps={cfg.max_steps}")
        ratio_min = cfg.lr_min / cfg.lr if cfg.lr > 0 else 0.0
        cosine_steps = cfg.max_steps - warmup

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, cosine_steps)
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return ratio_min + (1.0 - ratio_min) * cos

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")


def _confidence_loss(
    pred_conf_raw: torch.Tensor,
    gt_conf: torch.Tensor,
    joint_weights: torch.Tensor,
    cfg: TrainConfig,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = cfg.confidence_loss_type.lower()
    if mode == "bce":
        per = F.binary_cross_entropy_with_logits(pred_conf_raw, gt_conf, reduction="none")
    elif mode == "smooth_l1":
        per = F.smooth_l1_loss(pred_conf_raw.clamp(0.0, 1.0), gt_conf, reduction="none")
    elif mode == "l1":
        per = (pred_conf_raw.clamp(0.0, 1.0) - gt_conf).abs()
    else:
        raise ValueError(
            f"confidence_loss_type must be one of {{'bce','smooth_l1','l1'}}, "
            f"got {cfg.confidence_loss_type!r}"
        )
    weights = joint_weights.view(1, 1, -1).expand_as(per)
    if frame_mask is not None:
        weights = weights * frame_mask.unsqueeze(-1)
    return (per * weights).sum() / weights.sum().clamp(min=1.0)


def _coord_conf_weights(gt_conf: torch.Tensor, cfg: TrainConfig, dtype: torch.dtype):
    mode = cfg.confidence_weight_mode.lower()
    if mode == "hard":
        return (gt_conf >= cfg.confidence_mask_threshold).to(dtype)
    if mode == "soft":
        return gt_conf.to(dtype)
    if mode == "none":
        return None
    raise ValueError(
        f"confidence_weight_mode must be one of {{'none','soft','hard'}}, "
        f"got {cfg.confidence_weight_mode!r}"
    )


def _combine_frame_mask(
    conf_weights: torch.Tensor | None,
    frame_mask: torch.Tensor | None,
    n_joints: int,
) -> torch.Tensor | None:
    """Fold a per-frame (B, T) supervision mask into the (B, T, J) coord weights.

    Returns ``conf_weights`` unchanged when there is no frame mask, a broadcast
    of the frame mask when there are no confidence weights, or their product.
    """
    if frame_mask is None:
        return conf_weights
    fm = frame_mask.unsqueeze(-1)  # (B, T, 1)
    if conf_weights is None:
        return fm.expand(-1, -1, n_joints)
    return conf_weights * fm


def _weighted_xy_loss(
    per_xy: torch.Tensor,
    joint_weights: torch.Tensor,
    conf_weights: torch.Tensor | None,
) -> torch.Tensor:
    weights = joint_weights.view(1, 1, -1, 1)
    per_xy = per_xy * weights
    if conf_weights is None:
        return per_xy.mean()
    mask = conf_weights.unsqueeze(-1).expand_as(per_xy)
    return (per_xy * mask).sum() / mask.sum().clamp(min=1.0)


def compute_losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: TrainConfig,
    joint_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mc = _model_cfg(model)
    raw_feats = batch["keypoints"]
    gt_xy, gt_conf = _split_features(raw_feats, mc.num_keypoints)
    gt_xy = gt_xy.clamp(-10.0, 10.0)
    gt_conf = gt_conf.clamp(0.0, 1.0)
    input_xy, input_conf, corruption_metrics = _apply_pose_corruption(gt_xy, gt_conf, cfg)
    feats = _join_features(input_xy, input_conf)

    out = model(feats)
    # part_rvq may supervise the whole-body loss on only a subset of residual
    # layers (reconstruction_for_loss); other models use the full reconstruction.
    recon_for_loss = out.get("reconstruction_for_loss", out["reconstruction"])
    pred_xy, pred_conf_raw = _split_features(recon_for_loss, mc.num_keypoints)

    # Causal latency budget: the causal model's recon trails the input by a few
    # frames, so grade recon[t] against input[t-shift] (align pred's later frames
    # with the gt's earlier frames). The model sees the full sequence; we only
    # shift the loss target, letting it sharpen at a known fixed delay.
    shift = int(getattr(cfg, "causal_loss_shift", 0) or 0)
    if shift > 0:
        pred_xy = pred_xy[:, shift:]
        pred_conf_raw = pred_conf_raw[:, shift:]
        gt_xy = gt_xy[:, : gt_xy.shape[1] - shift]
        gt_conf = gt_conf[:, : gt_conf.shape[1] - shift]
    # Causal warmup prefix: supervise only the frames flagged by ``loss_mask``
    # (the trailing segment); prefix/padding frames are 0 and contribute no
    # gradient. None => supervise every frame (legacy).
    frame_mask = batch.get("loss_mask")
    if frame_mask is not None:
        frame_mask = frame_mask.to(gt_xy.dtype)
        if shift > 0:
            # Keep the mask aligned with the (front-sliced) gt targets above.
            frame_mask = frame_mask[:, : frame_mask.shape[1] - shift]

    conf_weights = _coord_conf_weights(gt_conf, cfg, gt_xy.dtype)
    coord_weights = _combine_frame_mask(conf_weights, frame_mask, mc.num_keypoints)
    rec_loss = _weighted_xy_loss(
        (pred_xy - gt_xy).abs(),
        joint_weights,
        coord_weights,
    )

    vel_pred = pred_xy[:, 1:, :, :] - pred_xy[:, :-1, :, :]
    vel_true = gt_xy[:, 1:, :, :] - gt_xy[:, :-1, :, :]
    vel_per = (vel_pred - vel_true).abs()
    if conf_weights is not None:
        vel_w = torch.minimum(conf_weights[:, 1:, :], conf_weights[:, :-1, :])
    else:
        vel_w = None
    # A velocity sample is supervised iff its target frame t (the [1:] index) is
    # supervised; the t-1 frame may be real prefix, which is fine to reference.
    vel_w = _combine_frame_mask(
        vel_w, frame_mask[:, 1:] if frame_mask is not None else None, mc.num_keypoints
    )
    vel_loss = _weighted_xy_loss(vel_per, joint_weights, vel_w)

    # Acceleration (2nd-difference) L1: penalises over-smoothing of sharp motion,
    # the lossy downsampling codec's dominant failure mode (error tracks
    # acceleration far more than velocity). A sample's trust is the fuzzy-AND
    # (min) of its three endpoint confidences so it does not learn tracker jitter.
    acc_pred = pred_xy[:, 2:, :, :] - 2.0 * pred_xy[:, 1:-1, :, :] + pred_xy[:, :-2, :, :]
    acc_true = gt_xy[:, 2:, :, :] - 2.0 * gt_xy[:, 1:-1, :, :] + gt_xy[:, :-2, :, :]
    acc_per = (acc_pred - acc_true).abs()
    if conf_weights is not None:
        acc_w = torch.minimum(
            torch.minimum(conf_weights[:, 2:, :], conf_weights[:, 1:-1, :]),
            conf_weights[:, :-2, :],
        )
    else:
        acc_w = None
    # Supervised iff the latest frame involved (the [2:] index) is supervised.
    acc_w = _combine_frame_mask(
        acc_w, frame_mask[:, 2:] if frame_mask is not None else None, mc.num_keypoints
    )
    acc_loss = _weighted_xy_loss(acc_per, joint_weights, acc_w)


    conf_loss = _confidence_loss(pred_conf_raw, gt_conf, joint_weights, cfg, frame_mask)

    # PRQ term (B): per-part coordinate reconstruction L1 (part_rvq only). Each
    # part must be independently decodable, so supervise every part's own xy
    # reconstruction (before aggregation) against its joints' targets, using the
    # same joint/confidence weighting as the whole-body coord loss above.
    part_loss = torch.zeros((), device=rec_loss.device)
    per_part_loss = None
    part_names = None
    if "part_recon" in out and cfg.part_loss_weight != 0.0:
        um = model.module if hasattr(model, "module") else model
        part_names = um.part_names
        pj = um.part_joint_idx                                  # (N, M) long
        N, M = pj.shape
        # part_recon entries are (B, T, M, channels); take xy and align to gt.
        part_pred = torch.stack(out["part_recon"], dim=2)[..., :2]   # (B, T, N, M, 2)
        if shift > 0:
            part_pred = part_pred[:, shift:]
        tgt_parts = gt_xy[:, :, pj, :]                          # (B, T', N, M, 2)
        jw = joint_weights[pj].view(1, 1, N, M, 1)             # (1, 1, N, M, 1)
        part_per = (part_pred - tgt_parts).abs() * jw
        red = (0, 1, 3, 4)
        if coord_weights is not None:
            cw = coord_weights[:, :, pj].unsqueeze(-1).expand_as(part_per)
            num = (part_per * cw).sum(dim=red)
            den = cw.sum(dim=red).clamp(min=1.0)
            per_part_loss = num / den
            part_loss = num.sum() / den.sum().clamp(min=1.0)
        else:
            per_part_loss = part_per.mean(dim=red)
            part_loss = per_part_loss.mean()

    loss = (
        cfg.reconstruction_weight * rec_loss
        + cfg.velocity_weight * vel_loss
        + cfg.acceleration_weight * acc_loss
        + cfg.confidence_loss_weight * conf_loss
        + cfg.part_loss_weight * part_loss
        + cfg.commitment_weight * out["commitment_loss"]
        + cfg.codebook_weight * out["codebook_loss"]
    )

    metrics = {
        "loss": loss,
        "rec_loss": rec_loss,
        "vel_loss": vel_loss,
        "acc_loss": acc_loss,
        "conf_loss": conf_loss,
        "part_loss": part_loss,
        "commitment_loss": out["commitment_loss"],
        "codebook_loss": out["codebook_loss"],
        **corruption_metrics,
    }
    # Per-part reconstruction L1 (part_rvq) so we can see which body parts learn
    # fastest/slowest. 0-dim tensors so the caller's `.item()` works.
    if per_part_loss is not None and part_names is not None:
        for i, name in enumerate(part_names):
            metrics[f"part_loss/{name}"] = per_part_loss[i]
    return loss, metrics, out


@torch.no_grad()
def validate_confidence(
    model,
    val_loader,
    cfg: TrainConfig,
    joint_weights: torch.Tensor,
    *,
    accelerator,
    silent: bool = False,
) -> dict[str, float]:
    model.eval()
    mc = _model_cfg(model)
    device = next((model.module if hasattr(model, "module") else model).parameters()).device
    cb_size = (mc.part_codebook_size
               if getattr(mc, "model_type", "conv") == "part_rvq"
               else mc.codebook_size)

    is_ddp = accelerator is not None and getattr(accelerator, "num_processes", 1) > 1

    def gfm(*tensors):
        return accelerator.gather_for_metrics(tensors) if is_ddp else tensors

    z = lambda: torch.zeros((), device=device)
    loss_sum = z()
    rec_sum = z()
    vel_sum = z()
    acc_sum = z()
    conf_sum = z()
    commit_sum = z()
    cb_sum = z()
    n_batches = z()
    conf_mae_sum = z()
    conf_mae_count = z()
    rec_face = [z(), z()]
    rec_body = [z(), z()]
    rec_hand = [z(), z()]
    # Per-keypoint recon L1 (sum, count) over all confident xy entries -> (J,).
    n_kp = mc.num_keypoints
    rec_kp = [torch.zeros(n_kp, device=device), torch.zeros(n_kp, device=device)]
    # Per-timestep (within-clip) recon L1, accumulated by relative frame index.
    time_sum: torch.Tensor | None = None
    time_cnt: torch.Tensor | None = None
    seen_codes: list[torch.Tensor] | None = None

    n_face = 5
    n_body_end = mc.num_keypoints - 42
    val_thresh = cfg.val_confidence_threshold

    for batch in tqdm(val_loader, desc="Validation", ncols=100, leave=False, disable=silent):
        loss, metrics, out = compute_losses(model, batch, cfg, joint_weights)
        raw_feats = batch["keypoints"]
        gt_xy, gt_conf = _split_features(raw_feats, mc.num_keypoints)
        gt_xy = gt_xy.clamp(-10.0, 10.0)
        pred_xy, pred_conf_raw = _split_features(out["reconstruction"], mc.num_keypoints)
        pred_conf = _sigmoid_if_needed(pred_conf_raw, cfg)
        diff = (pred_xy - gt_xy).abs()
        conf_mae = (pred_conf - gt_conf.clamp(0.0, 1.0)).abs()
        mask = (gt_conf >= val_thresh).to(diff.dtype) if val_thresh > 0 else torch.ones_like(gt_conf)
        # Restrict all metrics to supervised (warm) frames when a prefix is used.
        frame_mask = batch.get("loss_mask")
        if frame_mask is not None:
            fm = frame_mask.to(diff.dtype)
        else:
            fm = torch.ones_like(gt_conf[..., 0])
        mask = mask * fm.unsqueeze(-1)
        conf_mae = conf_mae * fm.unsqueeze(-1)

        (diff_g, mask_g, conf_mae_g, fm_g, loss_g, rec_g, vel_g, acc_g, conf_g, commit_g, cb_g) = gfm(
            diff,
            mask,
            conf_mae,
            fm,
            metrics["loss"].unsqueeze(0),
            metrics["rec_loss"].unsqueeze(0),
            metrics["vel_loss"].unsqueeze(0),
            metrics["acc_loss"].unsqueeze(0),
            metrics["conf_loss"].unsqueeze(0),
            metrics["commitment_loss"].unsqueeze(0),
            metrics["codebook_loss"].unsqueeze(0),
        )

        loss_sum += loss_g.mean()
        rec_sum += rec_g.mean()
        vel_sum += vel_g.mean()
        acc_sum += acc_g.mean()
        conf_sum += conf_g.mean()
        commit_sum += commit_g.mean()
        cb_sum += cb_g.mean()
        n_batches += 1
        conf_mae_sum += conf_mae_g.sum()
        conf_mae_count += fm_g.sum() * conf_mae_g.shape[-1]

        mask_xy = mask_g.unsqueeze(-1).expand_as(diff_g)
        rec_face[0] += (diff_g[:, :, :n_face, :] * mask_xy[:, :, :n_face, :]).sum()
        rec_face[1] += mask_xy[:, :, :n_face, :].sum()
        rec_body[0] += (diff_g[:, :, n_face:n_body_end, :] * mask_xy[:, :, n_face:n_body_end, :]).sum()
        rec_body[1] += mask_xy[:, :, n_face:n_body_end, :].sum()
        rec_hand[0] += (diff_g[:, :, n_body_end:, :] * mask_xy[:, :, n_body_end:, :]).sum()
        rec_hand[1] += mask_xy[:, :, n_body_end:, :].sum()

        # Per-keypoint L1: reduce over (batch, time, xy) -> one value per joint.
        wdiff = diff_g * mask_xy
        rec_kp[0] += wdiff.sum(dim=(0, 1, 3))
        rec_kp[1] += mask_xy.sum(dim=(0, 1, 3))
        # Per-timestep L1: reduce over (batch, joints, xy) -> one value per frame.
        per_t = wdiff.sum(dim=(0, 2, 3))
        cnt_t = mask_xy.sum(dim=(0, 2, 3))
        if time_sum is None:
            time_sum = torch.zeros_like(per_t)
            time_cnt = torch.zeros_like(cnt_t)
        elif per_t.shape[0] > time_sum.shape[0]:
            grow_s = torch.zeros_like(per_t)
            grow_c = torch.zeros_like(cnt_t)
            grow_s[: time_sum.shape[0]] = time_sum
            grow_c[: time_cnt.shape[0]] = time_cnt
            time_sum, time_cnt = grow_s, grow_c
        time_sum[: per_t.shape[0]] += per_t
        time_cnt[: cnt_t.shape[0]] += cnt_t

        if out["codes"] is not None:
            if seen_codes is None:
                seen_codes = [
                    torch.zeros(cb_size, dtype=torch.long, device=device)
                    for _ in out["codes"]
                ]
            for cb_idx, codes_i in enumerate(out["codes"]):
                (codes_g,) = gfm(codes_i.contiguous())
                seen_codes[cb_idx].index_fill_(0, codes_g.flatten().unique(), 1)

    def ratio(pair) -> float:
        n, d = pair[0].item(), pair[1].item()
        return n / d if d > 0 else 0.0

    n_b = max(int(n_batches.item()), 1)
    metrics_out = {
        "loss": loss_sum.item() / n_b,
        "rec_loss": rec_sum.item() / n_b,
        "vel_loss": vel_sum.item() / n_b,
        "acc_loss": acc_sum.item() / n_b,
        "conf_loss": conf_sum.item() / n_b,
        "confidence_mae": conf_mae_sum.item() / max(conf_mae_count.item(), 1.0),
        "commitment_loss": commit_sum.item() / n_b,
        "codebook_loss": cb_sum.item() / n_b,
        "rec_face": ratio(rec_face),
        "rec_body": ratio(rec_body),
        "rec_hand": ratio(rec_hand),
        "cb_usage": 0.0,
    }
    if seen_codes is not None:
        per_cb_usage = [(s > 0).sum().item() / cb_size * 100.0 for s in seen_codes]
        metrics_out["cb_usage"] = sum(per_cb_usage) / len(per_cb_usage)
        if len(per_cb_usage) > 1:
            for i, usage in enumerate(per_cb_usage):
                metrics_out[f"cb_usage_{i}"] = usage

    # Array diagnostics (handled specially by run_validation_confidence: logged
    # as per-joint scalars + wandb charts rather than as plain val/ scalars).
    metrics_out["rec_per_kp"] = (
        rec_kp[0] / rec_kp[1].clamp(min=1.0)
    ).tolist()
    if time_sum is not None:
        metrics_out["rec_per_t"] = (
            time_sum / time_cnt.clamp(min=1.0)
        ).tolist()
    return metrics_out


def _joint_group(idx: int) -> str:
    """face (0-4) / body (5-12) / left_hand (13-33) / right_hand (34-54)."""
    if idx < 5:
        return "face"
    if idx < 13:
        return "body"
    if idx < 34:
        return "lhand"
    return "rhand"


def _log_array_diagnostics(rec_per_kp, rec_per_t, global_step: int) -> None:
    """Log per-keypoint and per-timestep reconstruction L1 to wandb.

    Per-keypoint: one scalar per joint (``val/rec_kp/<group>_<idx>``) for easy
    grepping/grouping, plus a bar chart. Per-timestep: a line chart of recon L1
    against the within-clip frame index (reveals frame-0 degeneracy or error
    growth over the clip). Main process only.
    """
    if rec_per_kp is not None:
        wandb.log(
            {f"val/rec_kp/{_joint_group(i)}_{i:02d}": v
             for i, v in enumerate(rec_per_kp)},
            step=global_step,
        )
        try:
            tbl = wandb.Table(
                data=[[i, _joint_group(i), v] for i, v in enumerate(rec_per_kp)],
                columns=["joint", "group", "rec_l1"],
            )
            wandb.log(
                {"val/rec_per_kp_chart": wandb.plot.bar(
                    tbl, "joint", "rec_l1", title="val recon L1 per keypoint")},
                step=global_step,
            )
        except Exception:
            pass
    if rec_per_t is not None:
        # Compact scalar summaries for quick monitoring (first / last / mean).
        wandb.log(
            {
                "val/rec_t_first": rec_per_t[0],
                "val/rec_t_last": rec_per_t[-1],
                "val/rec_t_mean": sum(rec_per_t) / max(len(rec_per_t), 1),
            },
            step=global_step,
        )
        try:
            tbl = wandb.Table(
                data=[[t, v] for t, v in enumerate(rec_per_t)],
                columns=["t", "rec_l1"],
            )
            wandb.log(
                {"val/rec_per_t_chart": wandb.plot.line(
                    tbl, "t", "rec_l1", title="val recon L1 over time (within clip)")},
                step=global_step,
            )
        except Exception:
            pass


def run_validation_confidence(
    accelerator,
    model,
    val_loader,
    cfg: TrainConfig,
    joint_weights: torch.Tensor,
    global_step: int,
    *,
    label: str = "val",
) -> None:
    vm = validate_confidence(
        model,
        val_loader,
        cfg,
        joint_weights,
        accelerator=accelerator,
        silent=not accelerator.is_local_main_process,
    )
    # Array diagnostics are logged as scalars + charts, not as plain val/ values.
    rec_per_kp = vm.pop("rec_per_kp", None)
    rec_per_t = vm.pop("rec_per_t", None)
    if accelerator.is_main_process:
        wandb.log({f"val/{k}": v for k, v in vm.items()}, step=global_step)
        _log_array_diagnostics(rec_per_kp, rec_per_t, global_step)
        accelerator.print(
            f"  -- {label} @ step {global_step} --  "
            f"loss={vm['loss']:.4f}  rec={vm['rec_loss']:.4f}  "
            f"conf={vm['conf_loss']:.4f}  conf_mae={vm['confidence_mae']:.4f}  "
            f"face={vm['rec_face']:.4f}  body={vm['rec_body']:.4f}  "
            f"hand={vm['rec_hand']:.4f}  cb={vm['cb_usage']:.0f}%"
        )
    model.train()


@torch.no_grad()
def _reconstruct_confidence_clip(
    model: torch.nn.Module,
    keypoints: np.ndarray,
    confidence: np.ndarray,
    device,
    cfg: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = keypoints.shape[0]
    downsample = max(1, int(getattr(model, "downsampling_factor", 1)))
    if downsample > 1:
        aligned_frames = (n_frames // downsample) * downsample
        if aligned_frames > 0:
            n_frames = aligned_frames
            keypoints = keypoints[:n_frames]
            confidence = confidence[:n_frames]
    coords = keypoints.reshape(n_frames, NUM_JOINTS, 2)
    features = np.concatenate([coords, confidence[..., None]], axis=-1)
    x = torch.from_numpy(features.reshape(n_frames, CONF_FEATURES)).float()
    out = model(x.unsqueeze(0).to(device))["reconstruction"].squeeze(0).cpu()
    # Keep this as a final guard for unusual convolution length rounding.
    out = out[:n_frames]
    # De-shift for rendering: when trained with causal_loss_shift, recon[t]
    # reconstructs the target at t-shift, so advance recon by `shift` frames to
    # realign it with the ground-truth timeline (repeat the last frame to pad).
    shift = int(getattr(cfg, "causal_loss_shift", 0) or 0)
    if shift > 0 and out.shape[0] > shift:
        out = torch.cat([out[shift:], out[-1:].repeat(shift, 1)], dim=0)
    pred_xy, pred_conf_raw = _split_features(out.unsqueeze(0), NUM_JOINTS)
    pred_conf = _sigmoid_if_needed(pred_conf_raw, cfg).squeeze(0).numpy()
    pred_kp = pred_xy.squeeze(0).numpy().reshape(n_frames, NUM_JOINTS * 2)
    return pred_kp, pred_conf


def render_comparisons_confidence(
    accelerator,
    model,
    test_clips,
    ckpt_dir,
    cfg: TrainConfig,
    global_step: int,
) -> None:
    if not test_clips or ckpt_dir is None:
        return
    unwrapped = accelerator.unwrap_model(model)
    video_dir = Path(ckpt_dir) / "comparisons"
    video_dir.mkdir(parents=True, exist_ok=True)
    fps = cfg.target_fps or cfg.source_fps
    paths = []
    unwrapped.eval()
    for clip in test_clips:
        recon_kp, recon_conf = _reconstruct_confidence_clip(
            unwrapped,
            clip["keypoints"],
            clip["confidence"],
            accelerator.device,
            cfg,
        )
        out_path = video_dir / f"{clip['clip_id']}_comparison.mp4"
        render_comparison_video(
            orig_kp=clip["keypoints"],
            recon_kp=recon_kp,
            global_state=clip["global_state"],
            confidence=clip["confidence"],
            recon_confidence=recon_conf,
            out_path=out_path,
            video_path=clip.get("video_path"),
            fps=fps,
            max_seconds=cfg.test_max_seconds,
        )
        paths.append(out_path)

    if paths:
        wandb.log(
            {"test/comparisons": [wandb.Video(str(p), format="mp4") for p in paths]},
            step=global_step,
        )
    accelerator.print(f"  Rendered {len(paths)} confidence comparison videos -> {video_dir}")


def _write_effective_confidence_yaml(run_dir: str, accelerator) -> None:
    """Keep run config compatible with resume/tokenize scripts."""
    if not accelerator.is_main_process:
        return
    path = Path(run_dir) / "config.yaml"
    if not path.is_file():
        return
    try:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw["input_features"] = CONF_FEATURES
        raw["keypoint_dim"] = COORD_DIMS
        raw["num_keypoints"] = NUM_JOINTS
        with open(path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
    except Exception as exc:
        accelerator.print(f"[warn] Could not update confidence config YAML: {exc}")


def _load_model_init_checkpoint(path: str, model: torch.nn.Module, accelerator) -> None:
    ckpt = load_model_weights(path, model, map_location="cpu")
    accelerator.print(
        f"Initialized model weights from {path} "
        f"(source step {ckpt.get('global_step', 'unknown')})"
    )


def main():
    args = parse_args()
    # Comparison rendering runs on rank 0 only (forward on test clips + ffmpeg +
    # wandb video upload) and can take minutes, during which the other ranks
    # block on the next gradient all-reduce. Raise the collective timeout well
    # above the render duration so the waiting ranks don't abort (the default
    # ~10 min NCCL watchdog otherwise kills the job at a render/save step).
    config_path = resolve_config_path(args)
    cfg, yaml_model_overrides = TrainConfig.from_yaml(config_path)

    accelerator = Accelerator(
        gradient_accumulation_steps=max(1, cfg.gradient_accumulation_steps),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))],
    )

    ctx = bootstrap_run(args, cfg, config_path, yaml_model_overrides, accelerator)
    cfg = ctx.cfg
    _write_effective_confidence_yaml(ctx.run_dir, accelerator)

    model_kwargs = dict(ctx.model_overrides or {})
    if model_kwargs.get("input_features", CONF_FEATURES) != CONF_FEATURES:
        accelerator.print(
            f"Overriding input_features={model_kwargs['input_features']} -> {CONF_FEATURES} "
            "for confidence-aware training."
        )
    model_kwargs["input_features"] = CONF_FEATURES
    model_kwargs["keypoint_dim"] = COORD_DIMS
    model_kwargs["num_keypoints"] = NUM_JOINTS

    model = PoseTokenizerModel(config=PoseTokenizerConfig(**model_kwargs))
    if args.init_from_checkpoint:
        _load_model_init_checkpoint(args.init_from_checkpoint, model, accelerator)
    apply_freeze_config(model, cfg, accelerator)

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay), lr=cfg.lr,
    )

    train_loader, val_loader, test_clips = build_dataloaders(
        cfg,
        downsampling_factor=model.downsampling_factor,
        accelerator=accelerator,
        include_confidence=True,
    )

    joint_weights = build_joint_weights(model.config, cfg.hand_loss_weight)
    if cfg.hand_loss_weight != 1.0:
        accelerator.print(
            f"hand_loss_weight={cfg.hand_loss_weight} -> joint weights "
            f"body={joint_weights[0].item():.3f}  hand={joint_weights[-1].item():.3f}"
        )
    accelerator.print(
        f"Confidence-aware input: {NUM_JOINTS} joints x (y, x, conf) = {CONF_FEATURES} features; "
        f"confidence_loss_type={cfg.confidence_loss_type}, "
        f"confidence_loss_weight={cfg.confidence_loss_weight}"
    )
    if cfg.pose_corruption_prob > 0.0:
        accelerator.print(
            "Pose input corruption: "
            f"prob={cfg.pose_corruption_prob}, modes={cfg.pose_corruption_modes or ['temporal_joint_conf_drop']}, "
            f"min_conf={cfg.pose_corruption_min_conf}, "
            f"span=[{cfg.pose_corruption_span_min}, {cfg.pose_corruption_span_max}], "
            f"conf=[{cfg.pose_corruption_conf_min}, {cfg.pose_corruption_conf_max}], "
            f"xy_noise_std={cfg.pose_corruption_xy_noise_std}, "
            f"xy_outlier_prob={cfg.pose_corruption_xy_outlier_prob}, "
            f"xy_outlier_std={cfg.pose_corruption_xy_outlier_std}"
        )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,
    )
    joint_weights = joint_weights.to(accelerator.device)
    accelerator.print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    global_step, start_epoch, wandb_run_id = 0, 0, ctx.run_id
    if args.checkpoint:
        unwrapped = accelerator.unwrap_model(model)
        global_step, start_epoch, wandb_run_id = load_checkpoint(
            args.checkpoint, unwrapped, optimizer, accelerator.device,
        )
        accelerator.print(f"Resumed from {args.checkpoint}  (step {global_step}, epoch {start_epoch})")

    scheduler = build_scheduler(optimizer, cfg, start_step=global_step)
    if scheduler is not None:
        accelerator.print(
            f"LR schedule: {cfg.lr_schedule}  "
            f"(lr={cfg.lr:.2e} -> lr_min={cfg.lr_min:.2e} over {cfg.max_steps} steps; "
            f"starting at step {global_step}, current lr={optimizer.param_groups[0]['lr']:.2e})"
        )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        wandb.config.update(build_wandb_config(cfg, unwrapped), allow_val_change=True)
        wandb.summary["train_samples"] = len(train_loader.dataset)
        wandb.summary["val_samples"] = len(val_loader.dataset)

    max_steps = cfg.max_steps if cfg.max_steps > 0 else float("inf")
    reached_max_steps = False

    run_validation_confidence(
        accelerator,
        model,
        val_loader,
        cfg,
        joint_weights,
        global_step,
        label="initial val" if global_step == 0 else "resume val",
    )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        accelerator.print(f"\n-- Epoch {epoch + 1}/{cfg.epochs} ({len(train_loader)} batches) --")

        if global_step == 0:
            accelerator.print(
                "Loading first batch (NFS + collate pad to max clip length in batch; "
                "large T can take minutes before the first step log)..."
            )
            first_batch_t0 = time.perf_counter()

        for batch in train_loader:
            if global_step == 0 and first_batch_t0 is not None:
                dt = time.perf_counter() - first_batch_t0
                t_steps = batch["keypoints"].shape[1]
                accelerator.print(
                    f"First batch ready in {dt:.1f}s -- padded T={t_steps} "
                    f"(batch_size={batch['keypoints'].shape[0]})"
                )
                first_batch_t0 = None

            with accelerator.accumulate(model):
                loss, metrics, _ = compute_losses(model, batch, cfg, joint_weights)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # accelerator.accumulate only steps the optimizer on the accumulation
            # boundary; sync_gradients marks it, so step-counted bookkeeping
            # (scheduler/logging/val/save) hangs off it.
            if not accelerator.sync_gradients:
                continue

            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if accelerator.is_main_process and global_step % cfg.log_interval == 0:
                log_train_step(
                    accelerator,
                    global_step,
                    epoch,
                    metrics={k: v.item() for k, v in metrics.items()},
                    lr=optimizer.param_groups[0]["lr"],
                    max_steps=cfg.max_steps,
                )

            if global_step % cfg.validation_interval == 0:
                run_validation_confidence(
                    accelerator, model, val_loader, cfg, joint_weights, global_step,
                )

            if global_step % cfg.save_interval == 0:
                ckpt_dir = save_checkpoint(
                    accelerator, model, optimizer, global_step, epoch, wandb_run_id, ctx.run_dir,
                    save_optimizer_state=cfg.save_optimizer_state,
                )
                render_every = cfg.render_interval if cfg.render_interval > 0 else cfg.save_interval
                if accelerator.is_main_process and global_step % render_every == 0:
                    try:
                        render_comparisons_confidence(
                            accelerator, model, test_clips, ckpt_dir, cfg, global_step,
                        )
                    except Exception as exc:  # never let rendering kill training
                        accelerator.print(f"[warn] comparison render failed: {exc}")
                    finally:
                        # render_comparisons_confidence leaves the model in eval();
                        # restore train mode so quantizer dropout / dead-code
                        # revival stay active for subsequent steps.
                        model.train()

            if global_step >= max_steps:
                accelerator.print(f"\nReached max_steps={cfg.max_steps}, stopping.")
                reached_max_steps = True
                break

        if reached_max_steps:
            break

    ckpt_dir = save_checkpoint(
        accelerator, model, optimizer, global_step, epoch, wandb_run_id, ctx.run_dir,
        is_final=True, save_optimizer_state=cfg.save_optimizer_state,
    )
    if accelerator.is_main_process:
        render_comparisons_confidence(accelerator, model, test_clips, ckpt_dir, cfg, global_step)
        wandb.finish()


if __name__ == "__main__":
    main()
