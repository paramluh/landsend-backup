"""Train the pose tokenizer.

    accelerate launch train.py --config configs/wide_l2norm.yaml
    accelerate launch train.py --config configs/wide_l2norm.yaml --run-name my_experiment
    accelerate launch train.py --checkpoint checkpoints/my-run/step_25000/checkpoint.pt
"""

from __future__ import annotations

import math
import time

import torch
import wandb
from accelerate import Accelerator

from pose_tokenizer import PoseTokenizerModel
from pose_tokenizer.config import PoseTokenizerConfig, TrainConfig
from pose_tokenizer.training import (
    bootstrap_run,
    build_dataloaders,
    build_wandb_config,
    load_checkpoint,
    log_train_step,
    parse_args,
    render_comparisons,
    resolve_config_path,
    run_validation,
    save_checkpoint,
)

torch.manual_seed(42)


def build_feature_weights(model_cfg: PoseTokenizerConfig, hand_loss_weight: float) -> torch.Tensor:
    """Per-feature weight tensor for the L1 reconstruction loss.

    Layout (kinematic.py): 8 upper-body kps + 21 left-hand + 21 right-hand,
    each `keypoint_dim`-D and flattened to `input_features`. Hand keypoints
    occupy flat indices [16, input_features). Returned tensor is normalised
    to mean=1.0 so other loss weights stay calibrated.
    """
    n_body_kp = model_cfg.num_keypoints - 42  # 50 - 21 - 21
    body_dims = n_body_kp * model_cfg.keypoint_dim
    w = torch.ones(model_cfg.input_features)
    w[body_dims:] = hand_loss_weight
    return w * (w.numel() / w.sum())


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into AdamW decay / no-decay groups.

    No-decay group contains:
    - Any 1D parameter (biases, LayerNorm γβ, weight-norm gain).
    - Codebook embeddings — discrete-ish lookup vectors that need to span the
      encoder's output distribution; shrinking them toward zero collapses the
      latent space and tanks codebook utilisation.
    Everything else (Conv1d weights) goes into the decay group.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "codebook" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, cfg: TrainConfig, *, start_step: int):
    """Create LR scheduler. start_step lets resume continue from the right point.
    Returns None for constant LR."""
    if cfg.lr_schedule == "constant":
        return None
    if cfg.max_steps <= 0:
        raise ValueError(
            f"lr_schedule={cfg.lr_schedule!r} requires max_steps > 0 to define the horizon."
        )
    # last_epoch is 0-indexed: schedule jumps to the LR for step `last_epoch + 1`
    # on the next .step() call. Passing -1 gives a fresh scheduler at step 0.
    # When resuming (last_epoch >= 0), PyTorch's scheduler needs `initial_lr` on
    # each param group — normally set automatically only when last_epoch == -1.
    last_epoch = start_step - 1
    if last_epoch >= 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    if cfg.lr_schedule == "cosine":
        warmup = max(0, cfg.warmup_steps)
        if warmup >= cfg.max_steps:
            raise ValueError(
                f"warmup_steps={warmup} must be < max_steps={cfg.max_steps}"
            )
        ratio_min = cfg.lr_min / cfg.lr if cfg.lr > 0 else 0.0
        cosine_steps = cfg.max_steps - warmup

        def lr_lambda(step: int) -> float:
            if step < warmup:
                # Linear ramp from base_lr/warmup to base_lr over `warmup` steps;
                # offset by +1 to avoid a wasted LR=0 step at iteration 0.
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, cosine_steps)
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return ratio_min + (1.0 - ratio_min) * cos

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=last_epoch,
        )
    raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")


def main():
    args = parse_args()

    config_path = resolve_config_path(args)
    cfg, yaml_model_overrides = TrainConfig.from_yaml(config_path)

    accelerator = Accelerator(
        gradient_accumulation_steps=max(1, cfg.gradient_accumulation_steps),
    )

    ctx = bootstrap_run(args, cfg, config_path, yaml_model_overrides, accelerator)
    cfg = ctx.cfg

    # Model
    model_kwargs = ctx.model_overrides or {}
    model = (
        PoseTokenizerModel(config=PoseTokenizerConfig(**model_kwargs))
        if model_kwargs else PoseTokenizerModel()
    )
    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay), lr=cfg.lr,
    )

    # Data
    train_loader, val_loader, test_clips = build_dataloaders(
        cfg, downsampling_factor=model.downsampling_factor, accelerator=accelerator,
    )

    # Per-feature loss weights
    feat_weights = build_feature_weights(model.config, cfg.hand_loss_weight)
    if cfg.hand_loss_weight != 1.0:
        accelerator.print(
            f"hand_loss_weight={cfg.hand_loss_weight} → per-feature weights "
            f"body={feat_weights[0].item():.3f}  hand={feat_weights[-1].item():.3f}"
        )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,
    )
    feat_weights = feat_weights.to(accelerator.device)
    accelerator.print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume
    global_step, start_epoch, wandb_run_id = 0, 0, ctx.run_id
    if args.checkpoint:
        unwrapped = accelerator.unwrap_model(model)
        global_step, start_epoch, wandb_run_id = load_checkpoint(
            args.checkpoint, unwrapped, optimizer, accelerator.device,
        )
        accelerator.print(f"Resumed from {args.checkpoint}  (step {global_step}, epoch {start_epoch})")

    # LR scheduler — built after resume so it knows the correct start step.
    scheduler = build_scheduler(optimizer, cfg, start_step=global_step)
    if scheduler is not None:
        accelerator.print(
            f"LR schedule: {cfg.lr_schedule}  "
            f"(lr={cfg.lr:.2e} → lr_min={cfg.lr_min:.2e} over {cfg.max_steps} steps; "
            f"starting at step {global_step}, current lr={optimizer.param_groups[0]['lr']:.2e})"
        )

    # Update wandb config with the full merged config (after sweep overrides applied)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        wandb.config.update(build_wandb_config(cfg, unwrapped), allow_val_change=True)
        wandb.summary["train_samples"] = len(train_loader.dataset)
        wandb.summary["val_samples"] = len(val_loader.dataset)

    max_steps = cfg.max_steps if cfg.max_steps > 0 else float("inf")
    reached_max_steps = False

    # Initial validation: fresh runs get a step-0 reading (untrained model
    # baseline → wandb chart starts from the very beginning instead of
    # validation_interval); resumes get a sanity check that the checkpoint
    # loaded correctly before any new gradient steps land.
    run_validation(
        accelerator, model, val_loader, cfg, feat_weights, global_step,
        label="initial val" if global_step == 0 else "resume val",
    )

    kp_dim = (model.module.config.keypoint_dim
              if hasattr(model, "module") else model.config.keypoint_dim)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        accelerator.print(f"\n── Epoch {epoch + 1}/{cfg.epochs} ({len(train_loader)} batches) ──")

        if global_step == 0:
            accelerator.print(
                "Loading first batch (NFS + collate pad to max clip length in batch; "
                "large T can take minutes before the first step log)…"
            )
            _first_batch_t0 = time.perf_counter()

        for batch in train_loader:
            if global_step == 0 and _first_batch_t0 is not None:
                dt = time.perf_counter() - _first_batch_t0
                t_steps = batch["keypoints"].shape[1]
                accelerator.print(
                    f"First batch ready in {dt:.1f}s — padded T={t_steps} "
                    f"(batch_size={batch['keypoints'].shape[0]})"
                )
                _first_batch_t0 = None

            with accelerator.accumulate(model):
                feats = batch["keypoints"]
                # Clamp obvious detection-failure outliers (kinematic offsets divided by
                # tiny shoulder widths can hit O(100); typical magnitude is < 5).
                feats = feats.clamp(-10.0, 10.0)
                out = model(feats)
                # part_rvq may supervise the whole-body loss on a subset of
                # residual layers (reconstruction_for_loss); falls back to the
                # full reconstruction for all other model types.
                recon = out.get("reconstruction_for_loss", out["reconstruction"])

                # Confidence weighting on the reconstruction (and velocity) loss.
                # `confidence` is (B, T, NUM_JOINTS); we broadcast each joint's
                # weight across its keypoint_dim contiguous flat features.
                #   "hard": binary mask (conf >= threshold), drop from num + denom
                #   "soft": SMPLify-X-style, weight = conf in [0, 1]
                #   "none": uniform L1 (legacy)
                mode = cfg.confidence_weight_mode.lower()
                w_joint = None  # (B, T, NUM_JOINTS) per-joint weight, if any
                if mode in ("hard", "soft"):
                    conf = batch["confidence"]
                    if mode == "hard":
                        w_joint = (conf >= cfg.confidence_mask_threshold).to(feats.dtype)
                    else:
                        w_joint = conf.to(feats.dtype)
                    conf_weights = w_joint.repeat_interleave(kp_dim, dim=-1)
                elif mode == "none":
                    conf_weights = None
                else:
                    raise ValueError(
                        f"confidence_weight_mode must be one of "
                        f"{{'none','soft','hard'}}, got {cfg.confidence_weight_mode!r}"
                    )

                # Per-feature L1 (mean=1 weights, so equivalent to F.l1_loss when uniform).
                rec_per = feat_weights * (recon - feats).abs()
                if conf_weights is not None:
                    rec_loss = (rec_per * conf_weights).sum() / conf_weights.sum().clamp(min=1.0)
                else:
                    rec_loss = rec_per.mean()

                # Velocity L1: penalises static-prediction strategies that look good
                # under per-frame L1 but freeze low-motion regions (e.g. hands).
                if cfg.velocity_weight > 0.0:
                    vel_recon = recon[:, 1:, :] - recon[:, :-1, :]
                    vel_true = feats[:, 1:, :] - feats[:, :-1, :]
                    vel_per = feat_weights * (vel_recon - vel_true).abs()
                    if conf_weights is not None:
                        # A velocity sample's trust is the AND of both endpoint
                        # confidences. `min` is the canonical fuzzy-AND: for binary
                        # masks (hard mode) it's identical to multiplication, but for
                        # continuous weights (soft mode) it avoids squaring conf —
                        # which previously turned a fingertip pair at conf≈0.4 into
                        # vel_weight 0.16 vs body's 0.81, killing gradient signal on
                        # exactly the fast-motion frames the vel loss exists to fix.
                        vel_w = torch.minimum(conf_weights[:, 1:, :], conf_weights[:, :-1, :])
                        vel_loss = (vel_per * vel_w).sum() / vel_w.sum().clamp(min=1.0)
                    else:
                        vel_loss = vel_per.mean()
                else:
                    vel_loss = torch.zeros((), device=feats.device)

                # Acceleration L1: penalises over-smoothing of sharp motion (the
                # lossy downsampling codec's dominant failure mode; error tracks
                # acceleration far more than velocity). Second difference over
                # time; a sample's trust is the fuzzy-AND (min) of its three
                # endpoint confidences so it does not learn to fit tracker jitter.
                if cfg.acceleration_weight > 0.0:
                    acc_recon = recon[:, 2:, :] - 2.0 * recon[:, 1:-1, :] + recon[:, :-2, :]
                    acc_true = feats[:, 2:, :] - 2.0 * feats[:, 1:-1, :] + feats[:, :-2, :]
                    acc_per = feat_weights * (acc_recon - acc_true).abs()
                    if conf_weights is not None:
                        acc_w = torch.minimum(
                            torch.minimum(conf_weights[:, 2:, :], conf_weights[:, 1:-1, :]),
                            conf_weights[:, :-2, :],
                        )
                        acc_loss = (acc_per * acc_w).sum() / acc_w.sum().clamp(min=1.0)
                    else:
                        acc_loss = acc_per.mean()
                else:
                    acc_loss = torch.zeros((), device=feats.device)

                # PRQ term (B): per-part reconstruction L1. Each part must be
                # independently decodable, so we supervise every part's own
                # reconstruction (before aggregation) against its joints' targets.
                part_loss = torch.zeros((), device=feats.device)
                per_part_loss = None   # (N,) per-part L1, for logging
                part_names = None
                if "part_recon" in out and cfg.part_loss_weight != 0.0:
                    um = model.module if hasattr(model, "module") else model
                    part_names = um.part_names
                    pj = um.part_joint_idx                       # (N, M) long
                    num_kp = um.config.num_keypoints
                    Bf, Tf = feats.shape[0], feats.shape[1]
                    feats_grid = feats.reshape(Bf, Tf, num_kp, kp_dim)
                    tgt = feats_grid[:, :, pj, :]                # (B, T, N, M, kp)
                    pred = torch.stack(out["part_recon"], dim=2) # (B, T, N, M, kp)
                    fw = feat_weights.reshape(num_kp, kp_dim)[pj]  # (N, M, kp)
                    part_per = fw * (pred - tgt).abs()
                    # Per-part L1: reduce over batch/time/joint/coord, keep parts.
                    red_dims = (0, 1, 3, 4)
                    if w_joint is not None:
                        cw = w_joint[:, :, pj].unsqueeze(-1)     # (B, T, N, M, 1)
                        cw_b = cw.expand_as(part_per)
                        per_part_num = (part_per * cw_b).sum(dim=red_dims)   # (N,)
                        per_part_den = cw_b.sum(dim=red_dims).clamp(min=1.0)  # (N,)
                        per_part_loss = per_part_num / per_part_den
                        part_loss = per_part_num.sum() / per_part_den.sum().clamp(min=1.0)
                    else:
                        per_part_loss = part_per.mean(dim=red_dims)          # (N,)
                        part_loss = per_part_loss.mean()

                loss = (cfg.reconstruction_weight * rec_loss
                        + cfg.velocity_weight * vel_loss
                        + cfg.acceleration_weight * acc_loss
                        + cfg.part_loss_weight * part_loss
                        + cfg.commitment_weight * out["commitment_loss"]
                        + cfg.codebook_weight * out["codebook_loss"])

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # accelerator.accumulate only performs the optimizer step on the
            # accumulation boundary; sync_gradients marks that boundary, so all
            # step-counted bookkeeping (scheduler/logging/val/save) hangs off it.
            if not accelerator.sync_gradients:
                continue

            if scheduler is not None:
                scheduler.step()
            global_step += 1

            if accelerator.is_main_process and global_step % cfg.log_interval == 0:
                step_metrics = {
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "vel_loss": vel_loss.item(),
                    "acc_loss": acc_loss.item(),
                    "part_loss": part_loss.item(),
                    "commitment_loss": out["commitment_loss"].item(),
                    "codebook_loss": out["codebook_loss"].item(),
                }
                # Per-part reconstruction L1 (part_rvq) so we can see which body
                # parts learn fastest/slowest.
                if per_part_loss is not None and part_names is not None:
                    for name, v in zip(part_names, per_part_loss.tolist()):
                        step_metrics[f"part_loss/{name}"] = v
                log_train_step(
                    accelerator, global_step, epoch,
                    metrics=step_metrics,
                    lr=optimizer.param_groups[0]["lr"],
                    max_steps=cfg.max_steps,
                )

            if global_step % cfg.validation_interval == 0:
                run_validation(accelerator, model, val_loader, cfg,
                               feat_weights, global_step)

            if global_step % cfg.save_interval == 0:
                ckpt_dir = save_checkpoint(
                    accelerator, model, optimizer, global_step, epoch, wandb_run_id, ctx.run_dir,
                    save_optimizer_state=cfg.save_optimizer_state,
                )
                render_every = cfg.render_interval if cfg.render_interval > 0 else cfg.save_interval
                if accelerator.is_main_process and global_step % render_every == 0:
                    render_comparisons(accelerator, model, test_clips, ckpt_dir, cfg, global_step)

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
        render_comparisons(accelerator, model, test_clips, ckpt_dir, cfg, global_step)
        wandb.finish()


if __name__ == "__main__":
    main()
