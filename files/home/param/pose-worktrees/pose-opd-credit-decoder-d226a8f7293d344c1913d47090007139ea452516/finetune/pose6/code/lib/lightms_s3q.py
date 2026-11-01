"""SAM's POSE-LIGHT MICRO-STEP scheme (10-10, user order "copy it exactly"; opt-in POSE_LIGHT_MS_K > 0, default 0 = off), from
/data/home/amartyavarma/qwen-1ch/infer/sam_pose/code/train/train_pose.py (pose_light_pass, sched, the step loop):
  per trunk step: the pose-light frames (all of them: POSE_RATIO 1.0) are shuffled (randperm) and split into K strided chunks perm[j::K]; for each
  chunk j: forward + backward of the pose light tower on the DETACHED tower input (SAM: the trunk hidden; ours: the pose 8B's output phv),
  chunk loss = w x mean CE over (chunk frames x 31 codebooks); the gradient w.r.t. the detached input is accumulated weighted by chunk size / N
  (= one full-batch mean); then clip_grad_norm(pose-light params, 0.5) and an OPTIMIZER STEP of the pose light (own AdamW: betas (0.9, 0.95),
  eps 1e-8, wd 0) at LR sched(step x K + j, warmup x K, horizon x K) = linear warmup, cosine from the peak to the floor (5e-5). The
  accumulated input gradient then goes through the rest of the graph ONCE, with the other losses, in the main backward.
Pure helpers here (unit-tested in tests/test_lightms.py against a verbatim transcription of SAM's loop); train_pose_s3q.py wires them in."""
import math

import torch


def sam_lr(s, warmup, horizon, base, floor):
    """SAM's sched() for one param group at (micro-)step s: linear warmup (s + 1) / warmup, then cosine base -> floor over [warmup, horizon]."""
    if s < warmup:
        return base * (s + 1) / warmup
    prog = min(1.0, (s - warmup) / max(1, horizon - warmup))
    return floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * prog))


def chunks(n, k, gen):
    """SAM: perm = randperm(n); chunk j = perm[j::k] (strided over the shuffled frames)."""
    perm = torch.randperm(n, generator=gen)
    return [perm[j::k] for j in range(k)]


def micro_steps(light_loss, hd, n_frames, k, gen, opt, params, clip, lr_of, step):
    """SAM's pose_light_pass. light_loss(idx) -> (scalar chunk loss = w x mean CE over the chunk, stats) on the LEAF hd (detached input);
    -> (accumulated d loss / d hd (fp32, size-weighted = the full-batch mean), list of chunk stats, list of LRs used, number of optimizer steps).
    Every chunk: fwd + bwd, grad accumulation, clip, opt.step(), zero_grad. Chunks that are empty (n < k) still step (SAM's loop runs K times)."""
    assert hd.is_leaf and hd.requires_grad, "the light micro-steps run on a DETACHED leaf input"
    acc = torch.zeros(hd.shape, dtype=torch.float32, device=hd.device)
    stats, lrs, n_steps = [], [], 0
    for j, c in enumerate(chunks(n_frames, k, gen)):
        hd.grad = None
        loss, st = light_loss(c)
        loss.backward()
        if hd.grad is not None:
            acc += hd.grad.float() * (c.numel() / max(n_frames, 1))
        stats.append(st)
        lr = lr_of(step * k + j)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step(); n_steps += 1
        opt.zero_grad(set_to_none=True)
        lrs.append(lr)
    hd.grad = None
    return acc, stats, lrs, n_steps
