"""Stage-3 LR schedule (multiplier of each group's peak LR), pure Python (no torch) so it is unit-testable anywhere.

  cosine (S3_SCHED=cosine, default; bit-identical to the pre-10-06 make_opt lambda):
      t < w: t / w;   else: mr + (1 - mr) * 0.5 * (1 + cos(pi * min(1, (t - w) / (N - w))))     (N = horizon, mr = min_lr_rate)
  wsd (S3_SCHED=wsd, 10-06 user decision):
      t < w: t / w  (warmup unchanged);   w <= t < S: 1.0 (stable = exactly peak);
      S <= t < E: F + (1 - F) * shape(p), p = (t - S) / (E - S), shape linear 1 - p | sqrt 1 - sqrt(p) | cos 0.5 (1 + cos(pi p));   t >= E: F.
  w = warmup if the group is in warmup_groups else 0. LambdaLR convention: lam(t) is the LR of the optimizer update that produces step
  t + 1, so with E = the leg's last step the final update (-> step E) runs at lam(E - 1) = F + (1 - F) / (E - S) (linear) and the step-E
  checkpoint's scheduler sits at lam(E) = F (F = 0 is safe: AdamW with lr 0 is a no-op; nothing divides by the LR).
  The WSD knobs (s3_sched, wsd_*) are NOT resume keys: a run can switch cosine -> wsd (or move the decay) at any resume.
"""
import math

SHAPES = ("linear", "sqrt", "cos")


def wsd_resolve(sched, decay_start, decay_end, decay_frac, floor, shape, max_steps_cap, horizon, warmup):
    """-> (decay_start, decay_end) resolved + validated. decay_end <= 0 -> horizon (the whole run);
    decay_frac > 0 -> decay_start = floor(end - frac * end + 0.5) (if decay_start is also set they must agree)."""
    if sched == "cosine":
        return int(decay_start), int(decay_end)
    assert sched == "wsd", ("S3_SCHED must be cosine | wsd", sched)
    assert shape in SHAPES, ("WSD_SHAPE", shape, SHAPES)
    assert 0.0 <= floor <= 1.0, ("WSD_FLOOR", floor)
    end = int(decay_end) if int(decay_end) > 0 else int(horizon)       # 10-06 PM: default = the whole run (HORIZON_S3), not the leg
    start = int(decay_start)
    if decay_frac > 0:
        assert 0.0 < decay_frac < 1.0, ("WSD_DECAY_FRAC must be in (0, 1)", decay_frac)
        fs = int(math.floor(end - decay_frac * end + 0.5))          # half-up (17050, 0.15 -> 14492.5 -> 14493; Python round() would give 14492)
        assert start <= 0 or start == fs, ("WSD_DECAY_START disagrees with WSD_DECAY_FRAC", start, fs, end, decay_frac)
        start = fs
    assert 0 < start < end, ("WSD needs 0 < WSD_DECAY_START < WSD_DECAY_END (or WSD_DECAY_FRAC)", start, end)
    assert start >= int(warmup), ("WSD decay starts inside the warmup", start, warmup)
    return start, end


def make_lambda(warm, cfg):
    """cfg keys: warmup, horizon, min_lr_rate, s3_sched (default cosine), wsd_decay_start, wsd_decay_end, wsd_floor, wsd_shape."""
    Wu, N, mr = int(cfg["warmup"]), int(cfg["horizon"]), float(cfg["min_lr_rate"])
    mode = cfg.get("s3_sched", "cosine")
    w = Wu if warm else 0
    if mode == "cosine":
        def lam(t):
            if t < w:
                return t / max(1, w)
            return mr + (1.0 - mr) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, (t - w) / max(1, N - w))))
        return lam
    assert mode == "wsd", mode
    S, E, F, shape = int(cfg["wsd_decay_start"]), int(cfg["wsd_decay_end"]), float(cfg["wsd_floor"]), cfg["wsd_shape"]
    assert 0 < S < E and shape in SHAPES and 0.0 <= F <= 1.0, (S, E, F, shape)

    def lam(t):
        if t < w:
            return t / max(1, w)
        if t < S:
            return 1.0
        if t >= E:
            return F
        p = (t - S) / (E - S)
        s = 1.0 - p if shape == "linear" else (1.0 - math.sqrt(p) if shape == "sqrt" else 0.5 * (1.0 + math.cos(math.pi * p)))
        return F + (1.0 - F) * s
    return lam


def describe(cfg):
    if cfg.get("s3_sched", "cosine") == "cosine":
        return (f"cosine: warmup {cfg['warmup']} on {cfg['warmup_groups']}, then cosine 1 -> {cfg['min_lr_rate']} x peak over horizon {cfg['horizon']}")
    return (f"WSD: warmup {cfg['warmup']} on {cfg['warmup_groups']}, stable 1.0 x peak until step {cfg['wsd_decay_start']}, {cfg['wsd_shape']} decay -> "
            f"{cfg['wsd_floor']} x peak over steps {cfg['wsd_decay_start']}..{cfg['wsd_decay_end']} ({cfg['wsd_decay_end'] - cfg['wsd_decay_start']} steps), "
            f"{cfg['wsd_floor']} x peak after (horizon {cfg['horizon']} / min_lr_rate {cfg['min_lr_rate']} unused in wsd mode, kept as resume keys)")


def ramp_wrap(lam, frm, peak, start, steps):
    """10-06 branch LR ramp (LR_RAMP_*; NOT resume keys): this group's LR goes linearly frm -> peak x lam over updates start .. start + steps
    (factor a + (1 - a) min(1, (t - start) / steps), a = frm / peak), i.e. the first update after the resume runs at frm, the update at
    t = start + steps at the full schedule; t < start (pre-branch steps) is unchanged."""
    a, s0, n = float(frm) / float(peak), int(start), int(steps)
    assert n > 0 and float(peak) > 0 and 0.0 <= a, (frm, peak, steps)

    def f(t):
        if t < s0:
            return lam(t)
        return lam(t) * (a + (1.0 - a) * min(1.0, (t - s0) / n))
    return f


def adapter_new_wrap(lam, start, warm, ramp):
    """10-06 parallel speech adapter 'adapter_new' (model_s3q doc): LR factor 0 EXACTLY for the `warm` updates at t = start .. start + warm - 1
    (the MOMENT WARMUP: Adam's exp_avg / exp_avg_sq accumulate real gradients, its own step count runs, the weights do not move), then a
    linear ramp: update t = start + warm + i (i = 0 .. ramp - 1) gets (i + 1) / ramp, full from start + warm + ramp - 1 on; x the base
    schedule lam(t). t = LambdaLR's last_epoch at the update (the update logged as step t + 1)."""
    s0, w, r = int(start), int(warm), max(1, int(ramp))

    def f(t):
        if t < s0 + w:
            return 0.0
        return lam(t) * min(1.0, (t - s0 - w + 1) / r)
    return f
