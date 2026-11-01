# No-shift tokenizer: alignment, cold-start, and the padding-code correction

> **CORRECTION (supersedes the earlier version of this doc).** The earlier
> writeup claimed the no-shift model (1) trails GT by ~1 frame and (2) has a
> severe start-of-clip instability fixed by replicate padding. **Both were
> artifacts of running inference with the wrong causal-padding code**, not
> properties of the model. This document records the corrected findings. The
> `analysis/` scripts were written against the buggy code and their conclusions
> should be read through this correction.

Models compared (both 32cb):
- no-shift: `base-causal-pretrain-32cb-pose20k-subset1.2tb` (`causal_loss_shift=0`)
- loss-shift: `...-2x-wide-32cb-...-lossshift1-...-cb_size_2048` (`causal_loss_shift=1`, 2x-wide, cb_size 2048)

---

## 0. Root cause: a padding-code mismatch (`pad_left=2` vs `pad_left=1`)

`CausalConv1d.pad_left` differs between branches:

| branch | formula | downsample (k=3, stride=2) |
|---|---|---|
| `main` | `(kernel-1)*dilation` | `pad_left = 2` |
| `james-4cb-frame0-root` | `max(0, (kernel-1)*dilation - (stride-1))` | `pad_left = 1` (minimal-latency) |

The checkpoints were **trained on the frame0-root code (`pad_left=1`)**. All of
the earlier analysis accidentally ran on `main` (`pad_left=2`), i.e. with one
extra leading pad position at the stride-2 `downsample_conv` that the model
never saw in training. That single mismatch produced **both** headline
"findings":

- **The lag.** Optimal frame-shift is exactly `d* = pad_left - 1`:

  | downsample `pad_left` | 0 | **1 (trained)** | 2 (`main`) | 3 |
  |---|---|---|---|---|
  | median optimal shift d* | -1 | **0** | +1 | +2 |

  So `main`'s `pad_left=2` manufactured the +1-frame "lag". On the correct code
  the no-shift model aligns at **d* = 0** (sub-frame estimate +0.010 ± 0.010
  frames), with a symmetric error curve (a ±1-frame shift roughly doubles L1).

- **The cold-start blowup.** With `pad_left=2`, zero-padding feeds an all-zeros
  "fake frame" into the downsample where the model expects real data, blowing up
  the first frames. Replicate padding filled that slot with real content and
  masked it. On the correct `pad_left=1` code there is no spurious slot, so there
  is no blowup and nothing for replicate to fix (see §2).

**Net: "no-shift lags +1 frame" and "replicate fixes cold-start" were two
symptoms of the same bug.** On the trained code, the no-shift model has no lag
and no meaningful cold-start, and plain zero-padding is optimal.

Correcting the branch (same weights, correct inference code) roughly **halves**
the measured full-clip L1 on the 10 valds clips: **mean 0.0185 -> 0.0108
(-42%)**. That is not the model improving; the old numbers were inflated because
a 1-frame-lagged recon was scored against GT, so inter-frame motion counted as
error.

---

## 1. How to run inference (`scripts/infer_reconstruct_video.py`)

Extract RTMW-X poses from an mp4 (or load a `.npz`), tokenize -> decode with a
pose tokenizer, and render comparison + overlay videos.

```bash
cd pose-tokenizer
uv run python scripts/infer_reconstruct_video.py <checkpoint> <clip.mp4> \
    --out         out_comparison.mp4 \
    --overlay-out out_overlay.mp4 \
    --pose-out    out_rtmwx.npz \
    --conf-lo 0 --conf-hi 1 \
    --deshift -1 \
    --pad-mode constant
```

> **Run against `james-4cb-frame0-root` (or any branch with the
> `pad_left = max(0, (kernel-1)*dilation - (stride-1))` fix).** Running the same
> script from a `pad_left=2` branch silently reintroduces the +1-frame lag and
> the zero-pad cold-start blowup.

Key options:
- `--out` : side-by-side `Original | Reconstructed` MP4 (skeletons coloured by a
  green->yellow->red confidence gradient).
- `--overlay-out` : both skeletons on one frame (cyan=original, magenta=recon).
- `--conf-lo/--conf-hi` : RTMW-score window mapped onto the gradient. Low-res
  model-resolution clips: `0 1`; real high-res footage: `1 6`.
- `--deshift N` : advance recon by N frames. `-1` = auto (reads
  `cfg.causal_loss_shift`), `0` = off. **For the no-shift model, auto resolves to
  0 and that is correct** (d* = 0 on the fixed code) - no manual shift needed.
- `--pad-mode {constant,replicate,reflect}` : left-padding mode of the causal
  convs. **Use `constant` (zero-pad, as trained) - it is optimal.** `replicate`
  is retained only as a diagnostic; on the correct code it *hurts* the opening
  frames (see §2).

---

## 2. Cold-start and padding on the correct (`pad_left=1`) code

Per-segment L1 for the no-shift model, same 10 valds clips, `--deshift 0`,
constant vs replicate padding, at both pad conventions:

| pad_left / mode | cold (frames 0-3) | warmup (0-7) | steady (8+) |
|---|---:|---:|---:|
| **1 (trained)** constant | 0.01186 | 0.01163 | 0.01110 |
| **1 (trained)** replicate | 0.01668 | 0.01427 | 0.01110 |
| 2 (`main`, buggy) constant | **0.09703** | 0.06382 | 0.01648 |
| 2 (`main`, buggy) replicate | 0.01337 | 0.01534 | 0.01644 |

- On the trained code, zero-pad cold-start is **0.0119 vs 0.0111 steady (~+7%)** -
  no meaningful instability.
- **Replicate makes the opening frames worse** (cold-start +40.6%, warmup +22.8%)
  and leaves steady-state identical - it is an inference-time OOD input the model
  was not trained on.
- The dramatic `pad_left=2 constant` cold-start (0.097, ~6x its steady state) is
  the artifact the earlier doc described as the no-shift "instability"; replicate
  only ever helped in that broken regime.

**Recommendation: keep `constant` padding.** If a more robust cold-start were
ever wanted it would have to be *trained* in (train with replicate/edge padding),
not swapped in at inference.

---

## 3. No-shift vs loss-shift (correct code, `--deshift 0`)

Same 10 clips, coordinate-space L1:

| metric | no-shift | loss-shift | winner |
|---|---:|---:|---|
| full-clip mean L1 | 0.01112 | 0.01172 | **no-shift** (~5%) |
| steady-state (skip first 8 frames) | 0.01110 | 0.01054 | loss-shift (~5%) |

Both align at **d* = 0** on the correct code (the loss-shift model does *not*
need `--deshift 1` here). The loss-shift model is marginally better in steady
state (plausibly its extra 2x-wide / 2048 capacity) but has a real cold-start
penalty (its warmup frames add ~+11%), so over the full clip the plain no-shift
model wins slightly. They are within ~5% either way.

---

## TL;DR
- The earlier "no-shift lags +1 frame" and "replicate fixes cold-start" findings
  were **both artifacts of the `main` `pad_left=2` train/inference mismatch**.
- On the trained `pad_left=1` code: no-shift recon is **frame-aligned (d* = 0)**,
  has **no meaningful cold-start**, and **plain zero (`constant`) padding is
  optimal** - replicate hurts the opening frames.
- Correcting the code (same weights) roughly halves measured L1 (0.0185 ->
  0.0108) purely by removing the spurious lag.
- No-shift and loss-shift are within ~5%; the zero-latency no-shift model with
  default zero-padding is the recommended configuration.
- Always run inference against a branch that has the
  `pad_left = max(0, (kernel-1)*dilation - (stride-1))` fix.
