# Conclusions

Metric: AutoBench background colour ΔE vs the sink, at 1 fps. A clip is **over** if ΔE stays above 1.0 for at least 5 s in a row. GT never exceeds 1.0.
Sets: **bb** = `aloe-h4-limited` (black-band chunks included; 7461 / 11190 / 14922) and **no-bb** = `aloe-h4-limited-no-blackbands` (6960 / 10440 / 13920), at 1 / 1.5 / 2 epochs, raw and EMA. Clips: seen16 without clip 10, so 15. Seed: 2026 + clip.

## Colour drift

- **Every checkpoint drifts.** Clips over (of 15):


| epoch | bb raw | bb EMA | no-bb raw | no-bb EMA |
| ----- | ------ | ------ | --------- | --------- |
| 1     | 12     | 7      | 8         | 9         |
| 1.5   | 12     | 10     | 9         | 4         |
| 2     | 13     | 10     | 8         | 9         |


- **Drift belongs to the clip, not the seed.**
  - Clips 03, 12, 13 and 14 are over in all 12 sets.
  - On the 8 worst clips of no-bb 13920 EMA, 20 runs with fresh seeds all went over. Median onset was 64 s.
  - With reproducible sampling, clip 03 of no-bb 13920 EMA went over and mosaicked under all 8 seeds.
    - Onset was 32–51 s, and 83–87% of the time was above 1.0.
    - Minute-5 background texture was 1.4–3.2, vs 0.27 for GT.
    - The seed only sets how bad it gets: minute-5 ΔE ranged from 2.2 to 5.3.
    - In every seed a colour cast comes first; the visible blocks appear at about 120–180 s. Results: `/home/vansh/eval/seeds8/`.
- **Recovery is temporary.**
  - 35–81% of episodes come back below 1.0, with a median recovery time of 12–49 s.
  - Only 0–5 clips per set end below 1.0.
  - There's no trend with epoch or with raw vs EMA. Script: `/home/vansh/eval/recovery/recovery.py`.
- **Not precision.** Fully fp32 sampling on clip 03 (bb raw 14922) drifts and mosaics like bf16.
- **Not visible in eval loss.** The one-step eval loss on GT context falls at every EMA checkpoint in both runs.
  - Seen loss: bb 0.1205 → 0.1190; no-bb 0.1222 → 0.1206. EMA is below raw at every step.
  - It also falls from no-bb EMA 10440 to 13920, while clips over go from 4 to 9. Results: `/home/vansh/eval/ema_eval/summary.txt`.



## Black bands

- Dropping the black-band data fixed clip 10: in no-bb it never goes dark, at any checkpoint, raw or EMA.
- The mosaic remains.



## Which history slots get copied

No-bb 13920 EMA, one AR step with corrupted history, 8 clips × 5 times, trained colour and grid corruption. Copy α is the fraction of the corruption that appears in the finished frame. The no-copy level is |α| ≈ 0.006. Shapes run oldest → newest slot, 1 = corrupted.

- **The model copies from the newest slots.** The leading shapes (0001, 0011, 0111) are copied; the trailing shapes (1110, 1100, 1000) are not.
- **Colour:** 0001 is copied most, at +0.02–0.07 (high amplitude) and +0.05–0.09 (low).
  - The training shape 1111 is at the no-copy level.
  - Every shape with a clean newest slot gives a negative α, down to −0.05, i.e. the model pushes away from the corruption.
  - Each extra corrupted slot behind the newest lowers the copy. DARK 0001 / 0011 / 0111 / 1111: +0.024 / +0.021 / +0.013 / +0.002 (high), +0.054 / +0.038 / +0.020 / +0.001 (low).
  - The push-away is largest for 1100: −0.032 to −0.048 across the three directions and both amplitudes.
  - A random direction is copied more than DARK or DRIFT: 0001 gives +0.067 / +0.090 vs +0.024–0.029 / +0.054–0.057.
  - Halving the amplitude roughly doubles α, so the absolute colour copied is about the same. DARK 0001: 0.024 × 0.5 vs 0.054 × 0.25, i.e. 0.012 vs 0.014.
  - 0001 copies on every clip: the per-clip minimum is +0.016 (high) and +0.037 (low).
- **Grid:** 0011 is copied most, at +0.26 (high) and +0.42 (low). 0001 gives +0.24 / +0.28, and the oldest slots alone give ≈ 0.
  - Weaker grids are copied more: 1111 goes from +0.10 at high amplitude to +0.33 at low.
  - Clip spread is large: 1111 at low amplitude ranges from +0.16 to +0.65 across clips, while 0001 is the tightest, at +0.23 to +0.33.
- Results: `/home/vansh/eval/corruption_viz/slot_probe_v2/table.txt`.



## Reproducibility

- **Before the fix, sampling wasn't reproducible.** The same checkpoint, clip and seed gave different outputs per process and per node, starting from the first denoising step.
  - Outputs decorrelate within about 13–19 s.
  - So per-clip differences between the 12 sets include this noise. For example, compiled vs eager bf16 on clip 03 gives a colour onset of 58 vs 126 s.
- **Cause:** `torch.compile` picks each Triton kernel's tile config by timing it at runtime. In the norm reductions (RMSNorm, layer-norm mean/variance), the config changes the order of the sums.
  - Seeds, noise, inputs, checkpoints, the environment, cuBLAS, cuDNN and FlashAttention are all deterministic.
- **Fix** in `production/cached_universal.py`: always use the first candidate config, and tag the cache key so caches with timing-picked configs aren't read.
  - Fresh runs on r4 (two GPUs), r3 and r2 are bit-identical, at the same speed (365 ms/frame).
  - Production `main` (clip 03, 30 frames, cold, fresh cache per run) is bit-identical on r4, r0 (two GPUs) and r3, at 367–380 ms/frame.
- **Caveats:**
  - Noise depends on clip length, so a truncated run gets different noise for the same seed.
  - Eager and compiled never match each other.
- **Checks** in `/home/vansh/eval/repro/`:
  - `go.sh` does one run (`REPRO_AUTOTUNE=log` for the default, `first` for the fix).
  - `compare.py` prints the first differing frame and step.
  - `hash_ckpts.sh` hashes the checkpoints.

