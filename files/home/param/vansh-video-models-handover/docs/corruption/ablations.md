# Ablations: baseline to final model

One change per run, from the baseline to `aloe-h4-complete-corruption`. Method: [history-corruption.md](history-corruption.md).

## Conventions

- **Metrics** are [AutoBench](autobench.md), scored at 1 fps against the generation's own sink (frame 0). Background ΔE = AutoBench M1.
  - **End ΔE:** background ΔE over 293–297 s, mean over clips (median where marked). GT 0.19 (median 0.15).
  - **Minute-5 ΔE:** background ΔE over 240–299 s, mean over clips.
  - **Clips over:** clips with background ΔE > 1.0 for ≥ 5 s in a row. GT: never.
  - **Seconds over 1.0:** seconds with background ΔE > 1.0, summed over clips (of 4,500 for 15 clips).
  - **Mosaic clips:** clips with some minute 2–5 at background ΔE > 2 and background texture (M2) > 1.
  - **Identity:** M5, ArcFace cosine vs the sink, over the same seconds as the ΔE quoted with it. GT ≈ 0.82.
- **Clip sets.** Numbers on different sets are not comparable.
  - **studio13:** the 13 studio clips of the 30-clip set. Baseline only.
  - **seen16:** 16 studio chunks from `aloe-h4-limited-eval-seen`. Clip 10 is a data defect, so every number uses the other 15. All runs from bb on.
  - **MemoryBench:** 40 studio actors (0–21 h in training) × 2 seeds ([MemoryBench](membench.md)).
- **Checkpoints** are compared at equal epochs: 1.5 epochs = bb 11190, no-bb / cc / mosaic 10440; 2 epochs = bb 14922, no-bb / cc 13920. EMA unless marked raw.
- Plain sampling, seed 2026 + clip. Before the reproducibility fix (below), a seed diverged across processes within 13–19 s, so per-clip differences between runs are partly noise. Compare counts and means.

## Runs

**0. `aloe-h4/run_3` (no corruption).** Stopped at step 6,522 and never sampled, so there is no no-corruption reference.

**1. Baseline: `aloe-h4-corruption/run_1`.**
- Change: HistoryCorruption p 0.4 (colour offset + Nyquist checkerboard) and history dropout; data `aloe-h4` (40,378 chunks, 3,365 h), batch 2,400, 12,584 steps (2 epochs).
- studio13: minute-5 ΔE 1.75 / 1.56 / 1.25 (step9438 EMA / 12584 EMA / 12584 raw; real < 0.4); identity 0.63–0.64. Drift is on a few clips (04, 07, 11, 19, 24), starting at minute 2–3, often as coloured mosaic patches along the top of the wall. Checkpoint and weights move it between clips, they do not remove it.
- MemoryBench (12584 raw): training hours predict no score (identity Spearman +0.22, p 0.17); 8 of 40 actors reach identity 0.8; seeds agree (identity within 0.03 on 39 of 40).
- Led to: adaptive sampling at inference (below).

**2. bb: `aloe-h4-limited/run_1`.**
- Change: data capped at 2 h per actor (`aloe-h4-limited`: 25,537 chunks, 2,128 h); batch 2,400 → 1,280; 14,922 steps. Data, batch and clip set all change, so baseline vs bb is confounded.
- seen16, 1 / 1.5 / 2 epochs: end ΔE median EMA 0.52 / 0.83 / 0.78 vs raw 1.20 / 2.13 / 1.76 (GT 0.15); clips over EMA 7 / 10 / 10, raw 12 / 12 / 13. Mosaic on 03 and 13 in most variants, and on 6 clips in raw 11190.
- Clip 10 goes dark at the same seconds in GT and in all six variants (minute-5 ΔE ~10.5–11): its GT latents contain dark frames with frozen pose (source upload holes). Some training chunks have the same stretches.
- Led to: EMA weights for every later comparison; no-bb drops the black-band chunks.

**3. no-bb: `aloe-h4-limited-no-blackbands/run_1`.**
- Change: drops chunks in the upload-holes list and chunks of speakers flagged by the pose-integrity audit: 23,822 of 25,537 chunks (−143 h); 13,920 steps.
- seen16: clip 10 fixed (no dark seconds; minute-5 ΔE 0.8–1.2). Mosaic not fixed: 03, 13, 14 (mosaic clips EMA / raw 3 / 5 at 13920). Clips over EMA 9 / 4 / 9 at 1 / 1.5 / 2 epochs. EMA 10440 is the least-drifting plain set of all runs (end ΔE 0.73, median 0.43; 561 s over). Eval loss does not see drift: from EMA 10440 to 13920 it falls (seen 0.1208 → 0.1206) while clips over go 4 → 9.
- Led to: clean data does not remove the mosaic, so the next two runs each change only the corruption, on top of no-bb: slot profiles (cc) and mosaic corruption (mosaic).

**4. cc: `aloe-h4-cc/run_1`.**
- Change: slot profiles. A corrupted window hits all real history slots (40%), the newest n (30%) or the oldest n (30%), n ~ U{1, 2, 3}, with per-slot masks. p 0.4 → 0.556 keeps 1.48 corrupted history frames per window. Still checkerboard. These are the partial windows that corrupt-and-cache produces at inference.
- seen16, 2 epochs, cc vs no-bb: EMA clips over 8 vs 9, seconds over 705 vs 1,672, end ΔE 0.96 vs 1.65, mosaic clips 0 vs 3. Clip 13 is never over; 03, 12, 13 and 14 had been over in every earlier set. Onsets: 03 at 217 vs 56 s, 14 at 198 vs 54 s. Clip 04 is worse (end ΔE 3.22 vs 1.30). Raw is worse: 11 vs 8 clips over, 1,728 vs 1,389 s. At 1.5 epochs the EMA gain does not hold: 8 vs 4 clips over, 975 vs 561 s.
- Sheet: [clip 14, no-bb vs cc, EMA 13920](../../deliverables/aloe-h4-cc/sheets-vs-nobb-13920-ema/seen16-nobb-vs-cc-13920-ema-clip14-15s.png).
- Led to: slot profiles and per-slot masks are kept in the final config.

**5. mosaic: `aloe-h4-mosaic/run_1`.**
- Change: mosaic corruption replaces the checkerboard. One kind per window: colour 30%, mosaic 15%, clean 55%. Mosaic amplitude 1.0–1.5, on the top 70–100% of the pose-hull background. Data is streamed from Hugging Face. Built on no-bb, so no slot profiles. Crashed at step 11,501, so its last checkpoint is 10440 (1.5 epochs); it is compared with no-bb and cc at 10440.
- seen16, 1.5 epochs, mosaic / no-bb / cc: EMA clips over 7 / 4 / 8, seconds over 576 / 561 / 975, end ΔE 0.71 / 0.73 / 0.89. Raw clips over 7 / 9 / 10, seconds over 761 / 2,092 / 1,278, end ΔE 0.94 / 1.80 / 1.18, mosaic clips 1 / 4 / not scored. Mosaic is the best raw set of all runs; its EMA matches no-bb on seconds and end ΔE, with 3 more clips over.
- Sheet: [clip 01, no-bb vs mosaic, raw 10440](../../deliverables/aloe-h4-mosaic/sheets-vs-nobb-10440-raw/seen16-nobb-vs-mosaic-10440-raw-clip01-15s.png).
- Led to: mosaic corruption is kept in the final config, with a larger share and a wider amplitude range.

## Comparison (seen16, 15 clips, plain sampling)

Each cell is EMA / raw; "–" = not scored. The baseline is not in the table: it was scored on studio13.

| Run | Checkpoint | Clips over | Seconds over 1.0 | End ΔE | Mosaic clips | Identity |
|---|---|---|---|---|---|---|
| bb | 11190 (1.5 ep) | 10 / 12 | 1,393 / 1,634 | 1.15 / 4.21 | – | 0.717 / 0.673 |
| no-bb | 10440 (1.5 ep) | 4 / 9 | 561 / 2,092 | 0.73 / 1.80 | 0 / 4 | 0.723 / 0.692 |
| cc | 10440 (1.5 ep) | 8 / 10 | 975 / 1,278 | 0.89 / 1.18 | – | 0.719 / 0.689 |
| mosaic | 10440 (1.5 ep) | 7 / 7 | 576 / 761 | 0.71 / 0.94 | 0 / 1 | 0.724 / 0.712 |
| bb | 14922 (2 ep) | 10 / 13 | 1,214 / 2,631 | 1.09 / 2.48 | 0 / 6 | 0.727 / 0.705 |
| no-bb | 13920 (2 ep) | 9 / 8 | 1,672 / 1,389 | 1.65 / 2.10 | 3 / 5 | 0.728 / 0.712 |
| cc | 13920 (2 ep) | 8 / 11 | 705 / 1,728 | 0.96 / 2.08 | 0 / 2 | 0.727 / 0.703 |
| GT | – | 0 | 0 | 0.19 | 0 | 0.83 (240–299 s) |

## Inference ablations

Adaptive sampling: once per generated second, the background ΔE of the newest frame is checked against the sink. While it is above a threshold, the history is corrupted with the run's own HistoryCorruption block (`production/adaptive_sampling.md`).

- **Threshold 2.0** (baseline 12584 raw): holds the studio drift on clip 07; saves 2 of 15 app clips.
- **Threshold 1.0 vs matched controls** (control = the same rollout resumed without corruption from the first onset). Baseline 12584 raw, studio13: minute-5 ΔE 0.61 vs 1.33, texture 0.41 vs 0.72, identity 0.63 in both; the 04 and 07 mosaics are gone. bb 14922 EMA, seen16: minute-5 ΔE 0.73 vs 1.11; 03 and 13 are partly fixed. The trigger is colour-only, so texture drift stays.
- **Placement** (no-bb 13920 EMA, 8 drifting seen16 clips; plain predates the reproducibility fix, so it is not a matched control). Rewriting all 4 slots with a fresh draw on every active step (rewrite4) is best: minute-5 ΔE 0.90 vs 3.20, texture 0.53 vs 1.28, with 8% of steps corrupted. Corrupting only the newest 1 or 2 slots holds colour (0.97 / 0.93), but 03 and 13 keep their mosaic. Guards add nothing (guard1 runs away on 06, ΔE 21).
- **rewrite4 vs corrupt-and-cache** (seen16, end ΔE). Corrupt-and-cache commits each frame once as x + d, with one draw held for the whole episode. On cc 13920 EMA it matches plain until the first onset on 16 of 16 clips. Results: clips over 8 → 4, seconds over 705 → 236, end ΔE median 0.83 → 0.71, but mean 0.96 → 0.95, because two held draws never end (03 runs away into mosaic, end ΔE 4.40; 13 reaches 1.82). rewrite4 on mosaic 10440 EMA: clips over 7 → 5, seconds over 576 → 164, end ΔE 0.71 → 0.63; its plain run was sampled on other nodes, so it is not a matched control. These are different checkpoints, so this is not a head-to-head. rewrite4 on cc and corrupt-and-cache on no-bb were not run.

## Probes that shaped the corruption

- **Checkerboard calibration.** The real mosaic is broadband (Nyquist ≈ 2% of its energy), with non-DC RMS 0.65–1.6 vs the training grid's maximum of 0.25. The model already ignores grids at every strength up to 1.5 (copy α ≈ 0.03), but copies broadband mosaic-like patterns held in history (α 0.26–0.32). So the checkerboard was replaced by a mosaic generator fitted to 13 real mosaics.
- **Slot probe** (baseline 12584 raw, clips 07 and 04). The model copies spatial patterns almost only from the newest history slot: slots 1–3 alone give |α| ≤ 0.01; the newest slot alone gives 0.27 / 0.36, vs 0.26 / 0.32 for all four. Colour: newest slot alone 0.88; a uniform offset on all four slots (the training shape) 0.07–0.15. Re-check on no-bb 13920 EMA: colour 1111 is at the no-copy level, and 0001 is copied most. Led to slot profiles that corrupt the newest slots (cc); in the final config, the newest slot is corrupted in ≈ 51% of windows.
- **Drift onset vs training strength** (5 EMA sets, seen16). In the 10 s before an onset, background ΔE has a median of 0.80–0.94. Only 8–10% of training corruption draws are that weak (median draw 3.0–3.7 ΔE). Weaker corruption is copied more (grid 1111: α +0.10 at high amplitude, +0.33 at low). Led to the lower colour and mosaic minimums of the final config.
- **Seeds and reproducibility.** Drift belongs to the clip, not the seed: clip 03 (no-bb 13920 EMA) goes over and mosaics under all 8 seeds, and 20 fresh-seed runs on the 8 worst clips all went over. `torch.compile` autotuning made the same seed diverge across processes within 13–19 s; always taking the first Triton config fixes it and makes matched controls possible.
- **Ruled out:** precision (fp32 sampling of clip 03, bb raw 14922, still drifts and mosaics); a per-actor link between mosaic and upload-hole chunks (none found).

## Final model: `aloe-h4-complete-corruption` (not launched)

Config: `experiments/universal/aloe-h4-complete-corruption.toml`. Method and values: [history-corruption.md](history-corruption.md).

- **From the baseline:** model and init (`h6-squeaky/run_3/step44675` EMA), window `[sink | 4 history | 5 present]`, colour corruption (DARK / DRIFT / random directions, wall mask), history dropout [0.1, 0.1, 0, 0], EMA 0.999.
- **From bb:** data capped at 2 h per actor; EMA weights for evaluation.
- **From no-bb:** `aloe-h4-limited-no-blackbands` (no black-band chunks).
- **From cc:** slot profiles [0.4, 0.3, 0.3] and per-slot masks.
- **From mosaic:** mosaic corruption, one kind per window, no checkerboard; data streamed from Hugging Face.
- **New in the final config:**
  - shares: mosaic 23.3%, colour 46.7%, clean 30% (mosaic run: 15 / 30 / 55);
  - lower minimums: person colour 0.03 → 0.018, background colour 0.1 → 0.059 (weakest DARK / DRIFT draw ≈ 0.6 ΔE), mosaic 1.0–1.5 → 0.35–1.8;
  - directions DARK / DRIFT / random 0.4 / 0.4 / 0.2 (was 0.4 / 0.3 / 0.3);
  - 3 epochs (was 2), min lr 8e-6 (was 2e-5); batch 1,296 (6 nodes, 3 × 9; was 1,280 at 4 × 8).
- **Inference:** adaptive sampling at ΔE 1.0 with this config's corruption block. Whether to use rewrite4 or corrupt-and-cache is still open.
