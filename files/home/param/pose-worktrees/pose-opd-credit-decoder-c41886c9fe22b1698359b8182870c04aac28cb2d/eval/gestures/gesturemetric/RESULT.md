# gesturemetric — gesture–speech coupling, (1) timing + (3) cheap meaning: RESULT (10-26)

CPU only. Pose → strokes uses the same decode as `run_stats.py` and `hands.py`'s wrist speed. Words come from the data's own word files for real Aloe and from the model's word lane for generated runs. Whisper is never used. Code: `gm.py` (definitions), `real_extract.py`, `gen_extract.py`, `loud_extract.py`, `analyze.py`, `variants.py`, `gm_run.py` (per-run, for the playground). Numbers come from `out/analysis.json` and `out/variants.json`.

## Definitions
- **Stroke**: a peak of the 0.2 s-smoothed wrist speed of either hand. It must reach ≥ 0.015 shoulder-widths per frame (= hands.py MOVE_THR), have prominence ≥ half that, and sit ≥ 0.3 s from the previous peak. L and R peaks within 0.2 s are merged.
- **(1) TIMING κ (headline)**: hit = share of strokes within ±0.2 s of one of the same speaker's word onsets. Chance = the same with the words circularly shifted inside the clip by 3 s to (L − 3 s), K = 200 shifts. **κ = (hit − chance)/(1 − chance)**, clipped to [−1, 1].
  - 0 = strokes unrelated to the speech (random flailing or mismatched speech).
  - Also reported: "loud" = the same against loudness peaks of the speaker's audio (real: CPU Mimi decode of the Aloe codes; generated: b.wav; checked identical to b.wav, r = 1.000, 0 ms).
  - Also reported: "fine" = hit minus hit with every stroke jittered ±1 s. This isolates sub-second alignment from "gestures while talking".
- **Lag check (real, pooled)**: the excess of stroke-minus-onset over the shifted null peaks at **+0.04 to +0.08 s** (stroke just after the onset). So a ±0.2 s window centred at 0 is right. A sweep of threshold {0.015, 0.025, 0.04} × window {0.1, 0.2, 0.3} gives the same picture. 0.015 / ±0.2 is kept (per-clip AUC real vs shuffled 0.71).
- **(3) cheap meaning**:
  - (a) the proposed deictic/self/size/negation word categories vs other content words.
  - (b) **substance lift** = stroke rate in [onset − 0.5, onset + 0.3] s on substantive words ÷ the rate on fillers and backchannels (yeah, um, oh, okay, right…).
  - (c) a data-driven word lexicon learned on half the real conversations and tested on the other half.

## Data
- REAL: 120 tracks = 60 held-out (POSE_INDEX eval) Aloe conversations × both sides, 10 min each, cut into 2,562 clips of 18 s.
- SHUFFLED: the same strokes against the same side's words from a different conversation.
- LISTENER: own strokes against the other speaker's words.
- GENERATED: 425 sources: 279 genweb runs and 146 live calls; live calls are cut into 18 s windows; prefill excluded. 84 runs with < 7 s after prefill were skipped (8 s duplex test clips). user = not hidden and not the test client.
- Loudness for real covers the first 240 s of all 120 tracks (CPU Mimi decode, finished 10-27 00:26 UTC).

## (1) TIMING κ on 18 s clips (mean ± sem)
| set | clips | κ words | κ loud | fine pp | strokes/min |
|---|---|---|---|---|---|
| **real Aloe** | 2562 | **+0.28 ± 0.01** (p10–p90 −0.15…0.82) | +0.31 ± 0.01 | +5.2 | 34.5 |
| **shuffled (other conv's words)** | 2562 | **−0.01 ± 0.01** | −0.01 ± 0.01 | +0.7 | 34.5 |
| listener's words | 2562 | −0.16 ± 0.01 | – | +1.1 | 34.5 |
| real, whole 10-min tracks | 120 | +0.35 ± 0.01 (shuffled +0.01) | +0.37 | +3.8 | 35.0 |
| **generated, all user-made** | 432 | **+0.39 ± 0.02** | +0.43 ± 0.02 | +8.9 | 30.0 |
| monoplex export-357, user | 247 | +0.47 ± 0.03 | +0.51 ± 0.03 | +10.9 | 29.9 |
| monoplex export-357, hidden | 74 | +0.22 ± 0.07 | +0.35 ± 0.06 | +5.9 | 27.7 |
| live calls, pose-duplex v5 export-498, user | 147 | +0.31 ± 0.04 | +0.35 ± 0.04 | +6.9 | 30.9 |
| livedup runs v5, user | 28 | +0.29 ± 0.08 | +0.23 ± 0.07 | +9.4 | 30.3 |
| livedup runs v5, hidden | 25 | +0.48 ± 0.07 | +0.44 ± 0.08 | +20.7 | 24.9 |
| live calls step-dcp, user / hidden | 4 / 25 | +0.18 ± 0.29 / +0.08 ± 0.16 | | | 8–10 |
| **pose duplex export-498 offline runs, hidden** | 6 | **+0.02 ± 0.08** | +0.01 ± 0.12 | +2.3 | **77.5** |

What the table shows:
- **Real is clearly above chance, and shuffled sits at 0.** Pairing a person's strokes with the wrong conversation's speech drops κ from 0.28 to 0. The listener's speech gives a negative κ, because people gesture less while the other person talks.
- **The flailing case is scored as flailing.** The offline pose-duplex runs (78 strokes/min, 2× real) have κ ≈ 0; example run dup0 has 102 strokes/min and κ 0.05.
- The user's own generated runs and calls are at or above real. Live calls are 0.31, about real-like. Monoplex is 0.47: its "fine" alignment is about 2× real's (+10.9 vs +5.2 pp), so its gestures are more tightly locked to word onsets than humans' are.
- **Per-run noise**: a single 18 s real clip ranges from −0.15 to 0.82 (p10–p90). One playground run can't be judged on its own; compare settings over about 10 or more runs (sem ≈ 0.1 at n = 10).

## (3) cheap meaning
| set | proposed 4 categories: lift vs other content words | substance lift (substantive / filler) | learned lexicon, held-out |
|---|---|---|---|
| real | **0.95 [0.93, 0.97]** (deictic 0.99, self 0.93, size 1.03, negation 0.94) | **1.74 [1.66, 1.81]** | **1.53 / 1.60** (two halves) |
| shuffled | 0.99 | 1.11 [1.06, 1.16] | 1.08 / 1.04 |
| listener | 1.02 | 0.55 | – |
| generated user-made | 1.04 [1.00, 1.08] | 1.56 [1.45, 1.69] | 1.48 [1.24, 1.79] |
| monoplex user / live calls user | 1.04 / 1.00 | 1.71 / 1.30 [1.13, 1.50] | |
| duplex export-498 hidden | – | 1.02 | |

- **The proposed deictic/self/size/negation categories fail on real data.** Real Aloe speakers don't stroke more on those words (lift 0.95), so they can't score generated pose. Dropped.
- **Substance lift works.** Real speakers gesture 1.74× more on substantive words than on fillers and backchannels (shuffled 1.11). The learned lexicon replicates on held-out conversations (1.5–1.6 vs shuffled 1.04–1.08).
  - Generated (user) runs reach 1.56: monoplex 1.71, about real; live calls 1.30, weaker. The flailing duplex runs are at 1.02.
  - Caveat: this is a pragmatic "content vs backchannel" signal, partly tied to turn holding. It is not lexical semantics; that is what (2) is for.

## Per-run scorer (playground)
`gm_run.py RUN_DIR` (CPU, ~2 s per run once warm) outputs `gesture_timing_kappa`, `gesture_timing_loud`, strokes/min, hit counts and substance counts, plus the real and shuffled 18 s references from `out/analysis.json`. **Live since genweb v-79 (aloedemo, 10-26 ~22:55 UTC).** `gm_backfill.py --loop 30` (gm-cpu job 30753, nice 19) writes `runs/<id>/gesture_stats.json` atomically for every run with status done, hidden runs included. Backfill: 273 runs written; 2 runs with empty pose are skipped. The page shows "timing κ X (real 0.28)". The writer lives in the 12 h holder, which ends ~09:37 UTC 10-27. Successor job 30760 (gm-backfill, slinky-4, 1 CPU, 6 G, 3 days, afterany:30753) takes over automatically.

## (2) meaning judge — CPU-mini result (10-27 00:02 UTC)
No GPU pair was free. A trial of 24 CPUs inside motasr on slinky-18 slowed motasr from 8.9 to 13 s per step, and the guard stopped it in 40 s. So everything ran on the 4-CPU slot. Code and data are in `judge/` (`STATUS2.md`, `score2_base.json`).
- Setup: 0.87 M params. Train = 150 convs × 2 sides × 5 min (25 side-h). Pose = decoded kinematics with the window mean removed. Words = Qwen3-0.6B layer-14 features → 128 dims. Audio = own Mimi codebooks 0-7.
- Held-out top-1 among 16 moments (chance 6.25 %): full **10.2 %**, words only 9.6 %, audio only 8.8 %, shuffled 6.9 %, pose shifted 2 s 9.3 %.
- Words-only beats chance, so there is some lexical signal. But a 2 s pose shift keeps about 90 % of the effect, so the judge mostly sees slow context, not moment-level meaning.
- Generated runs, 20 s protocol (real 11.3 %, shuffled 7.0 %): all user-made 9.6 %, about 0.44 of real. Per group (monoplex 0.73, live calls 0.25, livedup ~0, flailing duplex at chance) the noise is ±0.5–1.5 pt, so these are not reliable.
- **Verdict: too weak for a per-run score.** It needs the full plan below on GPU with far more data (25 of 2,379 side-h used); more CPU epochs plateau at 9.6–10.1 %.

## (2) meaning judge — FULL GPU run (job 30799, slinky-10 GPUs 0-1, 10-27 00:33–01:10 UTC)
Setup: 7.07 M params (d256, 4 layers). Train = 1,000 train convs × 2 sides × 10 min (274 side-h). Eval = 60 held-out convs × 2 sides (16 side-h). Same features and protocol as the CPU-mini run. Files: `judge/eval2_full.json`, `judge/score2_full.json`, `judge/STATUS_FULL.md`.
- **Held-out gates (top-1 among 16, chance 6.25 %):** full **52.9 %**, words only 49.2 %, audio only 46.2 %, shuffled 6.6 %, **pose shifted 2 s 9.6 %**, tight (16 moments within 1–2 min) 50.6 %.
- On real data the judge now tracks the exact moment: a 2 s shift sends it to near chance.
- **Generated, 20 s protocol (real 50.6 %, shuffled 6.5 %; norm = (acc − shuffled)/(real − shuffled)):**

  | group | accuracy | norm |
  |---|---|---|
  | all user-made | 43.6 % | **0.84** |
  | monoplex, user | | 0.88 |
  | live calls v5, user | | 0.87 |
  | livedup v5, user | | 0.53 |
  | all hidden | | 0.47 |
  | duplex export-498 (the flailing runs, κ ≈ 0) | | 0.41–0.46 |
- **Caveats before calling this "meaning":**
  - (a) The words-only branch still carries word-onset *timing*, so 49 % words-only is not pure semantics. Needed: an eval-only ablation with word positions shuffled inside the window (CPU, cheap).
  - (b) Pose includes the head points (nose, eyes, ears). The flailing duplex runs still score 0.41–0.46 while their timing κ is ≈ 0, so the judge may be reading head/speech rhythm rather than hands. Needed: a hands-and-arms-only retrain (~35 min on a GPU pair).

### Ablations (10-27 01:15–01:48 UTC)
- **Eval-only checks on the full judge** (CPU, `judge/ablate_full.json`, chance 6.25 %):
  - as-is 52.9 %
  - words-only with word positions shuffled inside the window 16.5 % (3 seeds); bag of words 18.3 % (words-only as-is 49.2 %)
  - head points zeroed 10.6 %; head points only 39.4 %
  - So most of the words-only signal is onset *timing*. Word identity adds ~10–12 pt above chance. The full checkpoint leans mostly on head motion.
- **Hands/arms-only retrain** (job 30824, slinky-10, `--pose-keep 5-10,13,14` = shoulders, elbows, wrists, hand centroids, hand spreads; same data and recipe; `judge/eval2_hands.json`, `judge/score2_hands.json`):
  - held-out **27.0 %** (shuffled 7.2 %, pose shifted 2 s 9.0 %, words-only 29.6 %, audio-only 22.7 %, tight 25.5 %)
  - **Hands alone carry a real, time-locked speech signal: ~4× chance**, about half of what head + hands give.
- **Generated, hands-only (norm vs real 26.3 % / shuffled 6.7 %):**

  | group | norm |
  |---|---|
  | all user-made | **0.83** |
  | monoplex, user | 0.93 |
  | live calls v5, user | 0.73 |
  | livedup v5, user | 0.72 |
  | all hidden | 0.45 |

  - The flailing duplex export-498 runs read 0.87 (hidden) and 0.30 (user), from only 72 / 41 windows (±5 pt), so they are inconclusive. This is still at odds with their timing κ ≈ 0. The judge likely uses coarser 4 s activity-vs-speech cues on generated pose; treat it as a pooled model-level score, not a per-run one.

## (2) meaning judge — full plan (GPU)
See `PLAN_MEANING.md`.
- Contrastive 4 s pose-kinematics ↔ own speech (frozen local Qwen3-0.6B words + Mimi cb0-7), ~8–10 M params, trained on real Aloe train convs only.
- Negatives come from the same conversation, plus hard negatives shifted 1–3 s. Score = top-1 among 16 moments, normalised so 0 = shuffled and 1 = real eval.
- Needs 2 GPUs (1 suffices) for ~2 h. **No free GPU pair now**: every non-slinky-9 GPU belongs to training or live jobs. Start only when a pair frees; first candidate is posefix-phys's pair on slinky-4 when that job ends.
