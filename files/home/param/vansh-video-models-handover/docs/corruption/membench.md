# MemBench

A benchmark for whether an autoregressive (AR) talking-person video model holds a person better the more footage of
that person it saw in training. It is scored with ARBench (`docs/corruption/autobench.md`).

## Why

Training footage is spread unevenly over actors: some appear for minutes, some for tens of hours, some not at all. At
inference the model gets one real frame of the person (the sink) and has to keep generating that person for minutes.
Two questions follow.

- **Memory.** If the model leans on memorised actors, identity should hold better and drift be smaller for actors with
  many training hours, and worst for actors it never saw. If it works from the sink alone, scores should not depend
  on hours.
- **Seeds.** Is drift a property of the clip (actor, audio, pose) or of the noise draw? Running every clip with two
  seeds separates the two.

## How it is built

- **40 actors, one 5-minute clip each,** all in the same studio setting (plain grey backdrop, fixed frontal camera,
  checked on the first frame), so background and lighting do not change with hours.
- **0 to 21 training hours** (hours = the actor's total footage in the training set). 3 actors never seen in training
  (0 h), and 37 seen actors, each the nearest unused actor to one of 37 evenly spaced targets from 0.08 h to 21 h.
- **The spread is uneven at the top.** 0–8.5 h is evenly covered (about one actor per 0.6 h), but few actors have
  more than 12 h: all 4 that exist are included, and 18 of the 40 sit at 8.5–12 h. Median 8.7 h.
- **Bins** for reporting: 0 h (3 actors), 0–5 h (9), 5–9 h (9), 9–10 h (9), 10–21 h (10).
- **The clip** is a random 5-minute chunk of the actor's footage, fixed per actor. For seen actors it comes from the
  training set, so the model may have trained on that exact clip.
- **2 seeds per clip**, 2026 + i and 2066 + i for clip i: 80 generations per checkpoint.

## Metrics

The five ARBench metrics, each scored every second against the generation's own sink: M1 background colour drift, M2
background texture drift, M3 person colour drift, M4 person texture drift, M5 identity (ArcFace cosine, 0.8 = same
appearance). Definitions and real-footage values are in ARBench. MemBench reads them per actor, averaged over the two
seeds, against hours.

## How to read it

- **Memory = trend against hours.** Take the Spearman rank correlation of each actor's seed-averaged score (minute 5,
  or the clip mean) with hours. Memory shows as identity rising and drift falling with hours, with the 0 h actors at
  the bottom. With 40 actors, |ρ| below about 0.3 is indistinguishable from no trend (p > 0.05).
- **Identity level and identity decline are separate.** M5 is set within the first minute, then loses a few
  hundredths. Memory could lift the level, slow the decline, or both; read the minute-1 level and the minute 1 → 5
  change separately.
- **Pass rate.** Count the actors whose seed-averaged identity reaches 0.8, overall and per bin.
- **Read identity against the actor's own real footage.** Real clips differ (head pose, a hand over the face in the
  sink frame), so a low score can come from the sink rather than from forgetting. Scoring the real clip the same way
  gives each actor's own level (real studio footage sits around 0.82); a generation that tracks it is limited by the
  footage, not by memory.
- **Bins are small.** The 0 h bin has 3 actors and the top of the range is sparse, so one bad sink or drifting clip
  moves a bin mean. Check per-actor values before reading a bin trend.
- **Seeds.** Per actor, compare the two seeds' scores and whether a drift trigger (for example minute-5 background
  colour above 2 ΔE) fires under both, one or neither. Both = the clip drifts; one only = the noise draw.
- **Look at the frames** of any outlier: M1 and M2 jumping together is usually a local artifact such as mosaic.

## Running it

Generate the 80 samples at full 5-minute length and label them as for ARBench (full-quality real | generated
side-by-sides, plus a `manifest.csv` with the sample index, clip, uuid, seed and the actor's hours). Score them with
the real-footage variant of ARBench, so each actor's real clip is scored too:

```bash
.venv/bin/python /home/vansh/eval/bench/seen16_14922/arbench_gt.py IN_DIR OUT_DIR --gpus 0,1,...
```

- **Compute:** generation takes about 35 min on 40 GPUs (one clip per GPU, both seeds in turn); scoring both halves
  about 35 s per sample per GPU.
- **Numbers:** `/home/vansh/eval/bench/membench_csv.py` gathers the per-sample JSONs into one long-format CSV of the
  1 fps curves with each actor's hours and seed.
