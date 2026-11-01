# gesturemetric (2) CPU-mini meaning judge — STATUS
- 10-26 22:25 UTC: building prep2.py (pose kinematics + Qwen3-0.6B word states + Mimi cb0-7), measuring per-track cost.
- 22:37 UTC: per-track prep measured 7-13 s (5-min track: pose decode 4-6 s + Qwen3-0.6B layer-14 contextual word states 3-6 s, 3 CPUs) -> contextual states kept (no static fallback).
  Data: TRAIN 150 train convs x 2 sides x 5 min (25 side-h), EVAL 60 held-out convs x 2 sides x 5 min (10 side-h).
  ETA: prep done ~23:30 UTC, training ~45 min -> eval gates ~00:15 UTC, generated scores ~00:40 UTC (17:40 PT).
- 22:41 UTC: slinky-18 side step (6 prep shards in motasr 30741, coordinator shape) started; the guard touched STOP_S18 within ~40 s -> all 6 shards exited cleanly after 1 track each; NOT relaunched there. Prep continues on the 4-CPU slot only (3 CPUs, ~5 s/track, 55/420 tracks done).
  ETA (slinky-4 only): prep done ~23:15 UTC, training ~30-40 min -> eval gates ~23:55 UTC, generated scores ~00:20 UTC (17:20 PT).
- 23:13 UTC: prep done (300 train + 120 eval sides). 23:14 training base model (4000 steps max, early stop on eval).
- 23:23 UTC: base judge: eval top-1@16 10.2 % (chance 6.25, shuffled 6.9, words-only 9.6, audio-only 8.8, pose +2 s 9.3, tight 9.5); early-stopped at 1.5k steps (patience too short). 23:26 retraining base2 (5k steps, patience 4) + scoring generated with base in parallel.
- 23:37 UTC: generated scored with the base judge (score2_base.json). Same 20 s-source protocol everywhere (16 candidates: same-source windows >= 4 s away, topped up from the same pool).
  REAL eval: top-1@16 11.3 % (words-only 11.1, audio-only 10.4), shuffled 7.0 % -> judge is above chance but WEAK (+4-5 pp).
  GENERATED user-made 9.6 % (shuffled 7.7) -> (acc - shuf) / (real - real_shuf) ~ 0.44 of real; monoplex user 9.6 (shuf 6.4) ~0.73; live calls v5 user 10.8 (shuf 9.7) ~0.25;
  livedup v5 user 6.1 (shuf 8.3) ~0; duplex export-498 hidden 4.2 (72 windows). Per group SE ~0.5-1.5 pp (more with window correlation) -> only ALL-level numbers are solid.
  Caveat: session 'shuffled' is above 6.25 (same-call speech from other moments shares slow drift), so normalise against shuffled, not 1/16.
- 23:40 UTC: base2 (patience 4, 5k steps) still training; if its eval beats base, generated is rescored from the cached features (~3 min).
- 00:02 UTC: base2 (longer patience) plateaued at the same level (eval top-1@16 9.6-10.1 %, words-only 10.5, audio-only 8.7, shuffled 6.8, pose +2 s 9.0, tight 9.4) -> base kept (ckpt_base.pt, score2_base.json).
  VERDICT (CPU-mini): real speech<->gesture correspondence is detectable (~10-11 % vs 6.25-7 % chance; words-only above chance in random AND tight groups = some lexical/semantic signal),
  but it is weak and mostly slow (pose shifted 2 s keeps ~90 % of it). Too weak for a per-run score; generated user-made sit at ~0.4-0.7 of real (ALL-level only).
  Next lever = scale (25 side-h used of 2,379; GPU), not more CPU epochs.
