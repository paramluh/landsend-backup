# posedrift: does pose CFG 3.2 explain the low joint confidence and posture drift? (10-26, CPU only, existing data)

## Answer
- **Partly. CFG explains the loss of BODY confidence and the head-roll drift.** It does not explain the hands falling below the raster floor or the torso lean.
- **Body confidence.** With CFG on, body confidence starts about 10 % lower and falls about 3x faster over the call. Head roll also climbs about 3x faster.
- **Hands.** The hands-below-floor share (b62a: 52 % of joints ≤ 0.2) is driven by the light top-k, plus call-specific "hand collapse" episodes. The CFG effect on it is not significant.
- **Torso lean.** Lean above 10° is equally rare with CFG on, with CFG off, and in real data. b62a's 17.6° is a tail event.
- **This is observational.** posefix's GPU replays (same inputs, CFG on vs off) are the causal check.

## Data
| | |
|---|---|
| Duplex export-498 streams ≥ 20 s | 115: CFG on 39 / off 76 |
| Live (browser + playground) | 53 |
| Scripted test / offline duo | 62 |
| Monoplex-357 runs ≥ 20 s | 115: on 60 / off 55 |
| Real reference | 24 held-out Aloe tracks (10 min each) + 20 actor GT clips |

- **CFG labels come from each call's recorded params**, not from the clock. CFG-on calls exist on 10-25 too, so the "before 06:50 PT = off" cut is not exact.
- **Scorer.** posefix decode (score.decode + to_keypoints, person frame 0 = avatar), generated frames only.
  - Metrics: confidence over the 55 rendered joints; body confidence (joints 0-12); % of joints ≤ 0.2; % red (0.2-0.3).
  - Posture uses phys_posture.geo (lean, shear, roll) plus score.head_angles (pitch), and phys.py kinematic fails.
- **Check.** The scorer reproduces posefix's b62a bins exactly: body confidence 0.41 → 0.37, lean 4.9 → 17.6°, shear 0.82 → 1.03.

## 1. CFG on vs off (duplex, live calls; bootstrap 95 % CI over calls)
| metric | CFG on | CFG off | real Aloe | effect |
|---|---|---|---|---|
| body conf, first 20 s | 0.451 | 0.502 | 0.511 | −0.051 [−0.070, −0.032], d −1.35 |
| body conf slope / min (calls ≥ 60 s) | −0.055 | −0.019 | −0.001 | late − early diff −0.035 [−0.055, −0.013], d −1.28 |
| head roll, first 20 s | 8.6° | 6.0° | 3.4° | +2.6° [0.1, 5.0] |
| head roll slope / min | +4.6° | +1.7° | +0.03° | late − early diff +4.8° [1.6, 8.1], d 0.97 |
| 10 s bins with body conf < 0.4 | 44.5 % | 4.4 % | 5.5 % | |
| 10 s bins with head roll > 15° | 27.2 % | 4.9 % | 0.7 % | |
| 10 s bins with shear > 0.85 | 12.5 % | 1.1 % | 1.1 % | |
| 10 s bins with lean > 10° | 2.9 % | 1.8 % | 2.7 % | n.s. |
| lean slope / min | +1.4° | +1.1° | −0.8° | n.s. |
| % joints ≤ 0.2 (first 20 s) | | | 22 % | −2.6 pts, n.s. |
| kinematic fail (phys any) | 0.7 % | 0.6 % | 0.67 % | n.s. |

**Time curves (live duplex, mean per 20 s bin):**

| time | body conf, CFG on | body conf, CFG off | body conf, real | head roll, CFG on | head roll, CFG off | head roll, real |
|---|---|---|---|---|---|---|
| 0–20 s | 0.451 | 0.502 | 0.511 | 8.6° | 6.0° | 3.4° |
| 60–80 s | 0.402 | 0.478 | 0.501 | 15.0° | 7.3° | 3.6° |
| 100–120 s | 0.378 | 0.460 | 0.503 | 13.8° | 9.1° | 4.2° |

**Per-bin regression (all duplex streams, cluster SE by call).** The model is y ~ cfg + t + cfg×t + log2 cb0 top-k + log2 light top-k + live.

| metric | CFG | t (per min) | CFG × t | light top-k (per doubling) |
|---|---|---|---|---|
| body conf | −0.043 ± 0.014 | −0.004 ± 0.006 | −0.009 ± 0.007 | |
| head roll | +4.3 ± 1.2° | +1.25 ± 0.64 | +0.73 ± 0.69 | |
| shear | −0.072 ± 0.028 | | +0.055 ± 0.013 per min (the shear tail grows with CFG) | |
| % joints ≤ 0.2 | −0.05 ± 0.07 (n.s.) | | | −0.148 ± 0.028 |
| lean | +1.5 ± 0.6° | +0.75 ± 0.30° | n.s. | |

## 2. What CFG does NOT explain
- **Hands / % joints ≤ 0.2.** Light top-k sets the level:

  | light top-k | % joints ≤ 0.2 (mean) |
  |---|---|
  | 30 | 55–63 % |
  | 53 (user preset) | 13 % |
  | 151 | 10 % |

  - At the user preset (34/53 + CFG 3.2), 4 of 9 calls (b62a, 7b5a, b991, 5661) lose their hands within 20–40 s: hand confidence falls from 0.5 to 0.1–0.3. The other 5 calls keep it at 0.4–0.6 for up to 2 min (9c3f, 9d34).
  - So this is an intermittent hand-collapse mode, not a steady CFG effect.
  - Real Aloe has 22 % of joints ≤ 0.2 too, because resting hands go out of frame.
- **Torso lean.** Lean drifts slightly in every generated group: +1.1 to +1.4°/min, n.s. between CFG on and off. Lean > 10° is as rare as in real data.
  - Among long calls, max lean is 17.6° with CFG on (b62a, 1 of 17 calls) and 12.8° with CFG off (out of 38 calls).
- **Head pitch proxy (duplex).** No CFG effect (+0.016 ± 0.035).

## 3. Other factors
- **Exposure drift exists even with CFG off.** Over 2 minutes, live CFG-off body confidence goes 0.50 → 0.46 and head roll 6 → 9°, while real data stays flat. CFG makes the decline about 3x faster.
- **Monoplex-357, as a second model.**
  - CFG on and off are identical for the first 30 s.
  - In the few 2-minute generations (about 9 with CFG on vs 4 with CFG off), CFG-on body confidence falls 0.48 → 0.40 by 100–120 s, and head pitch goes −0.41 → −1.15 (the head tips back).
  - CFG-off generations stay at 0.48–0.54.
  - This points the same way.
- **cb0 top-k.** It correlates with head roll (−4.3° per doubling) but is collinear with CFG here (range 30–67), so this is weak evidence.
- **Not testable from this data.**
  - Pose rep-pen: it is 1.0 in every recorded call.
  - Model / checkpoint: every duplex call ran on export-498.
- **Client.** Live vs scripted has no effect once the other factors are controlled.

## Files
| File | Contents |
|---|---|
| `score_calls.py` | inventory, per-call scoring, real reference |
| `analyze.py`, `drill*.py` | grouping, regression, curves |
| `out/inventory.json` | all streams with their recorded params |
| `out/calls.*.jsonl` | per-call metrics and 10 s bins |
| `out/real.jsonl` | real reference metrics |
| `out/analysis.json` | grouped analysis output |

Re-run: `go.sh run SHARD 2 8` (srun --overlap, nice 19), about 40 s for all 393 streams.
