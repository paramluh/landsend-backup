
## 0. "Lack of starting pose" / frame0 (posefix, 10-26 ~21:25 UTC)
- Training: the LLM never sees a frame0 state; rows carry pose tokens only, relative to their window's frame 0. frame0 is used only at
  DECODE, and only as pose.global_state = (root y, root x, shoulder width) (joints55 unused by the decode).
- Live: the avatar decode (rt/seq_engine.py ~203/374) and the genweb skeleton use the package clip frame0
  (prompts/01_ab63d745/llm_inputs frame0_pose -> gs (0.5822, 0.5473, w 0.8026)), but calls prefill from the OPENING window, whose tokens
  are relative to its own frame0 (root (0.5765, 0.5305), w 0.8179). Mismatch = uniform 1.9 % scale + ~3 / 4 px (y / x) shift.
- Verdict: a real (small) bug, NOT the cause of non-physical poses (uniform scale / shift keeps all bone ratios and angles). Fix routed:
  server sends the opening frame0_global_state in the feed start (aloedemo), renderers use it when present (videort, genweb skeleton).

## 0b. Does the LLM need the starting pose / do codes encode diffs? (tokenizer @1d6ff4f0, posefix/score/perturb.py)
- Config: model_type conv (TCN), causal true, downsampling [1, 2] (25 fps -> 12.5 Hz tokens), kernel 3, 32-codebook RVQ (2048 x 16, cosine).
- Each frame = FULL absolute configuration: kinematic offsets joint-from-parent (root from the frame-0 root), x frame-0 shoulder width;
  positions = spatial cumulative sum along the kinematic tree within the frame (pose.kinematic.offsets_to_positions). No temporal deltas.
- Decode = causal TCN: frame t depends on token t + a few earlier tokens; no recurrent state.
- One token corrupted (P01 GT, t = 30/60/90; mean over 3): mean joint displacement / sw peaks 0.12-0.17 at frames 2t+1..2t+2, < 0.01 sw by
  frame 2t+10, exactly 0 from 2t+12 (~0.5 s), for swap-all / random light / random cb0. Errors do NOT accumulate or persist.

## A. Physicality scorer, decode/render check, generated vs real, drift (PHYS-A fork, 10-26 ~21:45 UTC)

**Scorer** `posefix/score/phys.py` (calibration `phys_calib.py` → `out/phys_calib.json`; marker `score/PHYS_READY`). It works per frame on the 55 rendered joints, decoded exactly like the avatar (score.py / LivePose), counting only confident joints (conf > 0.3). All lengths are in units of the actor's frame-0 shoulder width.

Five tests, each scored against its own per-item band from real data:
- `bone_len`: 54 bones, outside the real band p0.5–p99.5.
- `bone_jump`: frame-to-frame change in a bone's length.
- `vel`: per-joint speed (real p99.5).
- `acc`: per-joint acceleration (real p99.5).
- `angle`: finger zig-zag and over-turn, elbow fold with long bones, and wrist bend.

The flag threshold tau for each test is the real frame-score quantile q0.998, so by construction each test flags 0.2 % of real frames.

Self-crossing is not scored. Crossed arms and clasped or crossed fingers are physical, and in 2D they cannot be told apart from impossible crossings.

An `as rendered` variant (`PHYS_CONF=0.2` → `out/phys_calib_c020.json`) counts joints above 0.2, the raster's `conf_min`.

**Real data (pooled 146,704 frames):** any-fail 0.67 %. Held-out Aloe tracks: 0.62 %. Actor GT clips: 1.9 %.

**Decode / render check: CLEARED**
- **Same path as training data.** The render service's own GT path (`render_service.gt_feed`: GT pose tokens → the live engine → `render/out/gt-<pid>_sxs.mp4`) renders Person 01's real tokens through the exact live decode and skeleton raster. Contact sheet: `score/work/sheet_gt01_skel.png`.
  - The torso box is stable and the face line is normal.
  - Hands appear as finger fans, sometimes red when confidence is low.
  - It shows the same visual style as the generated skeletons.
- **Code consistency.**
  - The decode matches LivePose (verified earlier to 7.6e-6).
  - The joint order is `P.SEL` = WHOLEBODY_INDEX, the same order the raster's CONNECTIONS use.
  - The frame0 check (parent) found the package and opening frame0 differ by a uniform 1.9 % scale only.
- **Caveat that matters below.** The raster (`video/code/pose/raster.py RENDER`) draws every joint with conf ≥ 0.2: red at 0.2, ramping to green at 0.6, then fading below 0.2. Low-confidence joints therefore reach the video model as red bones in whatever position they decoded to.

**Generated vs real: share of frames failing (any test), generated frames only**

| Set | Any-fail |
|---|---|
| Real (by construction) | 0.67 % |
| User call 7b5a (502 frames) | 0.0 % |
| User call b62a (1,944 frames) | 0.72 % |
| 10 recent monoplex runs (mean) | 0.08 % |
| 11 recent live sessions (mean) | 0.2 % |

On bone lengths, joint angles, velocity and acceleration, the generated pose is NOT less physical than real decoded pose. The model does not produce impossible kinematics more often than the data does.

**What IS off in the user's long call (b62a): confidence collapse and posture drift**
- **Confidence collapse.**
  - Body-joint confidence is 0.33–0.41 in b62a, against 0.48–0.61 on real tracks.
  - From ~30 s, 60–83 % of all joints fall below the raster floor (hand confidence 0.07–0.16). Joints at conf 0.2–0.3 are drawn red: 11–38 % of joints in b62a, against 2–15 % real.
- **Posture drift, in 10 s bins** (`out/phys_posture.json`).

| Measure | b62a, 0–50 s | b62a, 60–80 s | Real tracks |
|---|---|---|---|
| Torso shear (hip width / shoulder width) | 0.45–0.51 | 0.82–1.03 | 0.54–0.73 |
| Torso lean | 0.5–2° | 4.9 → 17.6° | 0.6–3.5° |
| Head roll | — | 8 → 18° | 1–12° |

- **What it looks like.** The contact sheet `score/work/sheet_b62a_sxs.png` (every 2 s) and the worst frames `score/work/phys_worst_20261026-205651-b62a.png` show it:
  - From ~54 s the skeleton's torso box rotates and shears.
  - Red zig-zag arm lines appear outside the body.
  - The avatar leans back, throws its head back, and the face distorts.
  - The scorer's worst frames sit at 55–77 s (angle / acc / vel).
- **The short call (7b5a)** has no kinematic failures. Its hands go below the raster floor from 10 s (hand confidence 0.15; 60 % of joints below the floor), and fan-shaped hands appear when confidence is low.

**Drift (exposure bias):** yes in b62a. The kinematic fail rate per 10 s goes 1.6 / 0.8 / 0 / 0 / 0 / 0.4 / 0.8 / 2.6 %, rising in the last bin. Confidence falls and posture shears over the call. Monoplex runs and other live sessions show no comparable trend; one session reached 2–3 % in its last bins.

**Takeaway for item 3 (PHYS-B):** the non-physical look comes from low-confidence or drifting body geometry, not from impossible bone lengths or speeds on confident joints. Score sampling settings on:
- body confidence;
- share of joints drawn red or below the floor;
- torso shear and lean drift;
- the scorer's kinematic tests run with `PHYS_CONF=0.2` (as rendered).

Files:
- `score/phys.py`, `phys_calib.py`, `phys_runs.py`, `phys_drift.py`, `phys_posture.py`
- `out/phys_calib*.json`, `phys_real*.json`, `phys_runs.json`, `phys_drift.json`, `phys_posture.json`

## B. Sampling attribution (PHYS-B fork, 10-26 ~22:10 UTC)

**Setup.** Replays of the user's two calls through the CURRENT live engine (export-498, graphs; `posefix/phys/replay_attr.py`):
- Same prompt (Person 01 voice clip, P = 134) and the same 2 s conversation-start prefill (N = 26).
- Channel A = the call's own processed mic samples (`session.npz` 'a', exactly what `write_a` received).
- Every non-pose param is the session's own; 2 seeds per setting (the call's seed and seed + 1000).
- Fidelity: the replay equals the call's tokens bit for bit through the prefill (first 27 code frames), then the sampling paths diverge. The replays are therefore equivalent samples, not copies; the U replays reproduce the call's degradation.

Scoring: generated frames only (`posefix/phys/analyze_attr.py`):
- PHYS-A's kinematic tests;
- body conf (joints 0-12);
- % joints below the raster floor (< 0.2) and drawn red (0.2-0.3);
- `phys_posture.geo` torso shear (hip / shoulder width), lean, shoulder roll and head roll.

Results: `posefix/phys/attr_summary.json`; runs: `posefix/phys/runs/`.

**Long call b62a, full 80 s (3,880 generated frames per setting).** Real references: body conf 0.48-0.61, shear 0.54-0.73, lean ≤ 3.5°, head roll 1-12°, Aloe cb0 repeat 58 %, 10.9 gestures/min.

| setting | kin fail % | body conf | < floor % | red % | shear p50 / p95 | lean p95° | sh roll p95° | head roll p95° | cb0 repeat % | gest/min |
|---|---|---|---|---|---|---|---|---|---|---|
| U = user preset (pose CFG 3.2 cb0+light, k34 / k53, T1) | 0.75 | 0.408 | 6.5 | 19.9 | 0.67 / 0.82 | 5.7 | 8.1 | 25.8 | 64.3 | 14.3 |
| C0 = pose CFG off | 0.34 | 0.465 | 0.7 | 8.1 | 0.56 / 0.58 | 2.7 | 2.5 | 5.7 | 88.3 | 0.8 |
| Cb = CFG on cb0 only | 0.21 | 0.471 | 17.0 | 7.7 | 0.60 / 0.64 | 3.2 | 2.8 | 8.7 | 84.1 | 3.1 |
| Cl = CFG on light only | 1.24 | 0.411 | 6.0 | 20.5 | 0.62 / 0.73 | 4.9 | 7.7 | 17.1 | 69.3 | 6.2 |
| U + pose light temp 0.8 | 1.86 | 0.367 | 14.2 | 11.1 | 0.68 / 0.97 | 11.6 | 26.1 | 26.6 | 61.5 | 7.4 |
| U + pose cb0 temp 0.8 | 0.34 | 0.415 | 1.8 | 17.5 | 0.45 / 0.54 | 6.1 | 8.3 | 24.2 | 74.4 | 9.2 |
| U + pose light top-k 30 | 0.77 | 0.31 | 49.7 | 12.8 | 0.53 / 0.71 | 75.5 | 86.9 | 14.4 | 72.9 | 7.8 |

**Drift per 10 s bin** (body conf / % below floor / shear p50).

| bin | U | C0 |
|---|---|---|
| 0-10 s | 0.46 / 0.3 / 0.565 | 0.462 / 0.15 / 0.57 |
| 10-20 s | 0.44 / 0.8 / 0.555 | 0.465 / 0.15 / 0.56 |
| 20-30 s | 0.41 / 0.6 / 0.545 | 0.461 / 0.15 / 0.565 |
| 30-40 s | 0.40 / 7.9 / 0.76 | 0.461 / 0.4 / 0.555 |
| 40-50 s | 0.40 / 9.9 / 0.75 | 0.466 / 0.4 / 0.555 |
| 50-60 s | 0.40 / 18.1 / 0.715 | 0.462 / 1.65 / 0.545 |
| 60-70 s | 0.38 / 8.9 / 0.725 | 0.469 / 1.7 / 0.545 |
| 70-80 s | 0.39 / 5.3 / 0.735 | 0.474 / 1.5 / 0.55 |

Under U the torso shears past the real band from ~30 s and confidence keeps falling. C0 stays flat.

**Log-prob attribution (short calls, cond-only model, teacher-forced on the sampled codes).** Sampled codes:

| set | cb0 cond log-prob | summed cb1-31 light log-prob |
|---|---|---|
| U (7b5a / b62a) | −1.08 / −0.94 | −89.0 / −87.6 |
| C0 (7b5a / b62a) | −0.32 / −0.29 | −73.9 / −74.1 |
| REAL held-out B track (conv c1, real codes, 248 tokens) | −0.99 | −72.4 |

- Under CFG 3.2 the light codes are ~15 nats per token (~0.5 nat per codebook) LESS likely under the model than real codes. CFG extrapolation on cb1-31 pushes the light codes off the data manifold, and they decode to low-confidence, sheared bodies.
- The CFG-off cb0 is MORE confident than real (−0.3 vs −1.0): the pose freezes (cb0 repeat 88 % vs real 58 %).
- The rare kinematic failures sit on tail tokens: flagged frames have cb0 −3.2 / −2.5 and light −116 / −112, against clean frames at −1.0 / −0.9 and −89 / −87.
- The "any light codebook < 1e-3" share does not discriminate (real 51 %): deep codebooks are high-entropy. The summed light log-prob does.

**Verdict.**
- The non-physical look comes from pose CFG on the light codebooks. Light-only CFG reproduces the user preset's damage: conf 0.41, 20 % red, shear 0.62→0.77, head roll 17-26°. CFG on cb0 only does not; it stays close to real posture.
- It is not a decode/render bug (PHYS-A) and not impossible kinematics.
- Drift is CFG-driven, not inherent exposure bias: C0 is flat over 80 s.
- Truncating or cooling the light distribution makes it worse: light top-k 30 gives 50 % of joints below the floor and lean p95 75°; light temp 0.8 gives shoulder roll p95 26°.
- CFG off by itself is physical but frozen: 0.8 gestures/min against real 10.9.

**Candidate fixes (sampling, so they need the user's OK; none applied):**
- (i) `pose_cfg_light` = 0, i.e. CFG on cb0 only. Real-like posture and conf, but motion is low (84 % repeat, 3 gestures/min) and 17 % of joints sit below the floor (mostly hands).
- (ii) CFG off with a wider light top-k (151, posefix 10-25: restores motion with CFG off). Not re-tested in this grid.

## C. A/B of pose sampling arms (PHYS-B, main GO, 10-26 ~22:26 UTC) — SEED 1 (4 mics x 1 seed; seed 2 follows)

**Setup.** Live engine, export-498, graphs (`posefix/phys/ab_replay.py`, `analyze_ab.py` → `phys/ab_summary_seed1.json`). Each run: Person 01 voice-clip preset + the 2 s opening prefill, 750 frames (60 s), generated frames only.

Mics (channel A):
- 7b5a, b62a: the user's 2 calls, their processed mic;
- m1086: call 1086's mic;
- c2: held-out conv c2's partner stem.

Arms, starting from the user's preset (U):
- U: the user's preset;
- (a): U with `pose_cfg_light` 0 (CFG 3.2 on cb0 only);
- (b): U with `pose_cfg_on` 0 + `pose_light_topk` 151.

REAL row: 24 held-out Aloe B tracks in Person 01's frame, same scorers; mean (p10-p90).

| arm | body conf | % joints < 0.2 | % red (0.2-0.3) | lean p95° | head roll p95° | cb0 repeat % | gestures / min | hands moving % |
|---|---|---|---|---|---|---|---|---|
| U = user preset (CFG 3.2 cb0+light, k34 / k53) | 0.415 | 36.7 | 17.6 | 6.0 | 23.7 | 65.8 | 11.5 | 21.4 |
| (a) CFG 3.2 on cb0 only | 0.453 | 5.6 | 21.4 | 3.1 | 9.5 | 74.2 | 4.4 | 5.9 |
| (b) CFG off + light top-k 151 | 0.476 | 2.5 | 11.2 | 2.6 | 9.4 | 70.4 | 4.9 | 5.7 |
| REAL Aloe | 0.502 (0.44-0.56) | 19.6 (0.6-52) | 7.9 (3.8-14.6) | 11.0 (2.8-6.8) | 9.9 (6.3-12.8) | 58.3 | 10.9 (6.6-15.1) | 21.4 (10-35) |

The REAL lean mean is inflated by a few tracks; their median p95 lies in the 2.8-6.8 band.

**Drift per 10 s bin** (body conf / % red / head roll p95).

| bin | U | (a) | (b) |
|---|---|---|---|
| 0-10 s | 0.449 / 13.1 / 15.4 | 0.506 / 12.4 / 7.2 | 0.486 / 16.0 / 8.7 |
| 10-20 s | 0.425 / 15.0 / 20.9 | 0.504 / 14.1 / 10.2 | 0.489 / 15.1 / 7.4 |
| 20-30 s | 0.406 / 19.2 / 25.9 | 0.451 / 19.1 / 9.3 | 0.474 / 9.5 / 8.4 |
| 30-40 s | 0.407 / 20.6 / 17.3 | 0.411 / 25.2 / 8.4 | 0.473 / 11.3 / 7.6 |
| 40-50 s | 0.398 / 17.1 / 16.3 | 0.421 / 29.2 / 7.1 | 0.471 / 8.4 / 9.5 |
| 50-60 s | 0.398 / 21.5 / 15.6 | 0.417 / 30.7 / 7.8 | 0.457 / 5.9 / 8.2 |

**Seed-1 read:**
- **(b) is the most physical and the most stable.** Its body conf, lean and head roll sit at real levels and stay flat over 60 s.
- **(a) fixes the posture but still drifts.** Lean and head roll are real-like, but confidence decays 0.51 → 0.42 and red joints rise 12 → 31 %.
- **U moves like real data but is non-physical.** Its motion matches real (11.5 gestures/min, 21 % hands moving), but head roll p95 is 24° against real 10°, 37 % of joints fall below the floor, and confidence drops 0.45 → 0.40.
- **Both fixes under-move.** (a) and (b) give ~4-5 gestures/min and ~6 % hands moving, against real 10.9 and 21 %; cb0 repeat is 70-74 % against 58 %.

### C (full). A/B, 4 mics x 2 seeds = 8 runs per arm, 60 s each (`phys/ab_summary.json`). Means; REAL = 24 held-out Aloe B tracks, mean (p10-p90).

| arm | body conf | % joints < 0.2 | % red (0.2-0.3) | lean p95° | head roll p95° | cb0 repeat % | gestures / min | hands moving % |
|---|---|---|---|---|---|---|---|---|
| U = user preset (CFG 3.2 cb0+light, k34 / k53) | 0.421 | 24.7 | 17.5 | 5.4 | 19.0 | 69.8 | 9.2 | 15.2 |
| (a) CFG 3.2 on cb0 only | 0.453 | 3.8 | 14.1 | 2.8 | 9.1 | 78.0 | 5.2 | 7.7 |
| (b) CFG off + light top-k 151 | 0.469 | 1.8 | 8.8 | 2.6 | 9.5 | 71.5 | 5.3 | 6.6 |
| REAL Aloe | 0.502 (0.44-0.56) | 19.6 (0.6-52) | 7.9 (3.8-14.6) | 11.0 (2.8-6.8) | 9.9 (6.3-12.8) | 58.3 | 10.9 (6.6-15.1) | 21.4 (10-35) |

**Drift per 10 s bin** (body conf / % red / head roll p95).

| bin | U | (a) | (b) |
|---|---|---|---|
| 0-10 s | 0.452 / 14.1 / 14.9 | 0.489 / 11.7 / 7.6 | 0.480 / 11.1 / 8.4 |
| 10-20 s | 0.429 / 18.1 / 16.2 | 0.479 / 11.4 / 8.5 | 0.477 / 10.4 / 7.8 |
| 20-30 s | 0.416 / 20.5 / 19.2 | 0.452 / 12.9 / 8.4 | 0.466 / 7.8 / 8.0 |
| 30-40 s | 0.410 / 19.6 / 15.4 | 0.427 / 15.7 / 7.7 | 0.467 / 8.2 / 8.3 |
| 40-50 s | 0.408 / 14.8 / 14.6 | 0.434 / 16.4 / 7.6 | 0.464 / 7.0 / 9.2 |
| 50-60 s | 0.404 / 17.9 / 14.3 | 0.430 / 17.3 / 7.8 | 0.455 / 8.6 / 7.7 |

**Read (2 seeds, same as seed 1):**
- **(b) CFG off + light k151 is the most physical and the most stable.**
  - Red joints 8.8 % against real 7.9; body conf 0.47 against 0.50.
  - Head roll 9.5° (real 9.9) and lean 2.6°.
  - Flat over 60 s.
- **(a) fixes the posture** (head roll 9.1°, lean 2.8°), but its confidence decays 0.49 → 0.43 and red joints climb 12 → 17 %.
- **U is the only arm near real motion** (9.2 gestures/min, 15 % hands moving), but it is non-physical:
  - head roll p95 19° (real 10);
  - 25 % of joints below the floor, 17.5 % red;
  - confidence decaying 0.45 → 0.40.
- **Both fixes under-move.** (a) and (b) give ~5 gestures/min and 7-8 % hands moving, against real 10.9 and 21 %; cb0 repeat is 72-78 % against 58 %.
- **Choice for the user:** (b) gives physical pose with about half the real motion. Nothing applied; any change needs the user's OK.

### C (full, 2 seeds x 4 mics = 8 runs / arm, 60 s each; posefix/phys/ab_summary.json)
| arm | body conf | below floor % | red % | lean p95 | head roll p95 | cb0 repeat % | gestures/min | hands moving % |
|---|---|---|---|---|---|---|---|---|
| U user preset (CFG 3.2 cb0+light, k34/k53) | 0.421 (0.452 -> 0.404 over 60 s) | 24.7 (13 -> 36) | 17.5 | 5.4 | 19.0 | 69.8 | 9.2 | 15.2 |
| (a) CFG 3.2 cb0 only | 0.453 (0.489 -> 0.430) | 3.8 | 14.1 (11.7 -> 17.3) | 2.8 | 9.1 | 78.0 | 5.2 | 7.7 |
| (b) CFG off + light k151 | 0.469 (flat 0.48 -> 0.455) | 1.8 | 8.8 (flat) | 2.6 | 9.5 | 71.5 | 5.3 | 6.6 |
| REAL Aloe held-out (24 tracks) | 0.502 | 19.6 (p10-p90 0.6-51.7) | 7.9 | 11.0 | 9.9 | 58.3 | 10.9 | 21.4 |
