# posefix STATUS (tmux agent posefix; task from main amartyavarma-10, 10-25 ~07:35 PT). Prompt: PROMPT.md here.

## Facts gathered (10-25 ~07:50 PT)
- Training pose layout: B pose of row frame f at P+CD+f; first chunks start at frame 0; tokens normalised to the conversation's frame 0
  (root offset 0, shoulder width 1 at frame 0). Tokenizer noshift@1d6ff4f0 = videort's pose.TOKENIZER (same).
- Offline fastinfer samples (10 s real prefill incl. pose): generated cb0 92.5 % repeat / 22.8 distinct vs real 65.1 % / 103 -> the model
  is too static even with prefill.
- pretest (export-333, held-out conv c1): TF pose cb0 CE 1.08 (eval 1.07), light 2.29 (eval 2.94: gap to explain).
- User call 1086 (CFG off) had pose_light_topk 151, light_topk 150 (page values, not the 30 standing rule).
- videort align (rt/live.py LivePose._align, not deployed): 1086 out-of-frame 6.7 % -> 0 %, centre offset 0.17 -> 0.011 sw.

## Plan
1. simulator (sim/): LivePPM offline, replay session 'a' with session params + seed; TF CE on held-out rows.
2. scorer (score/): pose tokens -> keypoints -> metrics; GT baselines (20 actors, held-out Aloe B pose); sessions.
3. levers: CFG off, cb0 stay penalty / temp / top-k, pose prefill (actor gt tokens), align + orient, smoothing.

## 10-25 ~07:50-08:00 PT
- POSE PREFILL built (staged, not live): stage-next/live_ppm_pose.py (start(..., pose_prefill), forced after the pose graph like the parity
  path) + stage-next/server.py (start param pose_prefill default 1; N = 2 x round(prefill_s / 0.16) + 2 actor tokens = videort's real-video
  window; feed/session report pose_prefill_frames). prefill_s default unchanged (0) until numbers decide.
- Simulator sim/replay.py (+ sim.sbatch / sim3.sbatch, engine symlinks sim/eng): replay of 1086 is statistically equal to live (cb0 change
  0.36 vs 0.34) but not token-identical (diverges at pose frame 4: cross-process Triton autotune, as aloedemo documented).
- Codec: actor gt/pose_tokens = same tokenizer (noshift), different keypoint pipeline (studio posev3 conf-hold vs Aloe RTMW). Model CE on the
  forced actor tokens 2.7 / 2.4 / 2.3 (prefill 2 / 5 / 11.68 s) vs 1.07 held-out TF: in-domain control running.
- Scorer (score/score.py, baselines score/out/SUMMARY.md). 1086 replay, raw (no filters): no prefill centre 0.32 sw / width 0.75 / out of frame
  70 % / conf 0.42; prefill 2 s 0.04 / 1.02 / 1.5 % / 0.54; 5 s 0.09 / 0.99 / 5.7 % / 0.42; 11.68 s 0.11 / 1.06 / 2.8 % / 0.53. n = 1 each:
  16 held-out convs x {0, 2, 5, 11.68 s} + 3 seeds running.

## 10-25 ~08:00-08:10 PT: scope change (user via main)
- DROP align / orient / smoothing (user: "dont do position lock or head fixing"). KEEP pose prefill. MAIN TASK: duplex pose sampling as good
  as the monoplex (aloe-pose-mono-8020 exports/export-357). User add-ons: monoplex generations from the exact same prefill (0/2/5/10/11.68 s),
  rendered through videort's engine (no post-processing), ONE samples page in the existing pose page format.
- Audit (posefix/audit/): no inference bug; live pose == trainer maths == fast_ppm == monoplex engine. Both models fall into frozen cb0
  states (10 s windows at 1.00 repeat) under plain sampling; monoplex 60 s studio-nocfg clips freeze too (0.80-0.94); monoplex CFG 3 livelier.
  Side bugs: dropped feed frames -> pose code 0 glitch (videort fixed + rolled); single-GPU side-stream race (not our deployment).
- Matched runs (sim/runs_dupM*, runs_mono*; out/dupM, out/mono): fantastic-set text/audio, pose T1 k30/k30, CFG off. Renders: videort's
  render service inside job 30427 -> clips/{mono,dupM,gtc,gt}. Page builder agent -> posefix/page.
- POSE REP ported to stage-next engine (pose_rep / pose_rep_light / pose_rep_win; 1 = off, exact). Sweep (out/sweep, job 30436): pf 1.92 s,
  arms lk151 / lk0 / k0lk0 / k0 / rep13 / t12 / rep11 / rep12all on 1086 + c0-c2. Hint: 1086 with the user's session params (pose light
  top-k 151) moves like real data; the same with light top-k 30 freezes (cb0 change 0.06).

## 10-25 ~08:10 PT: max-parallel re-plan (user furious at pace)
- 30 s clips (prefill + 30 s). Shared run lists with atomic claims (sim/runs_S_{dup,mono}_all.json; replay.py / mono.py claim
  <out>/<name>.claim). Jobs: 30437 (s18 4-7: dup_a + mono_a), 30444 (s4 0-5: dup + mono + 2 render instances), 30446 (s18 0-3:
  dup + mono, reversed order). videort renders on s4 6-7 (job 30439) and slinky-10 GPUs. Renders -> clips/{mono,dupM,gtc,gt}.
- PAGE: https://claude.ai/artifact/HiVtukuNruXFJF6pyMSGEr (builder posefix/page/build_page.py; republish deltas via page/delta.py).
- Early headline: cb0 repeat on generated frames, 2 s prefill: monoplex 88 % vs duplex 85 %; 11.68 s: 92 % vs 71 % (real Aloe 58 %,
  Person 01's own 34 %). The monoplex freezes as much under the same plain sampling; the user's good monoplex videos (10-19 eval v3)
  used pose CFG 3 + a full 2 s prefill (pose + audio + words): cfg3 arms added for both models.

## 10-25 ~08:15-08:20 PT: CAUSE + FIX
- Playground (monoplex web UI) moved to aloedemo end to end; slinky-18 handed over (30437/30446 cancelled).
- CAUSE: (1) pose light top-k 30 freezes the pose in BOTH models (cb0 repeat 83-93 %, real 58 %); light top-k 151 (user's own call
  setting) / off restores motion (sweep: out/scores/sweep1.json, summ.py). (2) no pose prefill -> drift / scale / off-frame.
- FIX in stage-next (handed to aloedemo for the idle-only roll): pose prefill default on with prefill_s default 2 s (1.92), pose
  light top-k default 151, pose rep knobs (off). Page republished v3 (25 clips).

## 10-25 ~08:30-08:50 PT
- Live demo: stage-next rolled 15:24 UTC (pose prefill 2 s default, pose light k151 default, pose rep knobs off). Bug: /api/presets 500
  (POSE_LABELS lacked pose_rep keys) -> fixed in server.py + stage-next, idle backend restarted 15:28 UTC; aloedemo's roll now checks presets.
- Comparison page: main replaced index.html with a plain page (direct video src); I republished a/*.mp4 with B voice muxed (page/mux.py).
- FULL PREFILL (pose + words + audio) per user: runners replay.py / mono.py take full_prefill {bundle, s}; bundles sim/prefill/<person>.npz
  (make_prefill.py). Person 10: 48 runs done (out/fullD, out/fullM). Person 01 word timings: sim/align.py (monoplex causal constrained
  lane decoding, thr 0.2; counts 5 < 1.04 s ok, 11 vs 10 < 2 s) -> genweb/words/01_ab63d745; 24 monoplex runs queued on the playground
  (genweb/runs/posefix-p01_*); 24 duplex runs need a GPU pair (runs_F01_dup.json). Media builder: page/full_page.py -> full_manifest.json.

## 10-25 ~09:00-09:10 PT
- Page v13 (index_plain_full.html from main's build_plain_main.py over data.json = data_base.json + full-prefill groups via
  page/full_page.py): Person 10 full prefill 48 clips + GT, Person 01 duplex 24. Removed mp3s / weak arms / pose-only 5-11.68 s groups (cap).
- Person 01 duplex full prefill done (job 30487). Monoplex P01 on the playground (genweb/runs/posefix[b]-p01_*; import: sim/import_genweb.py).
- CONVERSATION OPENINGS (main: prefill = conversation opening 0-N s): runners take conv_prefill {s} (the conv's own system block, B's real
  lane / Mimi / pose, duplex also A's real audio from t = 0). Jobs 30497 (duplex, openD) + 30498 (monoplex, openM), 32 runs each.
- LIVE FULL PREFILL staged in stage-full/ (engine start(prefill={lane, codes}); server prefill_inputs + start params prefill_audio /
  prefill_words / prefill_src); offline checks pass (schema, lane == monoplex builder). GPU test queued (30507, after 30497).
- 16:15 UTC stage-full GPU test PASSED (job 30509, out/stagefull): forced lane / codes / pose exact, latency 63.9 ms p50; handed to
  aloedemo for the idle-only roll (defaults: prefill 2 s, src auto = conversation start, all streams on, light k151; video prefill kept for
  SUMMARY video=true actors). Duplex playground: aloedemo's helper on replay.py (conv paths sent).
- 16:19 UTC corrected stage-full PASSED (job 30513, out/stagefull2): A silent during the prefill (engine write_a mute), auto source =
  conversation start only where SUMMARY video=true (Person 01 -> her clip, Person 10 -> start). Handed to aloedemo for the roll.
- 16:22 UTC corrected full prefill LIVE (aloedemo roll); public-link check session 20261025-162619-9661: Person 01 -> avatar clip,
  26 frames pose + words + audio, video prefill 1.92 s, light k151.
- 16:35 UTC page v16: conversation openings (64 clips, both models) first, then Person 10 / Person 01 full prefill (k151). Openings
  headline (generated 30 s): k151 mono 65-85 % / dup 71-83 % cb0 repeat; k30 84-89 %; oof 27-47 % (app-framed speakers in the
  studio frame; GT 21 %). Page builder: page/full_page.py (FULL_ONLY=1 ACTOR_LK151_ONLY=1) + build_plain_main.py; publish deltas
  computed vs refs_now.json.
- BEFORE/AFTER on real public-link sessions (score.py, no filters, Person 01 frame): call 1086 (no prefill) vs live 9661 (2 s full
  prefill, k151): head roll -23.8 -> -1.4 deg, centre offset 0.174 -> 0.036 sw, width 0.90 -> 0.99, out of frame 6.7 -> 2.1 %,
  conf 0.34 -> 0.48, low-conf joints 34 -> 7 %, cb0 change 0.34 -> 0.37 (real 0.42). Side-by-side: clips/before_after_sxs.mp4.
- 16:47 UTC FINAL report sent to main (before/after numbers, clips/before_after_sxs.mp4, what changed, prefill seconds per setting).

## 10-26 (UTC ~03:00-03:45)
- Maths-departure audit sent to main (light k151 default = not user-ordered; Person 01 timings model-aligned; ring bug).
- RING FIX (main): forced prefill frames now leave the freq / rep-pen ring holding exactly the fed tokens (stage-ring/live_ppm_pose.py:
  save ring state before the text graph, restore, push the forced token). GPU test 30550 bit-exact vs a reference ring on 3 prefills.
  Rolled by aloedemo 03:28 UTC (both instances restarted 03:28 / 03:35); public-link check session 20261026-033727-21b6 OK.
- Contract-maths re-renders (videort, clips/<kind>_contract): page/full_page.py CONTRACT=1 -> a/c-* media + data_contract.json;
  waiting for the 117 page renders (page/contract_need.json).
- 04:30 UTC page set rebuilt on videort's SEQUENTIAL-sampler renders (clips/*_seq; user's choice, replaces _contract):
  page/data_seq.json + a/s-*.mp4 (129 files, 244 MB; full_page.py RENDER_SET=seq FULL_ONLY=1 ACTOR_LK151_ONLY=1). Path sent to main
  (amartyavarma-8a) for the republish with build_plain_main.py.
- 05:55 UTC data_seq2.json + a/s2-*.mp4 (corrected-reference sequential renders, clips/*_seq2, 117 page clips, 244 MB) READY; path
  sent to main (low priority, republish only if the user asks). data_seq.json is obsolete (wrong refs).

## 10-26 ~07:08 UTC — EXACT LIGHT DECODER (main c7 order: match the trainer's fp32-master + bf16-autocast decoder)
- Finding: every PPM engine (live pose-duplex dec_s/dec_p incl. pose-CFG 2-row, playground monoplex fast_ppm, duplex worker, offline fastinfer + posefix sims) ran the 0.6B decoders in pure bf16 (bf16 residual / norms / slot 0 / embeddings). Trainer: fp32 residual, fp32 RMSNorm out rounded once at each Linear, q/k-norm + RoPE fp32, P->bf16 for P.V, fp32 embeddings, slot0 = proj(hv).float() + zb.float(). Masters are not on disk (bf16 exports) -> fp32(bf16 export) is the reference.
- Fix: stage-exact/light_exact_ppm.py (motlive's kernels + S-row attention + device guards) and FusedDec/FusedDecS replaced in stage-exact/live_ppm_pose.py; stage-exact/mono/fast_ppm.py the same class. Sampling code line-for-line unchanged (same RNG calls).
- Validation (posefix/sim/test_light.py, job 30604, convs c5-c7, 654 frames, vs trainer eval path on fp32 copy under autocast):
  speech NEW top1 0.9933 TV 0.0038 mean|d| 0.0106 max 1.17 | OLD 0.9820 / 0.0099 / 0.0390 / 1.36 | CE ref 2.11906 new 2.11903 old 2.11907
  pose   NEW top1 0.9822 TV 0.0102 mean|d| 0.0196 max 0.50 | OLD 0.9603 / 0.0234 / 0.0918 / 3.00 | CE ref 3.36318 new 3.36296 old 3.36426
  pose-CFG 2-row vs 1-row: max|d| 0.5 (kernel-level; CFG off by default)
  monoplex (export-357, 240 fr) speech NEW 0.9952 / 0.0033 / max 0.19 vs OLD 0.9876 / 0.0086 / 0.56; graph==eager bit-exact (lane, codes, pose)
- Bug found by the harness: Triton launches use the CURRENT device -> off-device calls raced (NaN). Engines always run decoders under torch.cuda.device(dev), so serving unaffected; guards added anyway.
- GO to aloedemo 07:08 UTC (live both instances + duplex worker). Monoplex: rerun job 30605 then GO.
- 07:09 duplex graph==eager bit-exact (150 fr + repeat); ms p50 old 74.5 / new 74.3 (slinky-4 6-7, same pair) -> no slowdown.
- 07:15 monoplex rerun (job 30605, device-fixed, 320 fr): speech NEW 0.9954 / TV 0.0030 / max 0.19 vs OLD 0.9889 / 0.0077 / 0.63; pose NEW 0.9849 / 0.0087 / 0.25 vs OLD 0.9656 / 0.0203 / 1.5; CE ref 3.41632 new 3.41656 old 3.41816. GO monoplex to aloedemo.
- 07:21 LIVE: live demo inst 1 07:09:58 (p50 63.9 ms), inst 2 07:12:31 (63.5), duplex worker 07:14:17, monoplex worker 07:20:50 (73.9 vs 74.1 ms). Changelogs in. Main told.
- 08:00 COMPILE (fork): live 64.1->63.4 ms (-1.2%), mono 60.8->60.3 (-0.9%); NOT bit-identical (top-1 99.0-99.8%, TV<=0.008, CE +-0.002); staged stage-compile/, ON HOLD at aloedemo for the user's yes/no (maths rule). Main c7 unreachable; asked the user directly. Details posefix/compile/RESULT.md.
- 10:55 POSE REP-PEN SPLIT (user): live/duplex already split (pose_rep cb0 / pose_rep_light); monoplex fast_ppm -> pose_rep_cb0 / pose_rep_light (+ _speaking/_silent), old pose_rep* alias -> both (old key wins). Proof job 30615: byte-identical old vs new in all 4 cases (defaults, 1.3 win0, 1.5 win64, adaptive 1.4/1.2) via old + new keys. Staged stage-reppen/mono/fast_ppm.py (4e6e0797); GO to aloedemo; main told.
- 12:20 ADAPTIVE POSE REP (user): live engine + dup worker get pose_rep_{cb0,light}_{speaking,silent} (adaptive_rep_on / speak_window_frames, = fast_ppm). Proof job 30617: byte-identical old vs new in 4 cases (defaults; static 1.3/1.2 adapt off; + new knobs 2.0 ignored; adapt on with new = static). Staged stage-adapt/ (engine 9008a073, server d1b0dfde, worker_dup f21b58ef); GO to aloedemo.
- 12:22 LIVE: adaptive pose rep on live inst 1 12:17:52 (63.8 ms), inst 2 12:20:19, dup worker 12:21:50; changelog in; main (-33) told.
- 12:54 HANDS (user: track gestures / hand motion): posefix/score/hands.py + run_stats fields (gestures_per_min [+L/R], hands_moving_pct, hand_speed_p50/p95). Moving = smoothed wrist speed > 0.015 sw/frame; gesture = burst >= 0.3 s, gaps < 0.3 s merged. Real refs: Aloe 10.9/min (6.6-15.1), 21.4 % moving; actors 19.1/min, 41.7 %; P01 GT 30.1/min, 64.5 %. Eye check (hands_sheet.py) 5 vs ~5, 6 vs ~5-6. Handed to aloedemo; main told.
- 12:56 LIVE: gestures/min + hands moving % on the playground (v2026.10.26-24), backfill running newest-first.
- 13:17 POSE CFG SPEAKING-ONLY (user): pose_cfg_speaking_only gates the cb0+light mix per frame on the SPEAKING flag (uncond rows still every frame); live engine + server + worker_dup + mono fast_ppm + worker (summary cfg_guided_frames / cfg_frames_total). Proof job 30659: off -> byte-identical (defaults + cfg on) both engines; on -> guided == speaking frames (live 105/148, mono 135/150). GO both to aloedemo.
- 13:20 LIVE: pose_cfg_speaking_only live x2 (13:18/13:20, 63.8/63.7 ms), dup worker 13:16:58, mono 13:18:51/13:19:52; check run 76d4 146/176 guided. Main told.
- 13:32 PCG REPORTING (aloedemo/dupui): engine last_pcg (counter diff read after pkp sync) + server ws 'pcg', session.npz pcg, session.json + stats cfg_guided_frames / cfg_frames_total. Proof job 30665 byte-identical, sum pcg == counter (0 / 148 / 131). Staged stage-pcg (engine 74fd26f2, server 218047a9); GO.
- 13:42 PER-ACTOR REFS (user: own actor's real rate): posefix/score/out/actor_refs.json (actor_refs.py; 20 actors, run_stats definitions, own frame 0); run_stats returns ref_person. Sent to aloedemo.
- 22:06 PHYS VERDICT: non-physical look = pose CFG 3.2 on light codebooks (conf collapse, red joints, torso shear from ~30 s); decoder/render cleared; frame0 tiny; CFG off / cb0-only physical but low motion; light codes -88 vs -74 (off) / -72 (real) nats. Candidates for user OK: CFG cb0-only, or CFG off + light k151. SAM-sep: 0/12 openings covered (studio vs handover); staged stage-samsep switch (default real); channel A zeros in prefill vs partner raw audio in training. 30749 released.
- 00:42 NAME (Lauren): first readout = model flat at name (Lauren 2.6 %, Sarah 2.3 %, Ella 2.2 % ...); likely cause = absolute min-p 0.05 in factorised word stage keeps only argmax when all names <5 % (+ freq pen) -> deterministic Lauren. Verifying with the engine's post-penalty draw distribution.
- 00:44 NAME CONFIRMED: absolute min-p 0.05 -> one-hot Lauren at 'My name is' (P(word)=0.894 -> 89 %), 20/20; relative min-p 0.05 -> 15 distinct names in 20; min-p off -> 13; freq-pen irrelevant. Trainer vs server prompt identical (no construction bug); Kate row had no clip; 13 % micros carry a B clip. Fix = min_p_rel 1 / min-p off (user OK).
- 01:03 NAME-B: Lauren = common female self-intro in duplex data (4th 'I'm X': 103 sides / 13 speakers); Kate ~only Kate's own (26 'I'm' sides, 21 hers); Lauren speakers' WeSpeaker cosine to Kate 0.17..-0.01 (no closer than random); no voice link. Consistent with the sampler cause (NAME.md B).
