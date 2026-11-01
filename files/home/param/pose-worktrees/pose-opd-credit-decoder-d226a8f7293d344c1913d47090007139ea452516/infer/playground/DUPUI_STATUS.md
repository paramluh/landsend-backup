# DUPUI status (tmux agent dupui = Claude amartyavarma-1b; brief DUPUI_PROMPT.md)

Goal: the playground's DUPLEX model (export-498) gets every monoplex UI feature, and duplex runs generate BOTH sides (user rule conversation-gens-duo).

## Agreed (10-26 UTC)
- aloedemo ~05:10: dupui owns worker_dup.py, a page block `<!-- BEGIN dupui -->` + optional dupui_api.py with a one-line server hook; small duplex-only fixes in aloedemo's blocks allowed (assert count==1, backup, err_check); larger / monoplex-touching changes go to aloedemo as patches (genweb/dupui_patches/).
- GPUs: aloedemo cancelled the old duplex worker 30533; dupui's job **30562** `genweb-dup` (slinky-18, 2 GPUs, 110G, dupui.sbatch) holds the pair. worker_dup.py runs in its loop only while `DUP_ENABLED` AND `DUPUI_LIVE` exist.
- main ~05:15: duo by default. Prefill: both states hear each other's real opening ALIGNED (as in training); the 243 ms line delay applies only from the first generated frame. gen_pd's delayed-everywhere behaviour = labelled option, off.

## Duo maths (worker_dup.py `run_duo`; nothing else changed)
- Two `LivePPM` states (the live demo engine, unchanged) on ONE weight copy, sharing the Voxtral tower + Mimi (= gen_pd.py's two FastPPM on one copy). State 0 = side K (B, the avatar), state 1 = side 1-K (A). Each is set up exactly like the replay path's B for its own side: its own system block / voice clip; real opening forced: lane frames f < N+CD, Mimi codes + pose code frames g < N, N = 2·round(prefill_s/0.16)+2 (1.92 s → N = 26).
- Channel A of state s = what the other state says:
  - **aligned (default)**: A samples t < N·1920 = the partner's real opening audio (raw stem), ALIGNED. The engine's live-demo prefill A-mute is turned off (a_mute = 0) for duo. The partner's generated code frame g ≥ N arrives at A sample g·1920 + 5832 (DELAY = 3·1920 + 72 = 243 ms, gen_pd's minimum causal delay). **The delay starts at A sample N·1920 + 5832 = the partner's first generated code frame N; the first generated step is f = N + CD = N + 2 (1.92 s prefill: N = 26, first generated step f = 28, A sample 55752 = 2.323 s). A samples [N·1920, N·1920 + 5832) = [49920, 55752) are silent (nothing generated has arrived yet).**
  - **delayed (option)**: gen_pd.py exactly: A(t) = partner(t − 5832) everywhere (real opening, then generated).
- Seeds: the state of side k = seed + 10·k (= gen_pd). Sampling = the duplex worker's existing defaults (LP.DEFAULTS = the live demo), forcing and engine calls = the replay path.
- Outputs: b.wav (B), a.wav (A as B heard it), mix.wav (a + b), a_own.wav (A on its own timeline), words.json (ch A = the A state's lane words on the heard timeline), tokens.npz (+ pose_a / mimi_a / lane_a), summary.json (dup_mode, prefill_line, line_delay_ms, delay_starts, seeds). feed.jsonl / avatar = B only (seq engine, same render path).
- Replay (`dup_mode=replay`) = the old path unchanged (A = real recording / silence; Call 1086 is replay only).

## Milestones
- [x] M1 first usable, 05:25 UTC: duo worker live (05:11:37, changelog v2026.10.25-20) + page block / dupui_api.py (05:16-05:17, v2026.10.25-21). Verified: batch B20261026-051149-7f86 (runs 20261026-051149-d935 / -e6e8, Both + GT): both sides' audio, Param + Vansh seq renders (render-pg30554 / render-vansh-pg30556), Conversation player with A synced to the video (drift < 0.1 s), 0 page errors, no sideways scroll at 1280/1440/1920/390, monoplex unchanged. Main told.
- [x] M2 all features, 05:30 UTC: aloedemo applied all 5 patches (v2026.10.25-22: B-only card text + A separately, zip with mix/a/b/a_own wavs, Model switch keeps gt/stream/vw, History Model filter + "real A (replay)" for old runs, duplex tooltip). Re-verified on the public link: 0 page errors on duplex + monoplex, no sideways scroll at 1280/1440/1920/390, flags survive mono↔duplex switches, batch API serves text + text_a. Second duplex worker: the pool starts a dup unit (unit.sbatch dup = duo code) on slinky-18 whenever ≥ 2 duplex runs wait; unit.sbatch turns dup → mono on nodes without the node-local export-498 DCP (backup unit.sbatch.bak-pre-dupui-ck-1026). GT toggle on duplex = the avatar actor's own GT clip (gt-<person>), as monoplex.

## Test results (dupui_test/, c0 side 0, 12 s, seed 4242, job 30562)
- duo aligned: B 19 / A 21 generated words, 132.8 ms/frame for the 2 states, GPU 62.1 / 40.6 GiB. a.wav checks: prefill = stem_1 to 16-bit rounding; [49920, 55752) exactly 0; after = a_own(t − 5832) exactly.
- duo delayed: a.wav = 0 for t < 5832, then stem_1(t − 5832) for the opening, then a_own(t − 5832) exactly.
- replay: unchanged path, 64.6 ms/frame.

## Log
- 05:05 read state; audit fork → DUPUI_AUDIT.md (most UI already model-agnostic; gaps: A audio never audible, batch text mixes A+B, labels, Model switch drops flags).
- 05:20 worker_dup.py: run_duo + rows dup_mode / prefill_line (backup worker_dup.py.bak-pre-dupui-1026). Test harness dupui_test/t.py (duo-aligned, duo-delayed, replay; c0 side 0, 12 s) on job 30562.
- 07:14 UTC: aloedemo rolled the exact light decoder (posefix light_exact_ppm.py) into LIVE/live_ppm_pose.py; the duplex worker now runs in job **30602** (unit.sbatch dup; job 30562 + its DUPUI_LIVE gate are retired). dupui checked the staged engine first (same LivePPM kwargs, a_mute unchanged, no globals). Post-roll check run 20261026-072011-6959 (c0 side 0, 12 s, seed 4242): both sides generated (B 15 / A 14 words), 123.2 ms/frame, avatar rendered; a.wav seam exact (prefill = stem_1 to 16-bit rounding, [49920, 55752) = 0, after = a_own(t − 5832)).
