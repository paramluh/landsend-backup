# posefix: fix the live avatar's pose OUTRIGHT (from main amartyavarma-10, 10-25 ~07:35 PT)

You are the tmux agent **posefix**. The user, verbatim:
> "i want you to just fix it outright try out whatever you need simulate inference etc use all non training compute"

The problem: in the live pose-duplex demo (${DEMO_URL}), the avatar's pose is "very fucked up". Your job is to find the cause and FIX it in the live demo: the avatar's body must look natural and stable for whole calls of several minutes. Work autonomously and fast, and spin up as many subagents and jobs as are useful. The user explicitly authorised trying any inference-side change (sampling, prefill, post-processing, engine bugs). Training or recipe changes are out of scope.

## Facts so far
- **The system.** The live server is ppm/aloe-posedup-live (live_ppm_pose.py + server.py; read the module docs). It serves the pose-duplex Aloe FT export-498 (job 30074; EVAL 498: pose cb0 CE 1.0695, pose light CE 2.9394), one bf16 copy per backend: slinky-10 (main) plus lb-B and lb-C on slinky-18, behind proxy.py.
- **Layout.** B's pose codes enter at P + CD + f. The system-prompt positions and the first CD conversation positions carry NO pose. So **the model NEVER sees the chosen actor's real pose: pose prefill = 0 s by construction.**
- **Avatar.** videort (tmux, amartyavarma-75; video/dmd-rt, rt/live.py + rt/engine.py) renders the avatar from the feed's pose codes with the DMD nine-ref video student. It uses the person's sink + 9 refs, plus optional real-video prefill (prefill_s).
- **The user's calls.** Both were Person 01 (video/demo-data/prompts/01_ab63d745, the default now), with **Prefill video = 0 s**:
  - 20261025-142135-d545: pose CFG ON at 3 (cb0 + light). videort measured keypoint confidence 0.20 (vs 0.37 for an earlier no-CFG duplex sample), so limbs fade and flicker. It is also ~3x jittier (speed p50 0.012 vs 0.004) with ~0.5 jumps/s. The whole body drifts right and widens until the right shoulder and arm leave the 256 px frame. Luke's call 20261025-140015-5d7e (CFG 3, default avatar Person 03) has the same low confidence but stays centred.
  - 20261025-142453-1086: pose CFG OFF, 1,479 frames, and the user STILL says the pose is bad. cb0 entropy is 4.7 bits, so the tokens are not collapsed.
  - Session dirs (sessions/<id>/) hold session.json (all params), session.npz (a = the user's mic audio at 24 kHz, b = B audio, pose [F,32], mimi [F,32], pcf = pose-CFG flag) and feed.jsonl.
- **Earlier notes.**
  - memory posedup-live-demo: "generated pose is mostly static (cb0 changes ~8% of frames vs 30-40% in data)".
  - 10-22 gaze study (ppm/gaze-drift/STATUS.md): pose CFG worsens held head angles. The "orient" fix (high-pass the slow head-angle offset, keep fast motion) is in ppm/gaze-drift/render/code/prep_edits.py.
  - Offline pose samples of this FT: claude.ai artifact JfjmHg9GUw9MooWA33aQbC (builder in ppm/fastinfer: fast_ppm.py, parity_pose.py).

## What to do
1. **Build an offline simulator of the live engine.** Use the same live_ppm_pose.py code path, the same checkpoint, bf16. Replay the user's recorded mic audio (session.npz 'a') frame by frame with the same params and seeds. Confirm it reproduces the live pose stream (token-identical for the same seed if possible). Then add held-out Aloe conversations as A input.
2. **Hunt for bugs before tuning.**
   - Teacher-force held-out eval rows through the LIVE engine's pose path. Its pose cb0 / light CE must match the trainer's EVAL 498 (1.0695 / 2.9394); a gap means a port bug.
   - Check the position/CD alignment, the decp_bb link and alpha, the pose code history fed back, and the cond row.
   - Check videort's pose-code → conditioning-skeleton conversion and its frame alignment.
   - Compare the live pose to the offline fast_ppm samples on the same inputs.
3. **Metrics.** Score every variant on replays of BOTH user calls plus ≥ 10 held-out conversations, over ≥ 2 minutes each, against the same metrics on GT pose (the 20 actors' gt/pose_tokens and held-out Aloe B pose):
   - keypoint confidence;
   - jitter / speed p50 and p95;
   - jumps/s;
   - centre drift and width vs the actor's frame-0;
   - out-of-frame fraction;
   - held head pitch / yaw;
   - cb0 change rate.
   Render short avatar clips (via videort's engine, Person 01) for the best candidates.
4. **Levers to try** (inference-only), plus anything else you find:
   - pose CFG off (it is already the default; keep it off unless something beats it);
   - pose temperature / top-k;
   - **pose prefill:** seed B's pose stream with the actor's own real pose tokens (gt/pose_tokens.safetensors, 12 s; verify it is the same tokenizer) for a prefill window, ideally together with the existing video prefill so pose and video both start from the actor's real data;
   - **align:** anchor the skeleton's centre and shoulder width to the actor's frame-0 values with a slow high-pass (remove only the slow drift, keep the motion), clamp limbs inside the frame, and apply the head-angle orient high-pass;
   - smoothing that does not add perceptible latency (the client queue cap is 200 ms end-to-end; any added latency counts against it).
5. **Ship it.**
   - Put the winning combination in the live demo as the default, with each new piece also a visible toggle on the page.
   - Coordinate with aloedemo (amartyavarma-2c, owns the page/server) and videort (owns the avatar engine). Rolling restarts happen ONLY on idle backends (no call open), with a warning to main first. Never cut the user's call; the URL stays the same.
   - Verify with a real public-link call. Then message main ONCE with before/after numbers, a short side-by-side clip path on /data, and what changed.
   - Also state, for the user, how many seconds of pose and video were prefilled in each tested setting.

## Compute and rules
- **Compute.** Use ALL non-training compute: slinky-18, slinky-4 GPUs 0-5, and any free slinky-10 GPUs.
  - NEVER the 16 MoT training hosts (slinky-0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,19).
  - NEVER slinky-9 (another user's node): pass -x slinky-9 on every srun/sbatch.
  - NEVER slinky-4 GPUs 6-7 (thermally throttled).
  - Keep the live serving backends (posedup-live 30190, posedup-lb-B 30346, posedup-lb-C 30347, videort's videort-be/front) untouched. Dev/bench jobs (posedup-fastdev*, videort-replay*) may be shrunk or cancelled after telling their owner (aloedemo / videort).
  - Max out CPUs for prep (measure idle).
- **Commands.** Compute only via srun/sbatch, never on the login node. No `cd … &&` in commands (use absolute paths). bf16 only (never FP8). Never Whisper.
- **Data.** No downloads, no GCS; use only data on disk.
- **Checkpoints.** Never delete any checkpoint or training state. Write nothing big to /data except clips/results (keep /data ≥ 300 GB free); stage weights node-local under /scratch/amartyavarma/posefix/ and check ppm/PROTECTED_SCRATCH.txt.
- **Reporting.**
  - Log progress in ppm/aloe-posedup-live/posefix/STATUS.md.
  - Message main (check ListAgents; it is amartyavarma-10 now; the name is also on line 3 of ppm/control/OPS_PLAYBOOK.md) only with (a) the confirmed cause, (b) the fix live with numbers, or (c) a blocker only the user can resolve. No interim chatter.
