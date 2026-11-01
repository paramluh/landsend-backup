# PPM 32B pose-FT server (model kept hot, resumable)

- URL (tunnel): see `URL` here (Cloudflare quick tunnel; rewritten if the tunnel restarts). Internal: `URL_INTERNAL` (port 8840).
- Job: `sbatch /data/home/amartyavarma/ppm/ppm-pose-server/code/start.sbatch` (slinky-15, --gres=gpu:4 -> GPUs 4-7; job id in `JOB`; GPUs in ppm/RESERVED_GPUS.txt).
- Workers: one PPM worker on the first 2 GPUs (code/ppm_worker.py: fast_ppm_srv engine, export loaded ONCE, CUDA graphs captured once, 4 instance rows
  + 4 CFG uncond rows; every knob is a live device tensor), one video worker per remaining GPU (code/video_worker.py: distilled pose-v3 pair,
  12/4 nets + pose encoder compiled at start). Stick figures: CPU subprocess per sample (code/srv_stick.py).
- Queue: requests/<id>/ on disk (request.json, s<k>/ outputs, claims with 45 s heartbeats): a restart finishes anything not done; identical
  requests return the cached result.
- Request fields + defaults: GET /api/defaults (form on /). Defaults = fantastic set (text + speech), pose temp 1 / top-k 30, CFG off.
- CFG uncond (shared by pose and audio CFG): silence_pad = every Mimi input of the uncond stream (voice clip, prefill, generated) -> digital-silence
  Mimi codes AND B's word lane -> PAD at every conversation frame; silence = Mimi only. System prompt text + pose history unchanged; one code
  per frame fed to both streams; towers' uncond rows see the instance's real lane / codes (rows differ in the trunk state only).
  Pose CFG: l = l_u + w (l_c - l_u) on cb0 and/or cb1-31 (scope all / cb0 / light). Audio CFG: the same on speech cb0 only (live-engine cfg_cb0 rule).
- Measured 10-20 19:09 PT: 2 s prefill + 18 s generated + video 12/4: generation 17.8 s (75.6 ms/frame), video 25 s, wall 43.7 s submit -> video.

## Stage 2 (10-20 19:35 PT)
- Gallery: /gallery = every past request (newest first) with ALL its parameters (bold = changed from the defaults), its export, wall time, players.
- Hot-swap: the form's ckpt dropdown (control: srv_common.EXPORTS) = 16,610 / 13,533 / 11,488 / 7,932. 13,533 and 7,932 are node-local shard copies on
  slinky-15 /scratch/amartyavarma/ppm-pose-server/stage/{13533,7932} (64 rank files + rep.pt + meta.json, sizes == the node-local originals; never on /data).
  A request for another export makes the PPM worker exit 3 and reload on it (measured 73-78 s load + capture, ~100 s wall); POST /swap {ckpt} sets the default.
- Persona picker: the live demo's 12 held-out Aloe personas (control/personas.json) into an editable box (default empty = no persona).
- Voice clip: rule (the clip's training-rule clip; prefix SELFTEST == the training-path row for all 30 clips) | none | upload (3-10 s wav).
- Uploaded audio: prompt=upload: wav prefill (mono 24 kHz, -23 LUFS, Mimi; word lane PAD, no ASR) + optional pose npz (prefill_pose, global_state, frame0_pose,
  sink_latent -> video); without pose the pose is generated from frame 0 while the upload's audio is teacher-forced over the prefill (no video).
- Multi-sample: n_samples 1-4 = instance rows of one batch (tested: 2 samples, 18.9 s gen, 42.9 s wall with video).
- Presets (one click): defaults; "pose light top-k 250"; "top-k 250 + pose CFG 3 + audio CFG 3".
- Prompt building (code/prompt_build.py) is re-imported per request: prompt changes need no model reload.

## Video WITH reference images (10-21, user rule: every render uses the video model's reference images)
- code/video_worker.py now renders through video/studio-posemono/pose-ft-samples/s20/code/vrefs_lib.py: the NON-distilled 9-sinks model
  (the distilled 12/4 pair has no reference input), demo-v3 settings (window [9 refs | sink | 3 history | 4 present], 40 stabiliser + 10
  resolver steps, audio guidance 2.0); defaults video_stab_steps 40 / video_res_steps 10. Old worker: code/video_worker.py.bak-distilled-1021.
- Refs: the clip package's bank (refs/property_1..9, demo-v3 pick); missing slots built from the clip's source video (gt/video_latents frames
  re-encoded as single-image latents); uploads with only a sink: all 9 = the sink image. video_status.json "refs" = bank / built counts;
  s<k>/refs/slot<N>.png = the 9 images used. Measured: ~62 s sampling per 20 s clip after the first (graphs compile ~95 s on the first clip).
- When its own queue is empty a video worker renders pose_ft_samples_20s re-render jobs (s20 control/vrefs_jobs.json; no-op once all DONE).

## Persistent HOT extra video workers (10-21 06:28 PT, user: models stay hot)
- s20/code/xvideo_persist.sbatch (jobs on slinky-18 GPUs 0-7 and slinky-4 GPUs 0-5; never slinky-4 6/7): one video_worker.py per GPU, loaded once,
  restarted if it dies, NEVER exits when a run ends; claims every new request here (FIFO). Watch loop: each s20/control/<tag>.watch -> <tag>.vdone when done.
  Stop: touch s20/control/XVIDEO_STOP-<node>. Chained runs: s20/code/chain_runs.py (submit run N+1 when run N is GEN_DONE; records <tag>_params.json).
- Measured: hot render 62 s sampling / 70 s per 20 s clip; a fresh worker's first clip compiles its graphs (+40 s with a warm node cache, +210 s cold).
  With every worker hot a run of the 11 page clips = 179 s generation (10 requests in series on the one PPM worker) + ~70 s render = ~4.2 min.
