# Gaze-drift investigation: shared brief (GAZE lead, 10-22 ~20:40 PT)
User question: "Why does gaze drift so much in our generated videos?" GT videos (GT pose tokens -> same decode -> same video model) do NOT drift;
videos rendered from MODEL-GENERATED pose drift. Find the cause with numbers, fast.

## Facts established by the lead (verified in code)
- Pose tokens = posev3, 32 codebooks x F frames at 12.5 Hz; decode -> 2F frames at 25 fps of 55 keypoints (dy, dx, conf):
  WHOLEBODY_INDEX = COCO-WholeBody 0..12 (nose 0, L-eye 1, R-eye 2, L-ear 3, R-ear 4, shoulders 5/6, elbows 7/8, wrists 9/10, hips 11/12)
  + 91..132 (two hands). NO face-68 / eye-contour landmarks: the ONLY head/gaze information the video model gets = nose, 2 eyes, 2 ears.
- Video path (both GT and generated): demo-v3 pose.tokens_to_latents(codes, frame0_pose) in /data/home/amartyavarma/video/demo-v3-1019/pose/__init__.py:
  PoseTokenizer.decode -> offsets_to_positions (kinematic chain, root = frame-0 shoulder midpoint, scale = frame-0 shoulder width)
  -> raw_to_recon -> render_pose_canvas 480x256 -> Electron VAE encoder -> pose latents [2F/4+1, 48, 30, 16] -> 9-sinks video model.
- Generated: ppm/ppm-pose-server/code/video_worker.py prep(): codes.npz pose (prefill 25 real frames + 225 generated) + codes.npz frame0_pose,
  sink = package gt/video_latents frame 0, our generated Mimi codes; render via s20/code/vrefs_lib.py Renderer (40 stab + 10 res steps, audio guidance 2.0, seed from request).
- GT: video/studio-posemono/sam8b-eval-cfg/code/vprep.py write(): package real pose tokens (runs/<row>/codes.npz gt_pose) + package llm_inputs frame0_pose,
  real mimi; rendered by s20/code/vrefs.py (same 9-sinks model, demo-v3 video.run) into s20/vrefs/GT-<idn>/video.mp4.
- !! GT clips are SHORTER: e.g. GT-10_27286bdd = 76 latents = 150 pose frames = 11.9 s, while generated samples are 250 frames = 19.9 s.
  Any drift comparison must be on MATCHED time windows (e.g. 0-2 s prefill, 2-12 s), and separately show what generated does 12-20 s.

## Data
- Generated samples: /data/home/amartyavarma/ppm/ppm-pose-server/requests/<rid>/s<k>/ : codes.npz (pose [32,250] incl. 25 prefill frames,
  prefill_frames, global_state, frame0_pose, audio), meta.json (engine_params, clip id), video.mp4, stick.mp4, vdata/.
- Row maps {row: [rid, k]} (11 rows each) in /data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control/:
  xs17641_fs.json (no CFG, pose_light_topk 50), xs17641_d1/d2/d3.json (pose CFG 1/2/3, pose_light_topk 30). Ignore X76L/XC.
- Rows -> clip ids: S-05 10_27286bdd, S-13 06_74807fe0, S-15 10_27286bdd, S-20 19_47473a46, E-03 03_b437950e, E-05 05_e25c6859, E-09 09_d9ef3c94,
  E-12 12_72cb45d3, E-13 13_956c09af, E-15 15_d1e1322c, E-18 18_1ffa9083.
- GT per clip: /data/home/amartyavarma/video/demo-data/prompts/<clip>/gt/pose_tokens.safetensors (+ llm_inputs.safetensors frame0_pose),
  and /data/home/amartyavarma/video/studio-posemono/sam8b-eval-cfg/runs/<row>/codes.npz (gt_pose, gt_audio). GT renders: s20/vrefs/GT-<clip>/video.mp4.
- CPU decode recipe: ppm/ppm-pose-server/code/srv_stick.py (infer_ppm.decode_pose + PoseTokenizer on CPU, venv /data/home/amartyavarma/venvs/qwen-s2/bin/python,
  env: source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh; HF_HUB_OFFLINE=1). Prefer decoding exactly as the video path does (same tokenizer revision);
  if you use infer_ppm.decode_pose, verify its tokenizer revision equals demo-v3 TOKENIZER (InternalCan/pose-tokenizer-32cb-200k-noshift @1d6ff4f0).

## Compute + rules
- CPU work only via: srun --overlap --jobid 27658 -w slinky-18 -c 4 --cpu-bind=none nice -n 10 <cmd>   (or --jobid 27657 -w slinky-15). -c 4..8 max.
- NEVER use slinky-0,1,2,3,5,6,7,8,9,11,12,13,14,16,17,19, slinky-10, slinky-4. No compute on the login node (reading small files is fine).
- Read-only on request dirs and production code. Never `cd ... &&`; absolute paths. No downloads (HF_HUB_OFFLINE=1). Never print tokens/secrets.
- /data is near its 300 GB floor: keep your outputs < 200 MB, in your own subdir of /data/home/amartyavarma/ppm/gaze-drift/.
- Never kill other processes / Claude sessions.
- Plots: small PNGs (matplotlib), clear axis labels, generated vs GT vs prefill in distinct colours.
