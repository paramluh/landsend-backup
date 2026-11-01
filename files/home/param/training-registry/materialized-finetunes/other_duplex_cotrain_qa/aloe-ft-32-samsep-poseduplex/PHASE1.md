# poseduplex PHASE 1 (prep only, 10-25 ~01:50 PT; nothing launched, no running job touched)

Ask (user via main 10-25): the SAM speech-duplex FT (29345 / 28511 recipe) with the assistant's pose summed into channel B with pose
targets, as the Aloe pose FTs did it (26089 = aloe-v2pure-16610 run dir), init PPMVOX v5, steps a knob ("train a bit longer").

## 1. Pose data for the SAME handover none-crossbleed conversations
- Path: `/data/home/amartyavarma/data/aloe-handover-full/DATA/files/none/<session>/<seg>/w00.SPEAKER_0{0,1}.pose_tokens.npy` (+ `.pose.json`,
  index `DATA/pose_manifest.csv`). The local copy of the same stage2-handover-20261008 delivery the sam-sep stores come from
  (sam-sep stores `source: "stage2-handover-20261008 none bucket"`, `src_conversation = <session>/<seg>`). On disk, no download.
- Format: int16 [32, T], pose tokenizer `InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0` (= PPM's, = the 8,1,1 fresh pose
  stream's, so = v5's pose token space), normalisation `frame0-root-frame0-horizontal-shoulder-width-v1` (= PPM's), 25 fps pose -> 12.5 Hz
  tokens (2 pose frames / token), "aligned_to w00.SPEAKER_0N.codes.npy: same T, same 80 ms steps, same cut window". Per-file sha256 in the json.
- Alignment: pose T == manifest mimi_T == the sam-sep rows' src_frames for EVERY first-chunk row (asserted by tools/coverage.py over all
  8,107 rows); first chunks start at frame 0, so B's pose frame f sits with B's Mimi frame f. Model side = PPM's pose rule: pose of frame f at
  position P + CD + f (pose_delay = CD = 2), the same positions as B's Mimi frame f (26089 asserts this per micro).
- Coverage (tools/coverage.json; B = SPEAKER_00 on swap 0, SPEAKER_01 on swap 1):
  - train: **13,578 / 15,898 micros = 85.4 %** have B pose status ok (1,837.3 of 2,125.7 micro-hours = 86.4 %);
  - eval: 279 / 316 micros (88.3 %);
  - conversations (8,107 first-chunk rows): both sides ok 6,162, one side 1,533, none 412.
  - Missing = pose_status 'flagged' (999 tracks) or 'error' (mostly "video ends X s before t", often < 0.2 s short). Not recovered (no
    data change without OK).
- Mimi: the pose files carry NO audio. The conversation dirs also hold raw-lane `mimi-stream-v5` codes (-23 LUFS pure gain + 250-frame / 10 s
  sliding attention, L4: the lufs23-local family); the sam-sep codes are the same recipe on SAM stems (H100). Plan: B speech keeps the
  sam-sep codes from the sam-sep stores exactly as 29345; ONLY the pose tokens are taken from the handover dirs, so there is no second Mimi
  stream and nothing old-encode. (The 22091 `aloe-pose-ft/stores` also carry these raw-lane codes; they are NOT used.)

## 2. How 26089 puts pose in (code: aloe-v2pure-16610/code/{train_aloe_pose.py,data_aloe_pose.py}, model_ppm.py)
- Input: kind 'pose' micro; `pdepth [N, 32]` = B's pose codes (+cb x 2048 offsets) at P + CD + f; `model.inp`: e += inp.pose(pdepth)
  (32 x 2048 code table, -1 = absent -> exactly 0), summed into the backbone input with B's lane / Mimi / A.
- Targets: pose tower `tower_p` (Qwen3-4B arch) on (hn, pdepth): pose cb0 CE (pcb0); pose 0.6B decoder `dec_p` nanosteps on all pose
  frames (`pose_light_frac` 1.0 = PPM's rule, seed (light_seed, h64(row_key), 1, 0x9053)) -> cb1..31; D33 link `decp_bb`
  (Linear 5120->1024): zb = alpha x decp_bb(hn[T-1]) into dec_p slot 0, alpha 1.0, TRAINED ("link on"). D32 pose_head not built.
- Loss: text x LW_WORD + W_CB0 cb0 x LW_SPEECH + W_POSE_CB0 pose cb0 x LW_POSE; decoders x W_CB / W_POSE_CB. 26089: W_POSE_CB0 1,
  W_POSE_CB 1, LW_POSE 1, LW_SPEECH 1, **LW_WORD 10** (its channel-A-empty "uncond" multiplier). Pose terms normalised by the step's pose
  totals / W (tokens_step), as cb0 / light.
- LR / clip: tower_p 1.5e-5 / 1.0, dec_p 2e-5 / 1.0 (own nanostep AdamW, same WSD shape x K), decp_bb 1e-5 / 1.0, inp.pose in 'audio'
  (6e-5 / 1.0). DEC_K 8 for both decoders.
- Prompt: 1/3 voice-clip rule (`ppm-voice` hash % 3), persona as data had it (v2pure: none). Pose CFG: none in training (no uncond
  dropout anywhere); CFG is inference-only and stays OFF.
- 8,1,1 inits (17352 / 17641 / 17850 Xabi FTs): same recipe, link ON from the frozen 16861 decp_bb weights.

## 3. Resolved recipe diff vs 28511 (and conflicts) -- full file in Phase 2: CFG_DIFF_poseduplex.txt
Identical to 28511 / 29345 (kept): trunk 3e-6 / clip 0.5, spec 6e-5, proj 3e-5, audio 6e-5, tower_s 1.5e-5, dec_s 1.5e-5 (all 1.0 clips),
vadapter / vgain 5e-4, betas (0.9, 0.95), wd 0, WSD warmup round(10 x steps / 306) then linear to 0; text x5 factorised, cb0 1, light 1;
dec_s K 1, seed, micro frames 300, mb 470; speech light_frac 0.1 train / 1.0 eval (max 2048); CD 2, attn32, S3_MIN_FRAMES 125; sam-sep
stores, A = in-loop Voxtral latents of the raw-gain FLAC, B clip 30 %, no persona; S3_SWAP none; logical world 64; bf16; ALOE_INIT_VOX=1.
(26089 and 28511 share every one of these LRs / clips: no LR conflict on a shared group.)

ADDED (pose path; values = 26089's): model built WITH pose parts (no `nopose`); groups tower_p 1.5e-5 / 1.0, decp_bb 1e-5 / 1.0 (alpha 1.0);
inp.pose in 'audio'; dec_p AdamW 2e-5 / 1.0; W_POSE_CB0 1, W_POSE_CB 1; pose_light_frac 1.0; pose eval keys; pose data add-on.
CHANGED: init (v5), run names / dirs / HF repos, steps (knob), warmup / export steps (rule-derived from steps).

CONFLICTS (flagged for the user; my recommendation in [ ]):
- C1 decoder K: 28511 dec_s K 1 vs 26089 K 8 (both decoders). [dec_s K 1 (unchanged); dec_p K 1 too (one knob, cheaper). Alt: dec_p K 8.]
- C2 link decp_bb: 26089 / Xabi-8,1,1 FTs ON (trained, 1e-5) vs v5 init has it OFF (frozen 16861 weights, never computed since 16861;
  v5's dec_p trained without it). [ON, = the 17850 Xabi FT precedent on the same base; alt OFF = v5 as trained.]
- C3 word-lane multiplier: 26089 LW_WORD 10 (A-empty uncond) vs 28511 1. [1: it belonged to the A-empty monoplex run.]
- C4 voice clip: 28511 30 % B clip (`aloe-v2-voice` hash) vs 26089 1/3 (`ppm-voice` % 3). [30 %, samsep's.] Persona: none in both.
- C5 pose decoder frames: 26089 all pose frames (pose_light_frac 1.0) vs samsep speech light 0.1. [pose 1.0 = 26089 / PPM; speech stays 0.1.]
- C6 items without B pose (14.6 % of micros): 26089-style = drop them (only qualifying B tracks); samsep = keep all.
  [keep all 15,898 (speech micros identical to 29345, pose where available, pose-less micros run the pose path with no pose: zero pose
  input, zero pose loss); alt = pose-ok items only (13,578).]
- C7 longer than 1 epoch: the samsep plan is one epoch. [epoch e >= 1 = a fresh permutation of the same items (seed 20261011 + e), epoch 0
  = 29345's plan exactly.]

## 4. Hours and steps (logical world 64 = 64 micros / step)
- B speech: 15,898 train micros = 2,125.7 micro-h (= 1,062.9 h of first-chunk conversation, both orientations); eval 316 / 42.6 h.
- Pose-covered: 13,578 micros = 1,837.3 h (86.4 %); eval 279 / 37.8 h.
- Steps / epoch: keep-all (C6 rec.) **249** (warmup 8) = 29345's plan digest d1a3e3721328da0a; pose-ok-only **213** (warmup 7).
  QQA_STEPS knob: N epochs = 249 N (e.g. 498 = 2 epochs, warmup 16).

## 5. Init = PPMVOX v5 (job 29863, still training: vox step 133 / 500 at 01:47 PT, ~39 s/step -> final ~06:00 PT)
- v5 BUILDS and TRAINS the pose parts: log `[build] ... tower_p 4.208B, dec_p 0.575B, decp_bb 0.005B` (no pose_head); main-AdamW groups
  incl. tower_p (401 tensors) and audio (3 = inp.aud x2 + inp.pose); dec_p nanosteps; `FROZEN 2 pose tensors (['decp_bb'])` (link off).
  v5's pose stream = the 8,1,1 fresh lufs23-local pose rows (pose_cb0_ce ~1.68, pose_light_ce ~3.66 in training).
- Its exports (export_local_vox, ppm-local-bf16-shards-v1 world 128, vadapter + vgain_theta_fp32 in meta) carry every pose key: the
  same function's v3 export w-17127 had tower_p 401 / dec_p 314 / decp_bb 2 / inp.pose 1 keys (29345 log: skipped them under nopose).
  So the pose path inits entirely from v5 (decp_bb = the frozen 16861 weights it carries). No other source needed.

## PHASE 2 (10-25 ~03:30 PT): built + CPU-verified, NOT launched. Main's defaults applied as knobs (user still to confirm each)
C1 dec_p K 1 | C2 link ON | C3 LW_WORD 1 | C4 voice v2_30 | C5 pose light 1.0, speech 0.1 | C6 keep all items | C7 fresh perm per epoch | steps 498.

Files (all under poseduplex/):
- code/ = 29345's code dir byte-for-byte (SNAPSHOT_md5.txt) + data_aloe_posedup.py (wraps data_aloe_v2vox, sha-asserted) + train_samsep_posedup.py.
- data/POSE_INDEX.json (sha 7114d5a0...): 16,214 B-side pose files sha256-checked; pose read IN PLACE from /data (no copy, /data at 99 %).
- data/init_subset_v5.json: v5 rank files per host for world 64 (16-18 files + rep.pt; +-1 margin at staging) ~16 GB / host.
- run_poseduplex.env (+ plan table 249/374/498/747/996, pose_ok 213/426), run_poseduplex.sbatch, launch_poseduplex.sh, hf_watch.sh (copy),
  stage_init_v5.sbatch + tools/stage_v5.py, tools/{build_pose_index,plan_posedup,init_subset_v5,coverage}.py, plans/PLAN_<digest>.json.
- CFG_DIFF_poseduplex.txt, DUMP_pose.txt (3 rows), tests/ (identity.py, compare.py, cfg_dry.py, run_tests.sh; out/, dry/, logs/).

CPU checks (slinky-4 / slinky-18):
- pose OFF vs 29345: 4,992 digests (64 logical ranks: step-0 micro + singles + A wav + first eval micro), 0 mismatches -> PASS.
- pose ON vs 29345: same 4,992 digests, 0 mismatches (only kind speech->pose, own +2 columns, pose fields added) -> PASS; 128 pose checks
  vs the token files (positions = B Mimi positions, codes, pcb0, plight = all frames; pose-less = all -1): PASS.
- 2-epoch plan 2275f1eab21f3326: epoch 0 == d1a3e3721328da0a, epoch 1 a fresh permutation, 31,796 micros + 76 dummies; A FLACs 16,214/16,214.
- alt knobs (pose_ok 426 + ppm_third voice): micros build, shift/pose checks PASS.
- trainer cfg dry: vs 26232 abort rule PASS (85 keys, 0 bad); vs 28511 70 keys (all listed in CFG_DIFF), 211 identical.
- NOT testable on CPU: the GPU path (pose decoder nanosteps, FSDP parity on pose-less micros, memory ~60 GiB est., step time est. 12-20 s).

Commands (after v5's final export is on HF, verified; main picks 8 of v5's hosts, ORDER = rank order):
  POSEDUP_HOSTS=h0,h1,h2,h3,h4,h5,h6,h7 sbatch -N8 -w h0,h1,h2,h3,h4,h5,h6,h7 --export=ALL \
      /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/stage_init_v5.sbatch w-18350     # or 'auto' = newest complete w-N; writes control/INIT_TAG
  POSEDUP_INIT_TAG=w-18350 QQA_STEPS_SET=498 bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/launch_poseduplex.sh h0 h1 h2 h3 h4 h5 h6 h7 afterok:<stage job>
      (without POSEDUP_INIT_TAG the launcher reads control/INIT_TAG, which exists only after staging finished)
  Stop: touch poseduplex/control/STOP_AFTER_SAVE | save and continue: touch poseduplex/control/SAVE_NOW.
/scratch per host: init ~16 GB + exports ~11 GB each (intermediates deleted after HF verify) + END full save ~78 GB + 30 floor (save skipped loudly if short).
