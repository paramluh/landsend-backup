# pose: the PPM 32B pose models (speech + words + pose)

A snapshot, taken 10-27 ~14:45 PT, of the code, configs and docs behind our pose models: the PPM 32B pretraining trunk with its pose
side, the 8,1,1 pose stage and its s2p + G0COND continuation (job 31101), the Aloe / Xabi / Rosie fine-tunes, the inference engines and
playground workers, and the pose analyses and page builders.

This is a copy. Every file was copied read-only from its live location; `MANIFEST.tsv` maps each repo file to its source path, source
mtime and sha256. **Paths inside the code are absolute and point at the live tree** (`/data/home/amartyavarma/...`, node-local
`/scratch/amartyavarma/...`); they were kept as they were, so a script run from here still reads and writes the live locations. There are
no checkpoints, data stores, caches, media or logs here (see "Not in this repo").

## The system

PPM ("pose + podcast + text-duplex model"), spec in `docs/DESIGN.md` (10-11):

- **Trunk:** base Qwen3-32B, 64 layers, plain RoPE on the frame index, full fine-tune, no LoRA.
- **Input per frame:** for each channel (A = user, B = assistant), the text-lane embedding plus the Mimi-code embedding of frame minus CD,
  summed. Pose rows add the pose-code embedding on channel B. The live run uses `PPM_CD=2` (`configs/prod_811_fresh_append5_s2p.env`).
- **Speech pair:** a Qwen3-4B tower plus a 0.6B depth decoder (Mimi, 32 codebooks x 2048).
- **Pose pair:** a second Qwen3-4B tower plus a second 0.6B decoder (posev3 tokens, 32 codebooks x 2048).
- **Tower conditioning:** only the trunk's final normed hidden state at T - 1, projected 5120 -> 2560.
- **Text head:** the Qwen3-32B lm_head plus the speech control rows (PAD / NEW_WORD / EOB).
- **0.6B decoders:** nanostepped. In the live run there are `PPM_DEC_K=8` updates per micro over 100% of the frames.
- **8,1,1 schedule** (`train/README_811.md`, `PPM_S811=1`): each optimizer step runs 8 pose micros, 1 speech micro and 1 text micro
  (`PPM_GA_SEQ=pose*8,speech,text`).
  - Pose micro 1 trains the backbone. Pose micros 2-8 keep the trunk frozen: it runs forward only (`PPM_BB_MODE=full,frozen*7,full,full`).
  - The pose light decoder gets 64 updates per step.
  - The D33 backbone -> pose-decoder link is off (`PPM_DECP_BB_OFF=1`).
  - The goal (user, 10-20): "focus almost all the gains of pose into the tower and light pose and make sure the backbone doesn't forget
    the other tasks".
- **s2p** (`s2p/README.md`, user 10-27 05:15 PT, "speech now conditions pose also"): the pose tower's input gains `s_emb` of channel B's
  speech codes at the **same** frame T.
  - The flag is `PPM_S2P`. The live run uses `cb0`, which reads speech cb0 only.
  - `s_emb` starts at zero, so the run starts from exactly the old maths.
- **G0COND** (`ops/pose5/STATUS.md` "Change #2", user 10-27 ~06:00 PT): each pose clip is conditioned on frame-0 root xy plus shoulder width.
  - `g_proj(f)` is added to the trunk input embedding at the clip's SLOT position.
  - `f = [r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w]`, and an invalid frame 0 gives f = 0.
  - The projection starts at zero. Pose rows only. The flag is `PPM_G0COND=1`.
  - The new parameters train in their own optimizer groups, `s2p` and `g0cond`.

## Model lineage

| model | what | source |
|---|---|---|
| Qwen3-32B | trunk init (+ random-init 4B towers / 0.6B decoders) | `docs/DESIGN.md` |
| PPM pretraining (`ppm-v1`) | pose + podcast + text-duplex pretraining; constant LR. The pure-PPM run's last save is dcp-16861 | `train/prod_ppm.env`, `train/README_811.md` |
| PPM local-11488 | PPM step 11488, the init of the Aloe pose fine-tunes | `finetune/monoplex/README.md`, `finetune/xabi/README.md` |
| monoplex export-357 | `aloe-pose-mono-8020`, job 24272: monoplex Aloe pose FT from 11488, 357 steps, 80/20 prompt rule. The deployed playground model | `infer/engine/README.md` |
| 8,1,1 from dcp-16861 | job 26391 (launched 10-20 23:13 PT) from PPM's STOP_AFTER_SAVE dcp-16861 | `train/README_811.md` |
| 8,1,1 ends 17352 -> 17641 -> 18022 | fresh-pose deliveries appended (plans "ends [17352, 17641, 18022, ...]"). The leg 17641 -> 18022 was run 27653 | `ops/pose5/STATUS.md`, `train/fresh/` |
| dcp-17850 | paused save of 27653 (10-22 22:29 PT); HF `InternalCan/ppm-v1-c` | `ops/pose5/STATUS.md` (C DONE) |
| **ppm811 31101** (live) | from dcp-17850 with delivery 5 (npl5) + s2p cb0 + G0COND, max step 19382; RUNNING since 10-27 11:41 PT on 16 hosts | `ops/pose5/STATUS.md` ("POSE RUN LIVE"), `configs/prod_811_fresh_append5_s2p.env` |
| Xabi FTs | Aloe pose v2pure monoplex FTs (job 23051's recipe) from several bases: 7932, 13533, 16610 (PPM), 17352, 17641, 17850 (8,1,1) | `finetune/xabi/`, `finetune/xabi/series/`, `finetune/xabi-main-17850/` |
| pose6 (prep) | the monoplex FT again with s2p cb0 + G0COND, from the 31101 final (19382) or 17850; NOT launched | `finetune/pose6/STATUS.md`, `finetune/pose6/kit/README.md` |

## Component map

| dir | what | live source |
|---|---|---|
| `train/` | **the trainer of the live run**: `model_ppm.py`, `train_ppm.py`, `data_ppm.py`, `s811.py`, `lib/` (data/heads/pose libs), `vendor/` (kernels), run envs `prod_*.env`, `ppm811.sbatch`, watchdogs, `tests/` (incl. `tests/s2p`, `tests/g0`, `tests/s2p_postmerge`), `test811/`, `fresh/` (fresh-pose ingest: `fresh_pose.py`, `fresh/append/ingest_more.sh`, plan/append tools, EVALS per leg) | `ppm/pose5/code-811-s2p` (= `ppm/code-811` + s2p + G0COND) |
| `configs/` | the live run's env (`prod_811_fresh_append5_s2p.env`), its resolved-config diff vs 27653, and `FANTASTIC_SET.json` (user's live-inference params) | `ppm/pose5/`, `ppm/` |
| `data/npl5/` | delivery-5 ingest: fetch/verify/scan lanes, off-host plan build, dry runs, placement, launch-day staging scripts + SUMMARY | `ppm/pose5/npl5` |
| `s2p/` | the s2p branch: README (design + port recipe), TESTS.md (proofs), FT trainer copy (`code/`), engine copy (`infer/`), diffs, the dry FT config | `ppm/pose-s2p` |
| `ops/pose5/` | the pose5 leg: STATUS.md (full log), AUDIT_1-4, EVALS_pose5.md, the s2p/G0COND patches, launch/switch scripts, `kit/` (staging, hold chains v1-v4, watchdogs v1-v3, restage-from-HF, gates, end-of-run, hand-back), `c/` (dcp-17850 location/verify records), `g0-dev-tests/` | `ppm/pose5` |
| `finetune/monoplex/` | monoplex recipe (export-357): env, sbatch, run script, `code/` (`train_aloe_pose.py`, `data_aloe_pose.py`, ...), tests, README | `ppm/aloe-pose-mono-8020` |
| `finetune/monoplex-sam/` | the same monoplex retrained on SAM-separated Mimi codes (code = only files that differ from `finetune/monoplex/code`) | `ppm/aloe-pose-mono-8020-sam` |
| `finetune/xabi/` | the Xabi-data (Aloe v2pure) FT recipe, `code/` (incl. `build_v2pure_rows.py`), tools, `stores/README.md`; `series/<base>/` = the scaling-series run scripts per base (see its README) | `ppm/aloe-v2pure-ft`, `ppm/aloe-v2pure-{7932,13533,16610,17352,17641,clip80}` |
| `finetune/xabi-main-17850/` | the 17850-base Xabi FT (job 27757) + `monoft-prep/` (its 8-host prep/status) | `ppm/aloe-v2pure-main-17850`, `ppm/monoft` |
| `finetune/rosie/` | Rosie person FT: env, run, item/person builders, `code/` | `ppm/aloe-pose-rosie` |
| `finetune/pose6/` | monoplex FT + inference with s2p cb0 + G0COND (prep): `code/` (FT trainer), `kit/` (launch kit), `infer/` (engine + G0 sources), `worker/` (playground worker variant), `enginefix/`, `mono17850/`, `g0aloe/` (Aloe frame-0 builder) | `ppm/pose6` |
| `finetune/store-builders/`, `finetune/store-builders-sam/` | the Aloe handover pose stores' builders (job 22091's `code/`, `build_poseB_rows.py`) and the SAM-codes swap | `ppm/aloe-pose-ft`, `ppm/aloe-pose-ft-sam` |
| `infer/engine/` | **the deployed monoplex engine** (`ppm/fast_ppm.py`, `infer.py`, vendored `posetok/`), self-contained package + README | `ppm/aloe-pose-monoplex-infer` (has its own git remote; working tree copied, no `.git`) |
| `infer/playground/` | genweb playground (monoplex + duplex): `server.py`, `worker.py`, `worker_dup.py`, `front.py`, `index.html`, history/dupui/livedup APIs, `pose_scorer.py`, pool/flex/deploy scripts, schemas, status docs, page checks | `ppm/aloe-posedup-live/genweb` |
| `infer/posedup/` | pose-duplex live engine: `live_ppm_pose.py`, `light_exact_ppm.py`, `server.py`, `proxy.py`, `prompts_ppm.py`, `prefill_clip.py`, `feed.py`, `demo.sbatch`, `tools/`, `fastmg/` (fast engine dev), `posefix/` (pose-quality fix work: code + docs only), `verify/` scripts | `ppm/aloe-posedup-live` |
| `infer/pose-server/` | the hot PPM pose-FT server for samples (PPM worker, queue, gallery; video worker left out) | `ppm/ppm-pose-server` |
| `eval/` | `posedyn/` (pose-dynamics onestep vs free test, 10-27), `gaze-drift/` (gaze-drift investigation, 10-22), `posedrift/` (CFG vs drift, 10-26), `gestures/` (Rosie gesture counts + `gesturemetric/` gesture-speech coupling metric) | `ppm/posedyn`, `ppm/gaze-drift`, `ppm/aloe-posedup-live/posedrift`, `ppm/agents1026/gesturemetric`, `ppm/aloe-pose-rosie/gestures.py` |
| `render/` | `pose_infer.py`: the skeleton renderer the pose samples pages use (`pose_infer.render`, 8B SAM page's renderer, imported unchanged) | `qwen-1ch/infer/sam_pose/code/infer/pose_infer.py` |
| `pages/samples/` | pose samples page builders (`aloe-8b-vs-32b`, `pose-hands-cmp`, `aloe-posemono-poseonly`, `aloe-v2pure-200`, `aloe-v2pure-long`, `xabi-ft-*` ctl) | `ppm/samples/*/work`, `ppm/samples/xabi-ft-*/ctl` |
| `pages/xabi-view/` | Xabi viewer page builder | `ppm/xabi-view/code` |
| `pages/klweb/` | live-page builders for the pose run (copies, never edited at the source): `ppm_doc.py`, `ppm.tpl.html`, `build_stage3.py`, `stage3.tpl.html`, the pose/ppm `inc.js` tabs, `pose_tabs.py`, `s3branches_ppm811.json` etc., refresh sbatches, and the Aloe pose / v2pure FT page builders | `setup-logs/klweb` |
| `docs/` | `DESIGN.md` (PPM spec), `history/` (diffs `ppm/code` -> `code-811` -> `code-811-s2p`), `REDACTIONS.md` | |

## How to run each piece

Everything targets the slinky Slurm cluster and the live paths. Read each component's README or STATUS first, because most launches are
gated on kits, holds and checks.

- **Pretrain / 8,1,1 trainer:** `PPM_ENV=<env> sbatch -N 16 -w <hosts> train/ppm811.sbatch`. The usage line is in its header.
  - The script hard-codes `C=/data/home/amartyavarma/ppm/pose5/code-811-s2p`.
  - The live run's env is `configs/prod_811_fresh_append5_s2p.env`. 31101 was launched with `PPM_RESUME_FROM=dcp-17850` pinned at
    submission. The pose5 launch and switch path is `ops/pose5/launch_pose5.sh` and `ops/pose5/switch_to_pose.sh`.
  - The old production entry point is `train/ppm.sbatch` with `prod_ppm.env`.
  - CPU unit tests: `train/tests/cpu_all.sh`, `train/test811/test_*_cpu.py`, `train/fresh/append/test_append_cpu.py`. The s2p and G0COND
    proofs are in `train/tests/s2p/TESTS.md` and `train/tests/g0/`.
- **Fresh-pose ingest:** `train/fresh/fresh_pose.py` (scan / plan), `train/fresh/append/ingest_more.sh` and `ingest_more.sbatch`, and
  for delivery 5 `data/npl5/bin/` (see `data/npl5/SUMMARY.md`).
- **Fine-tunes:** each dir has `<name>.env` + `<name>.sbatch` + `<name>_run.sh`. The run script sources the env, and `sbatch` starts it.
  - monoplex: `finetune/monoplex/aloe_mono8020.sbatch`.
  - Xabi: `finetune/xabi/aloe_v2pure.sbatch` and `finetune/xabi-main-17850/run.sbatch`.
  - Rosie: `finetune/rosie/rosie.sbatch`.
  - pose6: `finetune/pose6/kit/launch.sh` (dry run first, then `--go`; see `kit/README.md`).
- **Monoplex inference:** see `infer/engine/README.md` ("Quickstart"): `fetch_checkpoint.sh` pulls export-357 from the private HF repo
  `InternalCan/ppm-aloe-pose-mono-8020`, then run `python infer.py --ckpt ... --package ...`. It needs 2 x H100.
- **Playground:** `infer/playground/app.sbatch`, `worker.sbatch` and `web.sbatch`. Deploys follow the zero-downtime path described in
  `POOL.md` and `DUPUI_STATUS.md`. The worker imports the deployed engine's `fast_ppm.py`.
- **Pose-duplex live engine:** `infer/posedup/demo.sbatch` runs `server.py`, `proxy.py` and a tunnel. The tunnel URL is not included
  (see `docs/REDACTIONS.md`).
- **Pages:** `pages/klweb/ppm811_refresh.sbatch` runs `S3_PAGE=ppm811 build_stage3.py`, which calls `ppm_doc.py`. The samples pages
  are `pages/samples/<page>/build_page.py` plus its `run_*.sh`.

## Key decisions and results (quoted with their source)

- **8,1,1 resume exactness** (`ops/pose5/STATUS.md`, "POSE RUN LIVE"): at 17850 the start eval is EXACT vs ppmdecay-j27810, and
  [811 rows] 17851/17852 match ppmdecay-27810. The new groups' grad norms are g0cond 0.0020 and s2p 0.0004.
- **31101 start eval at 17850** (same source): eval_pose/word_ce_norm 1.9017, pose_cb0_ce 1.6463, pose_light_ce 3.7304.
  - Steps 17852/17853 took 67.3 s, about 28.6 h to 19382.
  - Running evals are in `ops/pose5/EVALS_pose5.md`, copied as of the snapshot: pose/pose_cb0_ce is 1.6463 at 17850 and 1.6448 at 17984.
- **s2p cost and exactness** (`s2p/README.md`, status 10-27 06:35 PT):
  - With the flag off, and with full or cb0 at zero init, the engine is bit-identical to the deployed engine, both eager and with CUDA
    graphs.
  - Latency per frame is 61.0 ms today, 96.3 ms with full (over the 80 ms live budget) and 66.3 ms with cb0. That is why the run uses
    cb0.
- **Monoplex export-357** (`infer/engine/README.md`): normalised word CE on the 531 held-out items was 2.0112 at step 0 and 1.9156 at
  step 357.
  - Data: 27,517 train items (3,033 h) from 17,527 Aloe handover rows.
  - Recipe: job 23051's v2pure recipe, 357 steps of 64 rows, LR decaying linearly to 0.
  - Prompt rule 80/20: voice clip p 0.8 and persona p 0.2, as independent draws.
- **Xabi FT series, step-200 eval** (`finetune/xabi-main-17850/monoft-prep/STATUS.md`):

  | base | pose cb0 | pose light |
  |---|---|---|
  | 17850 | 1.5788 | 3.6384 |
  | 17641 | 1.5811 | 3.6438 |
  | 17352 | 1.5849 | 3.6530 |

  The same source calls the 17850 base "Best point of the series so far".
- **Generated pose is too still** (`eval/gaze-drift/STATUS.md`, 10-22):
  - The generated pose is about 2x too still. The head is still on 38-61% of frames vs 12% for GT, and the arms move 0.3-0.45x as much
    as GT. Hypotheses H1-H4 were rejected.
  - Pose cb0 top-k off gives tokens identical to the default, so "cb0 is near-deterministic; truncation is not the cause, the model's cb0
    is over-confident".
  - Follow-up: `eval/posedyn/` (10-27, onestep vs free rollout).
- **Pose CFG vs drift** (`eval/posedrift/RESULT.md`, 10-26): CFG "explains the loss of BODY confidence and the head-roll drift", but not
  the hands below the floor or the torso lean. CFG is off by default everywhere.
- **Gesture-speech coupling metric** (`eval/gestures/gesturemetric/RESULT.md`): stroke and timing-kappa definitions, and real vs
  generated results.

The full decision logs are in `ops/pose5/STATUS.md`, `ops/pose5/AUDIT_*.md`, `finetune/pose6/STATUS.md`, `train/README_811.md`,
`train/fresh/STATUS.md` and `train/fresh/append/APPEND_READY.md`.

## Not in this repo

- **Checkpoints:** on private HF repos, named in the READMEs and STATUS files (e.g. `InternalCan/ppm-v1-c` for dcp-17850,
  `InternalCan/ppm-aloe-pose-mono-8020` for export-357).
- **Data, logs, runtime state and caches:** data stores, plans, scans, fresh-pose parquet, `metrics.jsonl`, logs, runs, sessions,
  requests, exports, inductor and triton caches, and runtime state files (job ids, URLs, LOADED/SWAP files).
- **Test micro fixtures** (`ppm/pose5/{s2p,g0}-test-micros/*.pkl`): pickles of 0.35-6.4 MB, over the 2 MB cap, and pickles at that.
  The 6.4 MB `pose.pkl` is the largest.
- **Qwen3-32B tokenizer files:** `merges.txt`, `vocab.json` and `tokenizer.json`, the upstream HF files, dropped from every
  `ppm/configs/Qwen3-32B` copy. `config.json` and `tokenizer_config.json` are kept.
- **`pose_cfg_silence.npy`** (2.4 MB binary, pose CFG uncond input): only its `.json` sidecar is here.
- **Excluded by scope:** the video model, video workers and render jobs (Vansh's), the MoT / motasr / ASR adapter, Voxtral, and
  speech-only demos.
- **`.bak*` files:** none copied. Every component's current file is present.
- **The `ref/pose-tokenizer` repo** (95 MB, external): its vendored inference subset is `infer/engine/posetok/` (see `SOURCES.md` there).
- **External imports kept as absolute paths:** `setup-logs/s3_doc.py` (used by `build_stage3.py`), the `qwen-1ch` stage-3 libs, and
  `ref/pose-tokenizer`.
- **The pose-duplex demo page** (`aloe-posedup-live/index.html`): destroyed by the user 10-27, so it is not included.
- **Voxtral:** out of scope, but `infer/engine/ppm/vadapter_s4q.py` stays in the vendored engine because `fast_ppm.py` /
  `model_ppm.py` import it. The pose model has no vadapter weights.
- **Large analysis outputs:** gaze-drift result JSONs over 300 KB and `train/fresh/pagetest.json`.

Copies of the same trainer appear in several fine-tune dirs (`finetune/*/code`, `s2p/code`, `finetune/pose6/code`). Each is the exact
code that run used, so they were kept. The Xabi series and monoplex-sam dirs hold only the files that differ (see their READMEs).
