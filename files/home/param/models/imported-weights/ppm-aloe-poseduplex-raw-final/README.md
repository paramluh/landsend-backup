# Aloe duplex pose fine-tune: raw Mimi codes, black-band fix, Zoom text co-train ("pdxraw", job 33111)

This is a full-duplex speech and pose model, fine-tuned from the PPM 32B line.
- **Channel A** is the user. The model hears A through a Voxtral front end. A has no output loss.
- **Channel B** is the assistant. The model speaks B as Mimi codes with a factorised word lane, and moves B's body as pose tokens.

The run trained for 2 epochs (424 steps) on the Aloe handover data. It finished 10-30 01:30 PT, with exit 0.

| | |
|---|---|
| Package (on /data) | `/data/home/amartyavarma/ppm/packages/aloe-duplex-pose-raw-zoom/` |
| HF (private) | `InternalCan/ppm-aloe-poseduplex-raw-final`. That repo holds `export-211/`, `export-424-dcp/` (= the package's `export-424/`), `code/`, `config/`, `README.md` and `SHA256SUMS`. It also holds `export-424/`: the same step-424 weights as world-64 ppm-local rank shards (`rank*.pt` + `rep.pt` + `meta.json`), uploaded straight from the 8 training hosts by the in-job watcher (VERIFY PASS per host). That copy is not in the /data package. |
| Train kit | `/data/home/amartyavarma/ppm/poseduplex-raw/` |
| Hosted | Step 211 (1 epoch) is the checkpoint on the playground duplex **Live mic** (`genweb ?model=duplex`, backend job 33185 on slinky-10) |

## Which checkpoint to use

| ckpt | step | format | held-out normalised word CE | use |
|---|---|---|---|---|
| `export-211/` | 211 (1 epoch) | `ppm-local-bf16-shards-v1`: `rank{0..63}.pt` + `rep.pt` + `meta.json`, world 64 | **1.8004** | **The hosted one.** Use this to reproduce what the user heard on the playground. |
| `export-424/` (HF `export-424-dcp/`) | 424 (2 epochs, final) | torch DCP, consolidated bf16: `__*_0.distcp` + `.metadata` + `train_cfg.json` | **1.7954** | Final weights, with the lowest eval. Not yet compared by ear with 211. |

- Both checkpoints are bf16 weights only. Neither contains optimiser state.
- The full training state at step 424 (`dcp-424`: fp32 masters + AdamW + dec_p AdamW, about 78.5 GB per host) is NOT on HF. It exists only node-local, under `/scratch/amartyavarma/poseduplex-raw/saves/ppm-aloe-poseduplex-raw-j33111/dcp-424` on the 8 training hosts. `/data/home/amartyavarma/ppm/poseduplex-raw/HF_PENDING.md` lists those hosts and how to back the state up.
- In the package, `export-211/` and `export-424/` are **hardlinks** of `/data/home/amartyavarma/ppm/poseduplex-raw/export-poseduplex-raw-{211,424}`. They are the same bytes, so do not edit files in place.

## What it is

### Architecture (PPM 8,1,1 layout with a Voxtral A front)
- **Backbone:** the 32B PPM trunk, Qwen3-32B based. It has 64 layers at 12.5 Hz frames, with codebook delay `PPM_CD=2`.
- **Towers:** two 4B per-frame towers on the trunk's last hidden state:
  - `tower_s` (speech) feeds B's Mimi cb0 and the B word lane (factorised text: cat / word / timing / new-word / EOB).
  - `tower_p` (pose) feeds B's pose cb0.
- **Decoders:** two 0.6B depth decoders:
  - `dec_s` produces Mimi cb1..31. Speech light runs on 10 % of frames in training.
  - `dec_p` produces pose cb1..31. It runs on all pose frames.
  - The D33 `decp_bb` link conditions the pose decoder on the backbone, with alpha 1.0.
- **A front:** Voxtral-realtime encoder (970M, **frozen**, loaded from `/data/home/amartyavarma/models/voxtral-realtime`; it is not in the checkpoint). It feeds the `vadapter` MLP (stack 4, hidden 4096, out 5120), then a learned scalar gain `vgain`. The gain is s = exp(theta). The fp32 theta is in `meta.json` `vgain_theta_fp32` = -5.569, so s = 0.0038138.
- A's Voxtral latents are computed **in the training loop** from the partner's raw-gain waveform. They are never precomputed.
- **B outputs:** Mimi 32 codebooks (kyutai mimi, mimi-stream-v5), plus pose 32 codebooks (`InternalCan/pose-tokenizer-32cb-200k-noshift`, frame-0 root / shoulder-width normalisation), plus the word lane.
- **Base (init):** `w-18100`, the PPMVOX v5 export (private HF `InternalCan/ppmvox-v5-17850`, tag `w-18100`). That is the PPM 32B 8,1,1 pose + podcast model (lineage dcp-17641 / dcp-17850), plus the Voxtral A front trained in ppmvox v5 to step 18100. It was staged at world 64 by `config/stage_init_v5.sbatch`. `ALOE_INIT_VOX=1` keeps the export's own vadapter and vgain, with no recalibration.
- **Recipe lineage:** job 30074 (pose-duplex v5), plus raw codes, the band fix and Zoom co-train. `config/CFG_DIFF_vs_30074.txt` lists all 63 resolved-config keys that differ.

### Data
- **Aloe (speech + pose duplex):**
  - Source: the stage-2 handover **none-crossbleed** bucket (s2h-m1 rows; the `ALOE/s2handover-none/...` row keys). Only **first-chunk** rows are used (every row starts at the call start). Channel A is the user and B is the assistant, as in the data. There is **no A/B swap**.
  - **Clean targets:** every used B side is a clean lane, with no acoustic or transcript crossbleed. `config/data/CLEAN_TARGET.json` (sha256 888b71a5...) is asserted at plan time. 13,894 of 13,894 used items are clean.
  - **Items:** `POSEDUP_ITEMS=pose_ok`, meaning only items whose B side has pose. That leaves 13,578 items. The band drop below removes 12 lanes, so **13,566 train items per epoch**. Eval is a fixed held-out list of **316 micros**.
  - **Packing:** none. One micro is one row (one conversation chunk, with B as the assistant). Each step has **64 Aloe micros**: 32 Aloe GPUs x 2 slots, logical world 64. That gives **212 steps per epoch**, and **424 steps = exactly 2 epochs** (27,132 train micros, plan `7da6116e9f61e36b`, `config/plans/`). Epoch 0 uses 29345's order; epoch 1 is a seeded permutation (seed 20261011).
  - **Prompt:** the voice clip only (rule `v2_30`). On 30 % of rows, B gets a voice-clip prefix cut by the training rule from a later chunk of the same speaker. There is no persona or system text.
- **Raw vs SAM:** `POSEDUP_RAW_CODES=1`.
  - Both channels' **Mimi code values come from the RAW s2h-m1 lane** (`/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores`, mimi-stream-v5, -23 LUFS + 10 s local attention; the codes job 26232 trained on). They replace the SAM-separated codes.
  - Rows, plan, pose, words and A audio are those of the sam-sep stores (`qwen-1ch-data/s2h-m1-samsep/stores`, through `ALOE_SAMSEP_ROW_MAP`), and the voice-clip prefix is raw too.
  - Raw differs from SAM on about 21.8 % of cb0 frames. `config/RAW_CODES.md` has the asserts and tests.
- **Black-band fix:** `POSE_BAND_MASK=1`, `POSE_BAND_MASK_INPUT=0`, mask dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/bandmask`.
  - **(a) Mask.** These B pose frames carry no pose cb0 label (`pcb0 = -100`) and no pose-decoder frame:
    - frames inside a black band, meaning the speaker's no-detection intervals from the index, padded 1 token before entry and 3 after exit (sub-second blips included);
    - frames in runs of 13 or more identical pose frames (all 32 codebooks), padded 6 tokens before and 3 after.

    The loss normalisers follow the masked counts. **The pose INPUT is not blanked** (`POSE_BAND_MASK_INPUT=0`), so the tower still sees the frozen junk-skeleton codes as input.
  - **(b) Drop.** 12 lanes whose clip starts inside a band (index interval start <= 0.04 s) are dropped from the plan (`DROP_LANES.json`). In those lanes the frame-0 skeleton is junk, so frame-0 normalisation is wrong for the whole lane.
  - Masked frames are about 0.1 % of all frames (store m1n0: 11,310 of 10.4M assistant frames).
- **Zoom text co-train:**
  - Data: `InternalCan/zoom-strangers-5k` (local `/data/home/amartyavarma/zoom-charisma/full-5k`): 4,900 train and 100 eval conversations. They are chat-format text on the assistant side with EOB (`zoom_cap` 7500). 4,900 packs make one zoom epoch.
  - Ratio: **1:1 per step.** Each step has 64 Aloe micros and 64 zoom packs. Zoom runs on 4 separate hosts (4 Aloe + 4 Zoom hosts, 2 slots per GPU). The loss is `L = L_aloe + 1.0 * L_zoom`, each a token mean.
  - Over 424 steps that is about 27,136 zoom packs, or about 5.5 zoom epochs.
  - Zoom micros are pose-shaped no-ops (`pdepth -1`, `pcb0 -100`, zero roots), so the FSDP units match. Zoom gives no gradient to pose, the towers, the vadapter or vgain. `config/ZOOM_POSE.md` has the details.

### Recipe
- **Steps:** 424 (2 epochs). Warm-up 14 steps, then **linear decay to 0** at step 424 (WSD with decay from 14 to 424), for every group and both decoders.
- **AdamW:** beta2 0.95, bf16 GEMMs. World 64 on 8 nodes x 8 H100s (hosts slinky-11,12,16,15,19,13,14,17).
- **Learning rates:**

| group | LR | clip |
|---|---|---|
| trunk | 3e-6 | 0.5 |
| spec | 6e-5 | 1.0 |
| audio | 6e-5 | 1.0 |
| proj | 3e-5 | 1.0 |
| tower_s | 1.5e-5 | 1.0 |
| tower_p | 1.5e-5 | 1.0 |
| dec_s | 1.5e-5 | 1.0 |
| dec_p | 2e-5 | 1.0 |
| decp_bb link | 1e-5 | 1.0 |
| vadapter | 5e-4 | 1.0 |
| vgain | 5e-4 | 1.0 |
| Voxtral encoder | frozen | n/a |

- **Loss weights:** text 5, cat-word 1, timing 1, new-word 1, EOB 1, Mimi cb0 1, Mimi cb1..31 1, pose cb0 1, pose cb1..31 1, word-lane multiplier 1. Zoom group weight 1, Aloe group weight 1. Decoder K is 1 for both dec_s and dec_p.
- **Loss masks:**
  - Losses are on **B only**; A is input only.
  - Pose labels are masked inside black bands (above) and on pose-less items.
  - Speech light runs on 10 % of frames (light ratio 0.09-0.11) and pose light on 100 %.
  - The voice-clip prefix is input only.
- **Saves:** exports at steps 70, 141, 211, 283 and 354, plus the final export and the END full save. The intermediate exports are on private `InternalCan/ppm-aloe-poseduplex-raw/`.

## Results (own held-out eval)
The held-out eval is 316 first-chunk Aloe eval micros (n_word 340,743), evaluated every 25 steps.

| step | normalised word CE |
|---|---|
| 211 | **1.8004** |
| 424 | **1.7954** |

The full curve is in `config/run_logs/metrics.jsonl` and `config/run_logs/train-33111.log`.

## How to load and run the live duplex engine
The live engine needs 2 H100s:
- GPU 0: 37 trunk layers, the Voxtral front, the pose tower and dec_p.
- GPU 1: 27 layers, head, tower_s and dec_s.

It runs at about 64-68 ms per 80 ms frame (RTF about 0.83), using CUDA graphs. It loads either checkpoint format straight from disk. The ppm-local shards take about 35-50 s and the DCP takes about 2 min, plus about 50 s of graph capture.

`code/` is a byte snapshot. The engine imports its dependencies by **absolute /data paths** (`code/live_deps/` mirrors them; see the code layout below). On this cluster, run it in place from `/data/home/amartyavarma/ppm/aloe-posedup-live`.

**1. Exactly as hosted** (instance P, playground Live mic). This is the real submit line of backend job 33185:
```bash
sbatch --parsable -J posedup-lb-P-raw211 -w <spare host> -N1 -n1 -c 6 --gres=gpu:2 --mem=100G -t 7-00:00:00 \
  -o /data/home/amartyavarma/ppm/aloe-posedup-live/stage-lb/logs/backend-P-%j.out \
  --export=ALL,BACK=9126,INST=P,CK=/data/home/amartyavarma/ppm/poseduplex-raw/export-poseduplex-raw-211,PPMLIVE_SWAP_TO=/data/home/amartyavarma/ppm/poseduplex-raw/export-poseduplex-raw-211 \
  /data/home/amartyavarma/ppm/poseduplex-raw/host211/backend_P_raw211.sh
```
What the pieces do:
- `backend_P_raw211.sh` pins `CUDA_VISIBLE_DEVICES` to the job's GPU IDX and then execs `stage-lb/backend.sbatch`.
- `backend.sbatch` runs the GPU guard, sources `env_live.sh`, waits for a **verified** CK, and loops `server.py`. A verified CK has `COMPLETE` plus either `DL_VERIFY.json` with no bad file or a `SWAP_OK` file. export-211 has `DL_VERIFY.json`. export-424 has neither, so give it a `SWAP_OK` in your own copy or run `server.py` directly.
- The front proxy and tunnel are `stage-lb/front_p10.sbatch`.

**2. Bare server** (any 2-GPU allocation, inside the job):
```bash
source /data/home/amartyavarma/ppm/aloe-posedup-live/env_live.sh       # venv qwen-s2, PPM_CD=2, GEMM_PRECISION=bf16, VOXTRAL_DIR, PPMLIVE_DEFAULTS=nopersona
export CUDA_VISIBLE_DEVICES=<the job's 2 GPU ids>
export PPMLIVE_STATE_DIR=/path/to/state PPMLIVE_BIND=0.0.0.0 PPMLIVE_SWAP_TO=$CK
echo /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/openings-raw > $PPMLIVE_STATE_DIR/OPENINGS_ROOT.txt   # RAW-code voice clips (required for this model)
$PY -u /data/home/amartyavarma/ppm/aloe-posedup-live/server.py --ck $CK --port 9126 --token-file ~/.s4live_token   # token file mode 600
```
For this model:
- `CK_LABELS.json` (next to `server.py`) must carry the CK's entry `{"prompt": "voice_clip_only", "preset_codes": "raw"}`. That builds prompts from the raw-code rows, the same rule as the trainer's RawRows.
- On load the log should show `PREFIX SELFTEST PASS`, `PREFIX SELFTEST V2 PASS`, `[load] ... 2157 tensors ... skipped prefixes []`, `MODEL ready`, and then `SMOKE PASS`.
- The playground and page client is `genweb` (`?model=duplex`). `code/live/test_client.py` is a headless client.

**Playground default params** (server `DEFAULTS.json` "nopersona", the user's set):
- Text: factorised text, temperature 1, top-k 0, top-p 1, **min-p 0.05 absolute**, freq-pen 0.3 over 50, no rep-pen. Adaptive PAD on (3 s silent, bias -0.1).
- Audio: cb0 T1 top-k 99, light T1 top-k 79. **Audio CFG off.**
- Input: AEC on, NS on, AGC off. A level is the -23 LUFS rule.
- Prefill: 2 s, clip source `rule`, prefill from the start.

**The user's latest live call on this model** (genweb runs `request.json`, 10-30 02:13 PT, ck=211):
- Pose CFG **on**, w = 4 (earlier calls used 3.2, 3, 2 and 1). cb0 guidance off and light guidance on (the 01:58 PT call used w 3.2 with cb0 + light). **keep_mic on.**
- Pose sampling: cb0 top-k 52, light top-k 149, temperature 1, no pose rep-pen.

Exact query string:
```
?v=gw2&model=livedup&ck=211&preset=01_ab63d745&clip=same&person=auto&factorised_text=1&cat_temp=1&pad_bias=0&adaptive_pad_on=1&pad_silent_s=3&pad_bias_silent=-0.1&pad_bias_other=0&text_temp=1&text_topk=0&text_topp=1&minp=0.05&minp_mode=absolute&freq_pen=0.3&freq_win=50&text_rep=1&text_rep_win=64&cb0_temp=1&cb0_topk=99&light_temp=1&light_topk=79&audio_rep=1&audio_rep_win=64&adaptive_rep_on=0&cfg_on=0&cfg_s=1.5&cfg_cb0=0&cfg_uncond_src=quiet_frames&seed=0&max_len_s=600&aec=1&ns=1&agc=0&lim_on=0&pose_temp=1&pose_topk=52&pose_light_temp=1&pose_light_topk=149&pose_cfg_on=1&pose_cfg_s=4&pose_cfg_cb0=0&pose_cfg_light=1&pose_cfg_speaking_only=0&pose_cfg_keep_mic=1&pose_rep=1&pose_rep_light=1&pose_rep_win=50&a_target_lufs=-23&prefill_on=1&prefill_s=2&prefill_audio=1&prefill_words=1&pose_prefill=1&clip_src=rule&prefill_from=start
```

### Voice-clip and opening format (`code/voice_clips/openings-raw/<person>/`)
There is one package per voice, 15 in all. It is the openings-sam set with B's Mimi codes swapped to the **raw** lane codes. Pose and words are byte-identical. All tensors are safetensors with key `codes`, int16, at 12.5 Hz.

| file | contents |
|---|---|
| `voice_clip.safetensors` | `[32, 125]` Mimi codes, B's 10 s voice clip, cut by the **training rule** (a later chunk of the same speaker; same-row second half as a fallback). This is the prompt, input only. |
| `mimi_local.safetensors` | `[32, 150]`, the conversation-start Mimi window (12 s) used for the 1-2 s prefill |
| `pose_tokens.safetensors` | `[32, 150]` pose tokens for the prefill window |
| `clipwin_pose_tokens.safetensors` | `[32, 125]` pose tokens for the clip window |
| `words.json` | `[[t_start, t_end, word], ...]` |
| `clipwin_words.json` | words of the clip window |
| `meta.json` | provenance, the frame-0 pose normalisation, and a `raw_codes` block (rows, frames, sha256, cb0 fraction that differs from SAM) |

- `code/voice_clips/build_openings_raw.py` rebuilds the set from openings-sam and the row map.
- `code/live/prefill_clip.py` and `code/live/prompts_ppm.py` build the prompt and prefill from it.
- An uploaded clip must be raw-lane mimi-stream-v5 codes (-23 LUFS). SAM-separated codes are off-distribution for this model.

## Fine-tuning from it
- Weights-only init: point `QQA_INIT` / `VOX_ADAPTER` at a world-64 ppm-local shard dir, with `ALOE_INIT_VOX=1`. Use `export-211`, or for step 424 the HF `export-424/` rank shards (same layout). Then follow `config/run_poseduplex.env` and `config/launch_poseduplex.sh`:
```bash
POSEDUP_INIT_TAG=<tag> bash /data/home/amartyavarma/ppm/poseduplex-raw/launch_poseduplex.sh h0 h1 h2 h3 h4 h5 h6 h7
```
  This launches 8 distinct hosts x 8 GPUs, and the host order sets the rank order. `COTRAIN_ALOE_HOSTS` in the env names the 4 Aloe hosts; the other 4 run Zoom. Edit it for other hosts.
- The step knob is `QQA_STEPS_SET`. Each step count needs its own plan-table line in the env (`tests/run_tests.sh plan <S> pose_ok`).
- To stop a run, `touch control/STOP_AFTER_SAVE` (never a bare scancel).
- An exact resume needs the node-local END full state `dcp-424` (`COTRAIN_RESUME=...`, the same 8 hosts in the same order). See HF_PENDING.md.

## Caveats
- **Body-relative pose is not addressed.** Pose tokens are normalised to the conversation's frame-0 root and shoulder width, not to the body per frame. Drift and translation in the camera frame are therefore modelled as-is. The band fix only removes junk no-detection frames from the loss; the inputs still carry them.
- **Duplex pose CFG silences the user's mic in the uncond stream.** The pose-CFG uncond row replaces every Mimi input with digital-silence codes and replaces channel A with Voxtral(digital zero). CFG therefore also amplifies "react to the user's audio", which is likely why pose CFG looks worse here than in monoplex. The **`pose_cfg_keep_mic`** toggle is the workaround to test: the uncond row keeps the live mic and silences only B's codes. The user's latest calls had it on.
- Step 211 vs 424: the eval difference is small (1.8004 vs 1.7954). Only 211 has been listened to on the playground.
- **Path dependence.** The trainer and engine use absolute /data paths (stores, row map, bandmask, Voxtral, venvs `qwen-s2` / `duplex`). Moving them off this cluster needs those paths remapped.

## Package layout
```
README.md  SHA256SUMS (sha256 of every file below)
export-211/   rank0..63.pt rep.pt meta.json COMPLETE DL_VERIFY.json SHA256SUMS.txt      (hardlinks)
export-424/   __0_0..__63_0.distcp .metadata train_cfg.json COMPLETE HF_BACKUP.json      (hardlinks; HF: export-424-dcp/)
code/train/code/    exact trainer + data snapshot job 33111 ran (unchanged since launch): train_samsep_posedup.py (trainer),
                    data_aloe_posedup.py (data: pose, raw codes, band mask), data_aloe_v2vox.py / data_aloe_v2.py / data_zoom.py,
                    model_ppm.py, lib/ (speech_q, pose_s3q, vadapter_s4q, voxtral_front, ...), vendor/ (fused/FA3 kernels), task_samsep.sh
code/train/tools/   plan_posedup.py, build_pose_index.py, clean_target_check.py, init_subset_v5.py, stage_v5.py, verify_dcp_host.py
code/live/          server.py, live_ppm_pose.py (engine), prompts_ppm.py, prefill_clip.py, light_exact_ppm.py, a_level.py, feed.py,
                    proxy.py, test_client.py, env_live.sh, CK_LABELS.json, DEFAULTS.json, DUPLEX_SCHEMA.json, pose_cfg_silence.npy(.json),
                    stage-lb/{backend.sbatch, backend_P_raw211.sh, front_p10.sbatch, instP/*}, tools/gpu_guard.sh
code/live_deps/     modules the engine imports by absolute path: aloe-m1vox-ft-code/ (= /data/home/amartyavarma/ppm/aloe-m1vox-ft/code),
                    ppm-code/infer_ppm.py (= ppm/code), qwen-1ch/infer/{s4fast,s4self,s3fast}/, qwen-1ch/stage4/{voxtral_front,vadapter_s4q}.py,
                    aloe-v2vox-ft-code/data_aloe_v2.py
code/voice_clips/   openings-raw/ (15 voices + README/SUMMARY/BUILD_REPORT), build_openings_raw.py
config/             run_poseduplex.env, run_poseduplex.sbatch, launch_poseduplex.sh, stage_init_v5.sbatch, hf_watch.sh, LAUNCHED_JOB,
                    CFG_DIFF_vs_30074.txt, CFG_DIFF_poseduplex.txt (30074 vs 29345), RAW_CODES.md, ZOOM_POSE.md,
                    resolved_train_cfg_424.json (full resolved config), export211_meta.json, plans/PLAN_7da6116e9f61e36b.json,
                    data/{CLEAN_TARGET.json, init_subset_v5.json}, run_logs/{train-33111.log, metrics.jsonl, cfg_diff_*.txt}
```

## Checkpoint file lists (sha256)
The full list for every file, code and config included, is in `SHA256SUMS`. It was re-hashed fresh by the upload job and cross-checked against the hashes recorded at export time.

### export-211 (ppm-local-bf16-shards-v1, step 211), from `export-211/SHA256SUMS.txt` (source-byte hashes, re-verified on /data)

| file | bytes | sha256 |
|---|---|---|
| `rep.pt` | 2,301,002,223 | `eec88220c140178f0739261d81751b6da3cbff62160657b5ca5529c075ab8031` |
| `COMPLETE` | 3 | `093434a3ee9e0a010bb2c2aae06c2614dd24894062a1caf26718a01e175569b8` |
| `rank0.pt` | 1,347,037,149 | `565dda131dae4004ab09c29466aa7965548a50c25f68658c825eb650931b7cee` |
| `rank1.pt` | 1,347,006,173 | `b237ef558e2308faf76fbf6f3a145be2f154093d6ee74371bd46724c4e0f0057` |
| `rank2.pt` | 1,347,006,173 | `c917516808418d81546114cd8dcd36a3d969a6db59a59bf201361b3a0fc4e79d` |
| `rank3.pt` | 1,347,006,173 | `4cf17c3d25dc0d6581f31f0187d5a00ccbd6dfe60c433e1869a73fca77b3ba21` |
| `rank4.pt` | 1,347,006,173 | `209481aaf08195b8dcc376c8cf396a6161874b00bb610eced3cbb982ee0a5c83` |
| `rank5.pt` | 1,347,006,173 | `d8ae6a6d02615fb38095a37bdb8fa2aaf5e3e90773e3e039407fa522b59606b1` |
| `rank6.pt` | 1,347,006,173 | `feedb3b81d528215c04136f3e8dc5b6fa50395476355a13f136788a2cdd7ae9f` |
| `rank7.pt` | 1,347,006,173 | `1f76286c05c33322e32ca91a09f2a2550ce564fe612eefeac0b00c08d0c472e8` |
| `rank8.pt` | 1,347,006,173 | `3bc42ff1b4467b8225165a8e0339a023de58047a75b83ec7e19a760b133374a5` |
| `rank9.pt` | 1,347,006,173 | `84846b937b03fbce81d26b90bece8d27376d7d25447a65ca5062e3435c1450db` |
| `meta.json` | 9,519 | `139fa0120b702a1793abdcd32d3bcae935b6c125d378444be50a5d0af5a8cec2` |
| `rank10.pt` | 1,347,007,708 | `aa4c18f982ed44959faa4dd0da87f226ca8c2ff490b10117448b1c1c8a8ec468` |
| `rank11.pt` | 1,347,007,708 | `f629b62a008102caac255ef80957dfa7fe189099b0079bc8e2053d335b9ada04` |
| `rank12.pt` | 1,347,007,708 | `899e3261eeb3ad7bef3e68b1b9511eaba9d2c54787c127c6ca2717d4f0947013` |
| `rank13.pt` | 1,347,007,708 | `cf48fc424b7c003e39af64f10faf473e3bc32570cec0db21d44f130977d05032` |
| `rank14.pt` | 1,347,007,708 | `59e96ffb40f04fee885b83e89ba365f94d9849ca040be41aa74a69d309e752d9` |
| `rank15.pt` | 1,347,007,708 | `88653844a518ccbe1faab8607f9feefa7444fa2c1a87233ca51eec5ea49325df` |
| `rank16.pt` | 1,347,007,708 | `dc6034a5d5b96c8b36fc24d1b712aae3378e0ea0ffeeb952e31fc304a3c2c368` |
| `rank17.pt` | 1,347,007,708 | `729be9925132918f3a39ca5f9353b4425a11c971464dcbeb87a647dca37402d3` |
| `rank18.pt` | 1,347,007,708 | `cc66b4562e91b5f35dd57e1155ee8b72d031b8824405cf6f073ab73a700e79a9` |
| `rank19.pt` | 1,347,007,708 | `cb7ec6daee19a063b005890253e87a4b376e215e25b94cfdae3ba1d61bbb6c80` |
| `rank20.pt` | 1,347,007,708 | `f671a72365905dc82985d5d1028079948ad2edaeea6768494d9234a3b05faf51` |
| `rank21.pt` | 1,347,007,708 | `176082eda7565eea8580b62faa13d442b89591f687b003adddf701ebb2aed474` |
| `rank22.pt` | 1,347,007,708 | `408a35815273a7fd687b66c218a1dbfb85a114449a4c275bcde07248c8b34f1e` |
| `rank23.pt` | 1,347,007,708 | `1bd56405e93764f611e9510fe2c8052a2a924b35d39c78695c55cb09f11ae690` |
| `rank24.pt` | 1,347,007,708 | `a0c41def345b570a4528774f3d136ff13fe103df895e959ead597bf8e3449feb` |
| `rank25.pt` | 1,347,007,708 | `631fc9380f48b81da9e8deafeb92410ce43566c77b946d9564cc2503bffe6858` |
| `rank26.pt` | 1,347,007,708 | `d2f2ac21736b0c77d0b97186d3c60861420775b026b59a8f1b19ee6afda6f9a3` |
| `rank27.pt` | 1,347,007,708 | `ebed3b76241f53b950e782a1f358b5243c07ed8c1124489a29973c9fe58dbddc` |
| `rank28.pt` | 1,347,007,708 | `f377a08ac1941b7805773347668151af9e612bf1217e74d70fd7dc58b2984c2b` |
| `rank29.pt` | 1,347,007,708 | `a5856bb5afdfc454ea23202c611333db1dd41664aa53dcf460d1857a1c097045` |
| `rank30.pt` | 1,347,007,708 | `016eb774f49f652c31ae96a95a6a01b9f2c56afc79149dc4065f4ffba9b16dd5` |
| `rank31.pt` | 1,347,007,708 | `e9f5c64f77399be6f4de1fb721d356c3566936b667babe288b1f26051c42f3b5` |
| `rank32.pt` | 1,347,007,708 | `67ad342e397d48772902f406dd4f7697ca0bef1a7259b65b27e5b47c9651ed8c` |
| `rank33.pt` | 1,347,007,708 | `565bd971a68d4ecdd66aa3e4b0389947439b8b09c2b257787ad2555e3e8037e8` |
| `rank34.pt` | 1,347,007,708 | `daebfafd5ece72131b7f9de4a8f0406fc702689ca11f8970f02d7644a487b54c` |
| `rank35.pt` | 1,347,007,708 | `04c2dcce6bffe09c10e2b70d45c5b8b307d68a35685011673f45d5070ea194d6` |
| `rank36.pt` | 1,347,007,708 | `f69349c3d24039387e2dda9c9404e88af3a2c42c2bc5efa65ef8c35b27678bce` |
| `rank37.pt` | 1,347,007,708 | `886f776fa07802d486730f85b381fbfe743538704f12d307a77709eb476dd0c1` |
| `rank38.pt` | 1,347,007,708 | `fce93a63f5466cac01453e74ca5eccb4b55ab723d186b6d379c68c07462968a0` |
| `rank39.pt` | 1,347,007,708 | `d149b014b6c9e9a096e5729b27af44ee036f1755f8497581acaa77d7cacc9375` |
| `rank40.pt` | 1,347,007,708 | `9e6d955e6c1add8f5aee589822b9a676f5456f4d0457baa1a9dd4240924d3b2a` |
| `rank41.pt` | 1,347,007,708 | `1fe4f95cbd0e94e9c7d2933babf414d8f2b9062da76a6621df2ba4c3c06ad54c` |
| `rank42.pt` | 1,347,007,708 | `0af7e1fa4bbd5c04f2baf247e6262ceb07d991d3c111072fdfc1f625aeda0e10` |
| `rank43.pt` | 1,347,007,708 | `656367f91f325e11e2be919a3cf07eb4fc132021275d3865572b75ca7d11bf4d` |
| `rank44.pt` | 1,347,007,708 | `640055ff7f8c58111793c07d5a2f32d6d735be55ca14a6308596c04eaffa513a` |
| `rank45.pt` | 1,347,007,708 | `3520cd48e9acf84c962ce72f459185151de5344968959bd97c927131535c3bd8` |
| `rank46.pt` | 1,347,007,708 | `e6c5c17f680443df667c097af76a3e1faf608573eec25be200be874add7a0655` |
| `rank47.pt` | 1,347,007,708 | `378859509caee180fac083cb9b830178fbf72a2f56f42dec68616977d460dbb1` |
| `rank48.pt` | 1,347,007,708 | `cbe8a2ef154c62e841d9bfe694b7efb1ee4a2585c44ecbcf4b284c6171ba32f6` |
| `rank49.pt` | 1,347,007,708 | `c2be1bd5afa05c16d278f712630406b6128bc44daf09d23755f9d13c7042ed33` |
| `rank50.pt` | 1,347,007,708 | `e888e1a5a1ed7a76c93860e56f56a18e7a467ccc335e321f21c7634785f1bb4d` |
| `rank51.pt` | 1,347,007,708 | `33baeda17ef22c34246baae9f6c680400850072685b198690310444351c877fa` |
| `rank52.pt` | 1,347,007,708 | `65ef9e510521fe7431430f4f346a478dba157c06a0ef83586704a182e90d4c2c` |
| `rank53.pt` | 1,347,007,708 | `5a5a04ff12eb4b86281ced241a2c98f9fd646dfbdff3a62902bc1ae7033ef051` |
| `rank54.pt` | 1,347,007,708 | `bbddeb8c80365eaf9d67886e643cea7cc37773ea9edd7dc540e25732cbcb75ac` |
| `rank55.pt` | 1,347,007,708 | `9eb8544ee148f0e711a684c3d4313e6dbd1e306038b2fe3bbcd7d4631ff17881` |
| `rank56.pt` | 1,347,007,708 | `13643fb0e67e1ccc2f3a343989b38f1e73910ccc6a298d1609ef74f03050d10e` |
| `rank57.pt` | 1,347,007,708 | `cccfb054380307b81fdd7c116bb069e7ae9d68fa917cd7959671ceb402d5ce83` |
| `rank58.pt` | 1,347,007,708 | `f5bc76accc0642b5802d3649996d69371f2ad052aab701188150c50b1ce01d82` |
| `rank59.pt` | 1,347,007,708 | `98deeea9a05edd66dd81c7a92b31d6db1f87ed9ca8ed99ed1a62f0b84966576f` |
| `rank60.pt` | 1,347,007,708 | `98f87d334489da6c9aea6f0f87cc234ecfcc462fb610ad60d542ba175a65e5d6` |
| `rank61.pt` | 1,347,007,708 | `05872c127f9f5eaa2bb6ac026a9300a16a8ceda1460cc650e7b11ea681fafcc4` |
| `rank62.pt` | 1,347,007,708 | `af855c12b1e34c4af655617c4c3fb66ca660fe179a24d6884d581375b1342425` |
| `rank63.pt` | 1,346,372,828 | `d0a9ba20838d14b6a9f821ffae77eac229779fd9abb1222133ef634a0f6016e8` |

67 files, 88,508,885,803 bytes. `DL_VERIFY.json` and `SHA256SUMS.txt` are the verification records themselves; they are hashed in the package `SHA256SUMS`.

### export-424 (torch DCP, step 424; HF `export-424-dcp/`), from the HF upload's fresh sha256 of every file (job 33316, 10-30 01:47 PT)

| file | bytes | sha256 |
|---|---|---|
| `COMPLETE` | 3 | `814fd2e8e45e9a6d3e1f6ff86867aaf2251ccd07f3eed02708fae286192c29e3` |
| `.metadata` | 16,799,630 | `f7d20d7d4e05d4c6c103cecc468dac1d6cabdacddff3a058b326c11375d9e0f3` |
| `__0_0.distcp` | 1,348,956,513 | `0db9352cfead5850c3b00d967a13c5db49b0af168b159a52f25b3ef311911717` |
| `__1_0.distcp` | 1,379,350,103 | `1b5b218559f94e41064722b35e8c13006f2d78fa32d9d49d055d4b0c03831ef7` |
| `__2_0.distcp` | 1,376,203,840 | `f497bb3e911bb66e639c8f130e369c4170b179f131f7a2e1077c780692f5f1f1` |
| `__3_0.distcp` | 1,376,204,416 | `7ec6ec04bcd92339c5f01977184da28d5020770eae4e72c93d3d83ec8f9540b5` |
| `__4_0.distcp` | 1,478,944,486 | `8cc1fd0f5a1a761ff50435e794d30fefec0afa01ee2aa9626e67a0319cfe8598` |
| `__5_0.distcp` | 1,378,305,234 | `373fb9e004029f122114785a824a1dae2e465aea6da0a5ba6d588e680ebe42e2` |
| `__6_0.distcp` | 1,378,312,781 | `5d0d8d3afd2e02a218a41ed46413c41667fa8666ec69a6dc611418d2ff8ec906` |
| `__7_0.distcp` | 1,382,487,045 | `6cdc83914b7ec1fe2ee7deb199a5d89cfd4406c764ff596b90b774eff26b8101` |
| `__8_0.distcp` | 1,382,487,045 | `08118173657656bb1552796fcaaafef8779c00fe279d1bbb9a69c80fd8cfc922` |
| `__9_0.distcp` | 1,380,400,809 | `5118b935e3f2df1f5d9569a2b1eb46cd5125c3a11944bafe8d1ddfa4b9a21fdc` |
| `__10_0.distcp` | 1,378,312,525 | `05eae79fc45a2ee2917b69be008e23e5add9184098cfecb9452c9d009e36c73e` |
| `__11_0.distcp` | 1,378,293,806 | `86db4c7e7ca903e1d1f0d195878019533825d9681ea84b0027b3c89411a1c5d9` |
| `__12_0.distcp` | 1,491,532,385 | `b258f6825a2c0604758dae3fd1679e190a37acdae6e27e060245b4e00cc5eec1` |
| `__13_0.distcp` | 1,377,256,361 | `9395a2ecaf49a899e709bad14c2fe21ea65f3249b70f6f01fe0a87e7d9dc1870` |
| `__14_0.distcp` | 1,380,401,833 | `c03bf1402fbc025f95eebbb564433dc5d7a3928c585eb08f3464cee40229136d` |
| `__15_0.distcp` | 1,376,211,625 | `ee3850dc9d5e8658f09af5a21d90460ad952bc04a7e43cf25881800d88024a8f` |
| `__16_0.distcp` | 1,378,310,395 | `d34e26e04c1a9ca28934c3db56219362b75e7d8a7110623b2d9b3bff97555391` |
| `__17_0.distcp` | 1,380,402,304 | `02c09913b0fcfbaefe83e9a305e263b41d949fffe51c58ee51ceabf6c3133cd3` |
| `__18_0.distcp` | 1,378,294,533 | `358c32cf73c3fa18e06708471271fa5b08c086f8f10017dbcfb09332c17bad6a` |
| `__19_0.distcp` | 1,380,396,928 | `f258b3ca02726c0d0f516613701975b1f66d4e42dec6bd9579f8a4fd11b5252f` |
| `__20_0.distcp` | 1,378,305,449 | `e97c5829f772ab7827b507517d89833e07c2a607a4c35a9945e105d70ee8bffc` |
| `__21_0.distcp` | 1,376,200,727 | `97a8897b49480c124507794ad43a61edcd1ff63ab6450dfba4b9af85b0db9da1` |
| `__22_0.distcp` | 1,376,210,130 | `11de2ce41802cb5ea0e66884f92028aaceda871d16df01adac932e47766857cc` |
| `__23_0.distcp` | 1,380,406,267 | `5c145488968882e62600bdb2a785a64602d1f6859a0c9a06ced26e4a8ae6fca0` |
| `__24_0.distcp` | 1,376,199,150 | `25833401995b0ef3ed7355b925ccbbda50db01d119842755897dd26fd337235f` |
| `__25_0.distcp` | 1,376,204,631 | `98c4bc8871ca628bbee949084f2241fc4c717ab9c9f93c0da0c59d436c4ad0ec` |
| `__26_0.distcp` | 1,380,405,970 | `983ee1f9539709cace5f09facbc00b5674a7ef2049f170fa4a81d6ad3aa613bb` |
| `__27_0.distcp` | 1,378,298,199 | `1bdeeba50ed0029123054c9823f46f15cda967725f6fe1811be98fd0acc608e4` |
| `__28_0.distcp` | 1,380,387,804 | `3b19271658a4aec4536b8e639bc443a4232fb76af332689ddcd5fa993f09e351` |
| `__29_0.distcp` | 1,378,299,735 | `531ca7f4fcf287488b24548585f4d3fcf660538d08170bcf6e7a5b28d0425332` |
| `__30_0.distcp` | 1,380,389,893 | `793bdc9fe1587dd16a89564542f223c576de13ef382b98772d388da1c18495e2` |
| `__31_0.distcp` | 1,491,532,385 | `dc8901e878f54d87a6762b12131530c89951eb93918d1f53659cd1aa009aee59` |
| `__32_0.distcp` | 1,376,199,150 | `7e61a0d0e1a6fb454ba66dac23b691136f28a9f3053c7d7c56bfe9ef27e3c2a7` |
| `__33_0.distcp` | 1,378,290,652 | `fe76a95a57b5a178dd531020a11ba1906e33a0743ebd27bac9baa0b8c5985bbf` |
| `__34_0.distcp` | 1,378,292,741 | `0e0da04f5ecd4dff583e1b12ad38a245028424ce7c204cc41e9ae3374cca515a` |
| `__35_0.distcp` | 1,376,202,583 | `8c9e68254f4ebe55827825aa45782654295336ad116581773dd15f45ea10af75` |
| `__36_0.distcp` | 1,378,301,056 | `98f84b4fb0a38eaea2f798416cc6d979e2019097ef2238c3d656f8e6f9da015d` |
| `__37_0.distcp` | 1,376,200,750 | `b2f444580d5e50d50683208887e6cd30163f084bf93fdbae32c5b0a29d666db3` |
| `__38_0.distcp` | 1,376,202,839 | `90788e351ad3eb561493cdda5d5236c0e706dee7e82030cb52502963d7952bf2` |
| `__39_0.distcp` | 1,378,303,616 | `9bb030bc999996883786d6b1356ed0d0a4763f45795f8c8b8be38affbef8e494` |
| `__40_0.distcp` | 1,378,292,741 | `ddd127023d36921b1fe75f3b268dfd96bae42567c8d7f99f166360a0ef4d57d2` |
| `__41_0.distcp` | 1,376,202,839 | `a97574db458be014e542a02cce0e66f846157b4d0e20fae4abb5b1ea0dbce75d` |
| `__42_0.distcp` | 1,378,303,401 | `c6f35e3551071792f2d8871b5a10825abd6ab2093926424d3c7b4ad05be1cf59` |
| `__43_0.distcp` | 1,378,296,366 | `004749b01ef859abc05e2490f37ed9a93e330de32e01941c3e4817787d856758` |
| `__44_0.distcp` | 1,376,201,006 | `c04f7e9bb76d17595cea8ac99f16063aaa4c8f7680a8185e97388a24a9c58122` |
| `__45_0.distcp` | 1,380,386,483 | `15748c5a9e195d31dcc604f7b97fe71949271d56a7ebac8180a580b9afac927e` |
| `__46_0.distcp` | 1,380,386,483 | `6ae4a6964cff0706b2991bc593a537d28043052f0bb22e03a4d525068140d6b8` |
| `__47_0.distcp` | 1,376,199,429 | `1fe401ea4feef701863a0f833f1a2bd2eaea4a791e7900884a3e69ebd9c95f2d` |
| `__48_0.distcp` | 1,376,235,208 | `17ff9ef9b9336a6e2e224905a6479d4358b103c0387b19a1c80a9f5261ec4079` |
| `__49_0.distcp` | 1,378,294,789 | `364493782e995831b5b623281d99b0d45fa0d0626cd78590465f2b4d216e776e` |
| `__50_0.distcp` | 1,506,223,959 | `228aee37e37ccc864d70aec41b653cb010555f8d3bc730cff84a9fd128d65cfc` |
| `__51_0.distcp` | 1,376,202,327 | `3d9aa9558d6dd13b7fb667269f8c24afb74e823a86c83c9cdce72a226f840d3f` |
| `__52_0.distcp` | 1,376,208,041 | `443be1c3aff8c3df2868f6648695812f8571134d66f6333c9b7e0a1b4767b8e4` |
| `__53_0.distcp` | 1,378,299,991 | `71f2848ad7819052eb30273c3ef9e73fcccf0e1fcd7f06a0b1b33c0562a56169` |
| `__54_0.distcp` | 1,380,397,184 | `9de908994db9837df2f96e7e30c263d2076054b712415544eb872829cf5324f5` |
| `__55_0.distcp` | 1,376,206,464 | `6a9e6244d03718e8e93ab70c61f8cd4a1cbcdd0b86821cb794875a63cb6759dd` |
| `__56_0.distcp` | 1,378,294,533 | `209efc83eb29b309238735f63f9d2028676d95cee370afcfb29ecda160ab2332` |
| `__57_0.distcp` | 1,376,206,464 | `15512b1ceb19ad74d35e414d9e20f02da1c59bd5df5b158dfb116e2dfedcd0fc` |
| `__58_0.distcp` | 1,376,208,297 | `3b5d8a0be134374fa2bf2dabadcd1b8004fc71e4d8d6e27017c81cd1c537fdbf` |
| `__59_0.distcp` | 1,376,200,750 | `7c51ec3e7b808d118970139594a296fdcb3620670fe07978348070fca55323b3` |
| `__60_0.distcp` | 1,378,292,485 | `4e452f07a4c64183559c60395f3b8e24ec17747a35c408279bd6e8f342720656` |
| `__61_0.distcp` | 1,382,488,622 | `b81d9ac603b2223beed3a687defbc1430dec951a9b472154f73923afd9b27b00` |
| `__62_0.distcp` | 1,380,402,130 | `91e542cbc94b40568af2b590af1ae12767b1f8f4c3103c52decf4e62ac1c7ac3` |
| `__63_0.distcp` | 1,379,760,471 | `52523bf4fe17db3378e9de3c8c6e5287dd6ffc501195c232c480ff52074b1de8` |
| `train_cfg.json` | 8,375 | `6bccf7638793be44e06c02183e552fa2458525d8137004a62cc93ecb5eca9ef1` |

67 files, 88,648,897,165 bytes. `HF_BACKUP.json` is the upload marker, written after the upload.
