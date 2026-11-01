# aloe-monoplex-pose-raw-bandfix-355

The **PPM 32B monoplex Aloe pose fine-tune with the black-band fix** ("monoband"), final step 355.

Given a short prefill of one speaker, the model generates that speaker's **words**, **speech** (Mimi codes) and **body pose** (posev3 tokens).

| | |
|---|---|
| Run | `ppm-pose6-monoband-j33072`, Slurm job 33072, 8 nodes x 8 H100 (world 64). Ran 10-29 22:56 to 10-30 00:14 PT, exit 0. |
| Base | The 8,1,1 pose run's `dcp-17850`, converted to `/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850`. |
| Weights | `weights/export-355/`: consolidated bf16 torch DCP, 67 files, 88,522,369,887 B (88.5 GB), 2,150 tensors. |
| Serving | Playground (genweb) monoplex weights id `monoband355`, "Mono 17850 raw + band fix (355)". |
| HF | Private model repo `InternalCan/ppm-aloe-monoplex-pose-raw-bandfix-355` (this folder, as is). |

Contents:

- [Model](#model)
- [Data](#data)
- [Black-band fix](#black-band-fix)
- [Recipe](#recipe)
- [Results](#results)
- [Load and run inference](#load-and-run-inference)
- [Fine-tune from it](#fine-tune-from-it)
- [Known caveats](#known-caveats)
- [Package layout](#package-layout)
- [Weight files](#weight-files)

## Model

The architecture is PPM 32B, the same as `aloe-pose-mono-8020` export-357. It is described in `code/infer/README.md`.

- **Backbone:** Qwen3-32B, 64 layers, plain RoPE on the frame index.
- **Towers:** two Qwen3-4B-architecture towers, one for speech cb0 and one for pose cb0.
- **Depth decoders:** two Qwen3-0.6B-architecture decoders, one for Mimi cb1-31 and one for pose cb1-31.
- **Link:** the D33 link `decp_bb` (alpha 1.0) runs from the backbone into the pose decoder. It is ON in this run (`launch.sh --link-on`).
  - `dcp-17850` had the link off and frozen since 10-20, so turning it on revives that frozen link at step 0.
  - mono8020 trained with it on too.
- **Frame rate:** 12.5 Hz (80 ms frames, 1,920 samples at 24 kHz).
- **Channels:** two.
  - Channel A is always empty: monoplex.
  - Channel B is the speaker, with three streams:
    - **Words:** a text lane of Qwen3 word pieces on their true frames, NEW_WORD on the frame before each word, PAD elsewhere.
    - **Speech:** Mimi, 32 codebooks x 2048.
    - **Pose:** posev3 tokens, 32 codebooks x 2048, from `InternalCan/pose-tokenizer-32cb-200k-noshift`.
- **Code delay:** Mimi and pose codes enter at CD = 2, so the lane runs 2 frames ahead of the codes.
- **Precision:** bf16 compute with fp32 masters. No FP8 anywhere.

## Data

**Source.** The Aloe handover pose stores of job 22091, `/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p0..p7`, read in place.

- `ALOE_DATA=aloe-mono`: every qualifying track (pose ok, no crossbleed) becomes one monoplex item.
- Channel A is empty. Channel B is that speaker's words, Mimi codes and pose codes.
- The Mimi codes are the stores' **raw** codes, not the SAM codes. The sibling model `mono17850sam` uses SAM codes.

**Counts.** These are the trainer's own log lines (`logs/ft-33072.log`, the `[data]` lines).

| | Without the fix (mono17850, job 31240) | **This run (fix on)** |
|---|---|---|
| Train items | 27,517 | **27,492** (25 dropped by the band DROP rule) |
| Train frames | | 136,379,173 = 3,030.6 h |
| Packed rows (<= 7,500 frames) | 22,825 | **22,695** |
| Steps (64 rows / step) | 357 | **355** |
| Held-out eval | 531 items | 531 items, 2,639,831 frames, 529 packed micros |

**Packing.** Each item is its own segment (`pack_attn segments`). Rows hold at most 7,500 frames, and each step takes 64 rows.

- Micros use the layout `row_per_rank`: one row per rank, R = 1 round, with a mean of at most 6,000 positions and a bound of 9,502 positions.
- The plan is asserted in the job: rows digest `ea070582180a59f0`, plan digest `f79c08442f8d9bc6`.

**System prompt (the 80/20 rule).** Every item gets two independent seeded draws, with seed 20261011 and tag `aloe-pose-mono-8020`, keyed per item:

- **Voice clip**, drawn with p 0.8. The clip is the speaker's own 3-10 s clip from a later chunk of the same conversation, with its lane blanked.
- **Persona text**, drawn with p 0.2. It is B's own persona from the Aloe persona sidecar.

Many tracks have no later chunk, so the clips actually present are fewer than drawn:

| | Clip drawn | Clip present | Persona present | Empty block |
|---|---|---|---|---|
| Train | 0.8007 | 0.2818 | 0.2010 | 0.5756 |
| Eval | 0.8192 | 0.2938 | 0.1959 | 0.5744 |

The inference engine builds this block with prompt rule `mono8020`.

## Black-band fix

Some Aloe pose tracks contain "black bands": stretches where no person was detected. Their tokens are a junk or frozen skeleton.

The code is `code/train/data_aloe_pose.py` (`band_store`, `band_drops`, `band_apply`). The knobs are `POSE_BAND_MASK=1`, `POSE_BAND_MASK_INPUT=0` and `FRAMING_COND=0`.

**1. Loss mask.** Channel B's pose frames inside a band get **no pose cb0 label** (`pcb0 = -100`) and **no pose-decoder frame** (removed from `plight`). The step normalisers count only the frames that remain.

- **Band:** the index no-detection interval of the speaker, `[floor(s / 0.08) - 1, ceil(e / 0.08) + 3)`. That is 1 token before entry and 3 tokens after exit, and it applies to sub-second blips too.
- **Frozen runs:** this is a union with runs of at least 13 identical pose frames (all 32 codebooks), padded to `[start - 6, end + 3)`.
- **Mask files:** `POSE_BAND_MASK_DIR=/data/home/amartyavarma/ppm/aloe-pose-ft/bandmask`, as `p<k>/pose_band_mask_<side>.npy`, bool `[rows, 7500]` on the store row's frame grid.
  - Each file is checked against its `BANDMASK.json` sha256 and store digest when loaded.
  - The metadata is in `config/bandmask/`.
- **Coverage:** the offline prediction was 0.103% of all train frames masked. At step 1, the measured masked labels matched that prediction exactly (55 masked cb0 labels).

**2. Drop.** A lane whose clip **starts inside a band** (index interval starts at t <= 0.04 s) is dropped from the item list, not masked. Its frame-0 skeleton is junk, so the normalisation of the whole lane would be wrong.

- 17 lanes were dropped, covering 25 items. They are listed in `config/bandmask/DROP_LANES.json`.

**3. Input not blanked.** `POSE_BAND_MASK_INPUT=0`, so masked frames still feed their (junk) pose tokens as input. Only the loss is masked.

Every other config key equals job 31240's resolved config (`config/CFG_DIFF.txt`, 16 keys differ):

- 7 names and paths;
- 4 `pose_band_mask` keys;
- steps 357 -> 355, with the schedule horizons derived from it.

## Recipe

This is the aloe-pose-mono-8020 recipe (job 24272) with init from `dcp-17850`. The resolved config is `config/train_cfg.json`, the same file as in `weights/export-355/`.

**Steps.** 355 steps of 64 rows = one epoch. Eval every 25 steps.

**Schedule.** Warmup 12 steps, then linear decay (WSD) to exactly 0 at step 355.

- The decoders take K = 8 nanosteps per step: warmup 96, decay to 0 at 2,840.

**Peak LRs:**

| Group | LR |
|---|---|
| trunk | 3e-6 |
| spec | 6e-5 |
| proj | 3e-5 |
| audio (includes the pose input embeddings) | 6e-5 |
| speech tower | 1.5e-5 |
| pose tower | 1.5e-5 |
| decp_bb | 1e-5 |
| speech decoder | 1.5e-5 |
| pose decoder | 2e-5 |

**Optimiser.** AdamW with beta2 0.95.

- Grad clip: trunk 0.5, every other group 1.0.

**Loss weights.**

- Base weights: text CE x5, word / timing / NEW_WORD / EOB x1, speech cb0 x1, speech cb1-31 x1, pose cb0 x1, pose cb1-31 x1.
- Per-stream multipliers on top: B word lane x10, speech x1, pose x1.

**Masks.**

- Channel A is empty: PAD ids, no codes, present 0.
- The pose head (D32) is not built.
- The black-band loss mask and drop described above apply.

**Saves.** `--saves none --final-export-only`: there is no resumable full save (no AdamW state), only the final consolidated bf16 export.

**Commands.**

- Launch: `bash config/launch.sh --go` (submits `config/monoband.sbatch`, which runs `config/monoband_run.sh` with `config/monoband.env`).
- Verify: `config/verify_final.sh` PASSED on 10-30 at 00:20 PT. It checks all 2,150 keys and shapes against mono17850's export-357, that every tensor is bf16 and finite, and the init / link / cfg expectations.

## Results

These are the run's **own held-out eval**: 531 Aloe monoplex items, `eval_speech/*` in `logs/ft-33072.log`.

| Step | Normalised word CE | Pose cb0 CE | Pose cb1-31 CE |
|---|---|---|---|
| 0 (= dcp-17850) | 1.9864 | 2.2298 | 5.2316 |
| 25 | 1.9419 | 1.0907 | 3.0059 |
| 100 | 1.9120 | 1.0761 | 2.9635 |
| 200 | 1.9025 | 1.0720 | 2.9530 |
| 300 | 1.8984 | 1.0704 | 2.9482 |
| **355 (this export)** | **1.8978** | **1.0701** | **2.9471** |

The eval pose CEs exclude the masked band frames: 0.045% of eval pose frames.

## Load and run inference

**Hardware.** 2 x H100 80 GB. The bf16 weights alone are 88.5 GB, so one GPU is not enough. Host RAM ~32 GB.

**Environment.** On the cluster, use the venv `/data/home/amartyavarma/venvs/qwen-s2`. Elsewhere, install `code/infer/requirements.txt`.

### Load the weights in Python

The engine reads the DCP directory directly:

```python
import sys; sys.path[:0] = ["code/infer", "code/infer/ppm"]
import torch, fast_ppm as FP
devs = [torch.device("cuda:0"), torch.device("cuda:1")]
model, ldev, cfgs, binfo = FP.build_model("weights/export-355", devs)   # binfo: step 355, pose True, decp_bb True
```

### Standalone CLI

Use `code/infer/infer.py`. The flags are documented in `code/infer/README.md`, which was written for mono-8020 export-357; the same engine and key set apply here.

**What has been checked.** This exact command has **not** been run end to end on GPUs from this package.

- **Inference on export-355 is proven by the live playground worker.** It runs the same engine (`code/infer/ppm/fast_ppm.py`) on the same bytes; see [Playground serving](#playground-serving).
- **The patched CLI was checked CPU-only** (10-30, slinky-9):
  - `params_of()` validates clean through the engine's `FP.validate`, both with defaults and with `--params config/playground_sampling.json`, so the `clean == PARAMS` assert holds.
  - The DCP metadata of `weights/export-355` has 2,150 tensors, all bf16, with the same keys and shapes as `pose6/mono17850/export-357`.

```bash
cd code/infer
export PPM_CD=2 HF_HUB_OFFLINE=1 MIMI_DIR=/data/home/amartyavarma/models/mimi FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh      # Python.h for Triton's launcher builds
python -u infer.py --ckpt ../../weights/export-355 \
  --package /data/home/amartyavarma/video/demo-data/prompts/10_27286bdd \
  --words /data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/example-10_27286bdd/words.json \
  --voice-clip /data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/example-10_27286bdd/voice_clip_mimi.safetensors \
  --prefill-s 2 --seconds 20 --text-topk 50 --min-p 0.05 --pose-cfg 3 \
  --params ../../config/playground_sampling.json --seed 1 --devs 0,1 --out out/test
```

**Outputs:** `audio.wav`, `words.json`, `tokens.npz` (`audio_codes` / `pose_codes` [32, n], `lane`), `pose_tokens.safetensors`, `mimi.safetensors` and `meta.json`.

**The `--params` patch.** This package's `infer.py` differs from the upstream repo in one patch, shown in `code/infer/PACKAGE_PATCH_infer.diff`:

- **Bug fix:** the upstream CLI still passed the old combined `pose_rep*` keys. Since the engine's 10-26 cb0 / light rep-pen split, those keys failed its `clean == PARAMS` assert after the model loaded. The patch uses the split keys, with the same values (1.0 = off).
- **New flag:** `--params JSON` overrides any engine `PARAMS` key. Use it to get the exact playground settings.

### Playground default sampling for this model

`config/playground_sampling.json` holds the user's latest monoband355 batch: `B20261030-085444-9cce`, 01:54 PT 10-30. The full request is in `config/playground_request_example.json`.

| Stream | Settings |
|---|---|
| Text | temp 1, top-k 50, top-p 1, min-p 0.05; frequency penalty 0.25 over a 22-frame window; no repetition penalty; no lane grammar |
| Speech | cb0 temp 1, top-k 30; cb1-31 temp 1, top-k 30; no repetition penalty |
| Pose | cb0 temp 1, top-k 30; cb1-31 temp 1, top-k 256; no repetition penalty |
| **Pose CFG** | **on, scale w = 3, on cb0 AND cb1-31, at all times** (`pose_cfg_speaking_only 0`) |
| Prompt | voice clip on; persona empty; 1.92 s prefill (pose + audio + words) from the person's call start; 40 s clips |

**Pose CFG** uses the "silent-speech" unconditional row: every Mimi input is replaced by the codes of digital silence, while the words and pose history stay the same. Logits are `uncond + w x (cond - uncond)`.

### Playground serving

This is the proven inference path for this export. The hot worker `genweb-monoband355` (Slurm job 33168, slinky-18, 2 GPUs) has served it since 10-30 07:21 UTC. Its log is `logs/worker-monoband355-33168.log`:

```
[fastppm 07:21:35] [load] DCP /data/home/amartyavarma/ppm/pose6/monoband/export-355: 2150 tensors (58s); not used: []
[fastppm 07:21:35] [build] .../export-355 step 355 (dcp): pose True (D33 alpha 1.0), vadapter False; layers {'cuda:0': 37, 'cuda:1': 27}; ...
[fastppm 07:21:41] captured 8 graphs (text, speech, speech_full, pose, pose_full, pose_full_cfg, seg0, seg1) in 4.5s; NI 1 R 2; cuda:0 50.1 GiB, cuda:1 37.4 GiB
[genweb-worker monoband35533168 08:57:33] run 20261030-085445-bff2: 113 words · pose cb0 changes on 68% of generated frames · 64.3 ms/frame
```

The run in the last line is one of the user's batch with the settings above. The worker runs at ~63 ms per 80 ms frame (faster than real time) and uses 50 + 37 GiB on 2 x H100.

The command that started it:

```bash
sbatch -J genweb-monoband355 -w <node> --gres=gpu:2 -c 8 --mem=120G -t 3-0 --export=ALL,MONO_W=monoband355 \
  /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/stage-monoband/unit.sh
```

`unit.sh` pins `CUDA_VISIBLE_DEVICES` to the job's GPUs. `mono_unit.sh` then sources `env_live.sh` and loops `worker.py`.

To serve the model the way the playground does, copy the files in `code/playground/`:

- `worker.py`: the genweb monoplex worker. It imports `infer`, `fast_ppm` and `ppm_io` from `/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer`, which is snapshotted here as `code/infer/`.
- `mono_unit.sh` with `GENWEB_WEIGHTS=monoband355`.
- `monoband355.json` = `{ck: export-355, prompt: mono8020, openings: openings, voice: own}`.

`code/fastinfer/fast_ppm.py` is the upstream fastinfer engine (`ppm/fastinfer`). It is for reference only: the playground worker uses `code/infer/ppm/fast_ppm.py`.

## Fine-tune from it

The trainer is `code/train/` (`train_aloe_pose.py`, `data_aloe_pose.py`, `model_ppm.py`, `lib/`, `vendor/`). It is the exact snapshot the run used: md5 prefixes `138902cc` / `4bef145a` / `c8ad042f` for `train_aloe_pose.py` / `data_aloe_pose.py` / `model_ppm.py`, the same values logged in `config/CFG_DIFF.txt`.

The pre-launch tools are in `code/train_kit_tools/`: `preflight.py`, `stage_base.py`, `verify_export.py` and `dry_cfg.py`.

**Starting from this export:**

- Set `FT_BASE` / `launch.sh --base` to `weights/export-355`. The trainer's `load_init_dcp` reads a consolidated DCP directly, and its bf16 values become fp32 masters.
- There is **no optimizer state**, so AdamW starts fresh.
- Keep `QQA_DECP_BB_OFF=0` (link on), because this checkpoint trained the link.
- Every knob is explicit in `config/monoband.env`. Data paths point at the cluster stores.

## Known caveats

- **Body-relative pose is not addressed.** posev3 tokens are body-relative, and the model sees only code tables. It has no notion of frame scale or framing: where the hands leave the frame is decided only at decode. This FT does not change that (`FRAMING_COND=0`).
- **Raw codes, not SAM.** The stores' raw Mimi codes were used, not the SAM codes. Prefill and voice-clip audio should come from the same raw-code openings (the playground uses `openings` with voice `own`).
- **Band input not blanked.** Band frames are masked from the loss only. Their pose tokens are still model input during training (`POSE_BAND_MASK_INPUT=0`).
- **Monoplex.** Channel A is always empty, so the model has no partner input. For two-sided conversations, use the duplex FT.
- **Thin prompt coverage.** A voice clip was present on only 28% of training items, and persona on 20%.
- **Pose under-moves.** Earlier monoplex FTs from this recipe moved less than real people. This was not re-measured for this export.
- **Upstream CLI bug (fixed 10-30).** `infer.py` in `/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer` had the `pose_rep` assert bug. The same patch is now applied there too (backup `infer.py.bak-pre-paramsfix-1030`), so the upstream file and this package's `code/infer/infer.py` are byte-identical.

## Package layout

| Path | Content |
|---|---|
| `weights/export-355/` | Consolidated bf16 DCP. `COMPLETE` = 355. `train_cfg.json` = the resolved config. |
| `code/train/` | Exact trainer and data code (`pose6/code_bandmask`), including tests and its own README / TESTS. |
| `code/train_kit_tools/` | `pose6/kit/tools`: preflight, stage_base, verify_export, dry_cfg. |
| `code/infer/` | `aloe-pose-monoplex-infer` at git `6ca0f89`, plus the local changes the playground runs (`PROVENANCE_local_changes_vs_HEAD.diff`) and this package's `infer.py` patch. |
| `code/fastinfer/` | `ppm/fastinfer/fast_ppm.py` + `env_fast.sh` (reference only). |
| `code/playground/` | genweb `worker.py`, `mono_unit.sh`, `stage-monoband/unit.sh`, `env_live.sh`, `monoband355.json`. |
| `config/` | `monoband.env`, `monoband.sbatch`, `monoband_run.sh`, `launch.sh`, `CFG_DIFF.txt`, `train_cfg.json`, `preflight-33072.json`, `verify_final.sh`, `RUN_TAG.txt` / `LAUNCHED_JOB` / `RUNNING_JOB.txt`, `bandmask/` metadata, `playground_sampling.json`, `playground_request_example.json`. |
| `logs/` | `ft-33072.log` (full training log); `worker-monoband355-33168.log` + `genweb-monoband355-33168.out` (the live playground worker serving this export: load + runs). |
| `SHA256SUMS` | sha256 of every file in this package. Check with `sha256sum -c SHA256SUMS`. |

## Weight files

`weights/export-355/`: 67 files, 88,522,369,887 B.

| File | Bytes | sha256 |
|---|---|---|
| `__0_0.distcp` | 1,346,979,010 | `1400d3334049db3aa7840b31a63c1753c7da5bdd8f18d7e0fb03d438482b727f` |
| `__1_0.distcp` | 1,377,374,177 | `3ff8c8ca7ef7a637147710dfaccfc81d84e24bc17a6b030e3ee1200f36b597d8` |
| `__2_0.distcp` | 1,374,226,401 | `706578bf7cab5863d38b071f392931ad6be43da9e726f9c4652c5f1920305deb` |
| `__3_0.distcp` | 1,374,228,490 | `563d13b4ba40c38e5682d8f442a90d3908ea6fdf4829967f8150bdeb43c847b9` |
| `__4_0.distcp` | 1,476,968,560 | `7747392a602fab99624bf306695e6622290e725efaf3d02382f897f1456e5528` |
| `__5_0.distcp` | 1,376,329,308 | `27dbe4a348e6256ab2ff2d01f069ccf67e7c1db49bdf6178983b375248bec9b5` |
| `__6_0.distcp` | 1,376,336,855 | `f86c6663c7f524d61694f8288e59861342607f68dcba442d7e90c355a8078934` |
| `__7_0.distcp` | 1,380,511,119 | `7e4975273367a19f4088c17d61b473e53b082ab673551b1400e349e764876e36` |
| `__8_0.distcp` | 1,380,511,119 | `23a238b7c7f239a3f55a50b5c4faffe9cd12df9e1ec57ec0be13f04599ad6ffc` |
| `__9_0.distcp` | 1,378,424,883 | `37c6e1ef94b6e02264cb8f9185624267c6ef03f1e34ff411b7e652c69dbce3a0` |
| `__10_0.distcp` | 1,376,336,599 | `37e996449edf5033df40752e22324e2358b234e9a6bb406545634416aa169f0c` |
| `__11_0.distcp` | 1,376,317,880 | `bc1a9b494858079862cb7b853e9d2431c4b2896479269ec01c33bb44b738af8a` |
| `__12_0.distcp` | 1,489,556,459 | `58b0ffdea77264635ceba90d698cd25b11cfef5f12a3ef12bfadf96a5a5af776` |
| `__13_0.distcp` | 1,375,280,435 | `a2fcfac4138cb7b89a6a3d5338fef21a6393b39ef8fdf2ecd420fbb01c4b87ef` |
| `__14_0.distcp` | 1,378,425,907 | `9156bdfed05b5670751a2d80f85162d2bbcc4aacb2744b057546e7dafd84b842` |
| `__15_0.distcp` | 1,374,235,699 | `1afe0bbcfb8aa0cb02800fb3ee4b8c3327bff7e068863669efee995a3d8a443f` |
| `__16_0.distcp` | 1,376,334,469 | `4e9e09410c617f253437ad7bfafbbd1bbd2905182bd8a17ff6c338390921bfd8` |
| `__17_0.distcp` | 1,378,426,378 | `2a0d8b404b1b87b5506f0f4ba419366bf4999661ce6507d5bed8313a2e5baa5a` |
| `__18_0.distcp` | 1,376,318,607 | `9aee43d93c9254d6d6544cb8fc93cf48dcf6fe76ccc93431e2c6a6a46982fdd0` |
| `__19_0.distcp` | 1,378,421,002 | `24e35596defa2d85966f8f10e40008f289603d3245b02d7796c2c936f6efd72f` |
| `__20_0.distcp` | 1,376,329,523 | `917ec5d1ac2befebc7129b604e10f707701debf112c75976dff8ca924c7f4728` |
| `__21_0.distcp` | 1,374,223,224 | `50b16d1d683610478986138c2e97fc89307fd5cd11065571eae4fc52c64b56c6` |
| `__22_0.distcp` | 1,374,234,204 | `b678dbd86f344d98d8149d62183ca10d470db482e9e1d27c3bf64879d498338e` |
| `__23_0.distcp` | 1,378,430,341 | `1a24cf25b3919b0d059a3eac549278c7dd06bdc9dc1a6b49a23655025c0502dc` |
| `__24_0.distcp` | 1,374,223,224 | `e42402725dc7240351180c1cc6dc170f5fa662ac315c545b09855feef638817b` |
| `__25_0.distcp` | 1,374,228,705 | `c0ceb5ae7c2ef45e76441efd8deebe037c815a9dbfdffc4addb8a3ad4ce226ca` |
| `__26_0.distcp` | 1,378,430,044 | `fe560b63ff0c45361de1533b970c4683f9c9e29a5e8e971a2a349b76b164ad07` |
| `__27_0.distcp` | 1,376,322,273 | `64da93fba912a79a12661d41aeb80f17ff3f3f6596e746dea6e1eadf13e7aa29` |
| `__28_0.distcp` | 1,378,411,878 | `9ac8c71604759f0e090fc8d8a9a19db61f080f39f4705b305fe43bfd6ea7b0d2` |
| `__29_0.distcp` | 1,376,323,809 | `92bf9240f626ddb909123e28f86b661679e232f1ff6fd0f815226e75a04fb299` |
| `__30_0.distcp` | 1,378,413,967 | `be5b3b892326da2d496567ddb587ba45fb4111127de6467589e66a4eb87134d3` |
| `__31_0.distcp` | 1,489,556,459 | `8c25135b204e25f4228636a10334eb13f017454fdfef13d1d960ab6ec45b7355` |
| `__32_0.distcp` | 1,374,223,224 | `0f7ad3f2cd0446c32d24256044c1005b71c2ebaa2f32b72e61cfdc2e6e412a2f` |
| `__33_0.distcp` | 1,376,314,726 | `7c5b7cf7216e464b8fff51386a829778321df3c3a4dadb5e10edd8ee906c63c6` |
| `__34_0.distcp` | 1,376,316,815 | `befbd7c5028810d0a3a83150a1ce46b16339dae20724a8cd5c3aaf09b16ba497` |
| `__35_0.distcp` | 1,374,226,657 | `169214d784b75e2db68151e491ab2cf2a82432286e0b200d28c48b22689d2c90` |
| `__36_0.distcp` | 1,376,325,130 | `9de56d4f293121ed8cad93cbeda9421c4f853fb44dc5c9e0be17b4ea6b8b5867` |
| `__37_0.distcp` | 1,374,224,824 | `90fb0763f34086502efa6f8c8563372e3315fb4e3ca16593f8b203a3d638691e` |
| `__38_0.distcp` | 1,374,226,913 | `1bb1a07e6839ad092bc3857b85e0b2009d87c0a61a0bf1c62fd39f15155a6eed` |
| `__39_0.distcp` | 1,376,327,690 | `a12602c85250ee03c89f9f30619ed0b8a7c23f5f2e9a7945778bcf71747b170e` |
| `__40_0.distcp` | 1,376,316,815 | `e1fdd970797ef2227becaaa3a9e16ffb3a15c394695943a907e14c7191e6d3ec` |
| `__41_0.distcp` | 1,374,226,913 | `be1deab66dc191e45afb360ddb1c561d3dbf62204eae967ac8a8afdcf74eb3c7` |
| `__42_0.distcp` | 1,376,327,475 | `28094efb549767d77da00265021afa2f98af72ef62f4e141518c7ef498dc806f` |
| `__43_0.distcp` | 1,376,320,440 | `0e5a2e0d854d01648672069ed666c271493dc047bf960ddd7c99e10f1e9b29c9` |
| `__44_0.distcp` | 1,374,225,080 | `ad7604c0e30ed9b4edd46854c2437cf22c7d6552421477a32ed03d32c33b6cbd` |
| `__45_0.distcp` | 1,378,410,557 | `6cb4fe302d32bafdf773f0333fbb7bc8e35c1b25193f4b0ce684d31fa060fbe1` |
| `__46_0.distcp` | 1,378,410,557 | `c42773b733b704032f186a0889430cd1df8d8ef9f94816264a758570c4be7e7e` |
| `__47_0.distcp` | 1,374,223,503 | `13d3337c205374f7a7ea47adcd4ee5cd05b7223ff937bf46c6695b4b07df2f4a` |
| `__48_0.distcp` | 1,374,259,282 | `1bcf835bd5f3913e728497d08bc4a837404a0046a4b28d38afffa4bebd178fb0` |
| `__49_0.distcp` | 1,376,318,863 | `91155b3a948893dcff9aad37cd1c10acf47903afadd58f6dfdda1cfebaddd558` |
| `__50_0.distcp` | 1,504,248,033 | `16e1ff197dadf7a5860b81466ba29ae44f7b4909b46b2255b1b702ee5f302340` |
| `__51_0.distcp` | 1,374,226,401 | `c82bda503e1fa491ab0aa4c2930a0cffaca155e5d0ff4b62fcd402d37087f9cb` |
| `__52_0.distcp` | 1,374,232,115 | `cf331d200f207a05c2d75fbb7bceda9453bb1e0f4460164f95367e79cff26043` |
| `__53_0.distcp` | 1,376,324,065 | `311914d733ccd199e2b738235f77965626068a409d48828e1e7d0545e60dbcef` |
| `__54_0.distcp` | 1,378,421,258 | `f34a27325040b860df7fefa065d9c9eb2a04355266e7b03521a3199e1f591d1e` |
| `__55_0.distcp` | 1,374,230,538 | `481f24bcf1cc866e73efa8a580294ea2479d4426d0ace47c5064fa4472afb522` |
| `__56_0.distcp` | 1,376,318,607 | `92c5683aa7f884cbcc3064cd1e2ce9845c38e4f72b42bb26a41652cfe76bdbed` |
| `__57_0.distcp` | 1,374,230,538 | `b108c889e61524b84fda217d1f51f164c40f4d9535b6913b945861be4a037d73` |
| `__58_0.distcp` | 1,374,232,371 | `9b6f1d3a0889cd10a6d2f5f9ddbb6b589b0f92374e5f3a7779c2a10a180661fb` |
| `__59_0.distcp` | 1,374,224,824 | `d3391fca6ba0b403cf59f3e396d2d0a651082e1ae0346a25711330dd3808adb4` |
| `__60_0.distcp` | 1,376,316,559 | `1572e727c5f70d43352b1ee14fb7a0fdc8290afa2c9650cbc8a3ec4bfb8389cb` |
| `__61_0.distcp` | 1,380,512,696 | `8f3f0504a7d79b476185ea6aa34211d60c251b8feffd76cbbdd70ad4441120c9` |
| `__62_0.distcp` | 1,378,426,204 | `bb26bdb8e0fc9ba82c4c0a6bb5e9ecbc61ee7b244492ae9896d20cfbb7b747ef` |
| `__63_0.distcp` | 1,377,784,545 | `ea26e554f857ff8789e49a590d85f8dc65b25d3842129a397de4569be348a9fb` |
| `.metadata` | 16,740,992 | `120b405dea427d4a28ca9de1a2c091363d73917c41a40583d297731a072b10e6` |
| `COMPLETE` | 3 | `355d8c0ee4e5698eaed38b96aab64dbf0ad72eca3e352183be6e957e9d9230a7` |
| `train_cfg.json` | 3,666 | `be2bd77c1e1d7b6cd3436dbd1c126a5fb7693e06bd595b08df62653405b13dc1` |
