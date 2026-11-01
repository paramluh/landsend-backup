# aloe-pose-monoplex-infer

Inference for the **PPM 32B monoplex pose fine-tune** (`aloe-pose-mono-8020`, step 357). Given a short prefill of one speaker, the model generates that speaker's **body pose tokens**, together with their **speech** (Mimi audio codes) and **words**.

This is the model and the sampling path used for the 10-19 pose eval videos: a 2 s prefill from a studio prompt package, then everything autoregressed to 60 s, with pose CFG off.

Everything inference needs is vendored in this repo. The only things fetched at run time are the checkpoint, the Mimi codec and the FA3 kernels; see [What gets downloaded](#what-gets-downloaded).

- [The model](#the-model)
- [Quickstart](#quickstart)
- [Inputs](#inputs)
- [CLI reference](#cli-reference)
- [Outputs](#outputs)
- [Optional: render video with the pose-demo-website DiT](#optional-render-video-with-the-pose-demo-website-dit)
- [GPU and memory](#gpu-and-memory)
- [Known limitations](#known-limitations)
- [Test run](#test-run)
- [Repo layout and provenance](#repo-layout-and-provenance)

---

## The model

| | |
|---|---|
| **Architecture** | PPM 32B. The backbone is Qwen3-32B (64 layers, plain RoPE on the frame index). On top of it sit two Qwen3-4B-architecture towers (one for speech cb0, one for pose cb0) and two Qwen3-0.6B-architecture depth decoders (Mimi cb1-31 and pose cb1-31). A D33 link (`decp_bb`, alpha 1.0) runs from the backbone into the pose decoder. |
| **Streams** | Everything runs at 12.5 Hz. There are two channels: channel A (empty here) and channel B (the speaker), and channel B carries three streams. **Words** are a text lane with word pieces on their true frames, NEW_WORD on the frame before each word, and PAD elsewhere. **Speech** is Mimi, 32 codebooks x 2048. **Pose** is posev3 tokens, 32 codebooks x 2048, from `InternalCan/pose-tokenizer-32cb-200k-noshift`. Mimi and pose codes enter the trunk at code delay CD = 2. |
| **Init** | PPM pretraining, local step 11488. |
| **Data** | The Aloe handover pose rows (`ppm/aloe-pose-ft/stores` p0..p7). Every qualifying track (pose ok, no crossbleed) becomes one **monoplex** item: channel A is empty, and channel B is that speaker's speech, words and pose. This gives 27,517 train items (3,033 h) from 17,527 rows, plus 531 held-out eval items. |
| **Prompt rule (80/20)** | Each item gets two independent seeded draws: a voice clip with p 0.8, and persona text with p 0.2. The clip is the speaker's own 3-10 s clip from a later chunk in the second half of the same conversation, with its lane blanked. Because many tracks have no such chunk, a clip is actually present on 28% of train items, and persona on 20%. |
| **Recipe** | Job 23051's v2pure monoplex recipe. One epoch is **357 steps** of 64 rows (rows of at most 7,500 frames). Warmup is 12 steps, then the LR decays linearly to 0. LRs: trunk 3e-6, towers 1.5e-5, audio and spec rows 6e-5, decp_bb 1e-5. The decoders use K = 8 nanosteps. Compute is bf16 with fp32 masters. Job 24272 trained it on 10-18 (64 H100). |
| **Eval** | Normalised word CE on the 531 held-out items: **2.0112 at step 0 and 1.9156 at step 357**. |
| **Checkpoint** | Step 357, consolidated bf16 DCP: 67 files, 88.5 GB, 2,150 tensors. It is on the PRIVATE HF repo **`InternalCan/ppm-aloe-pose-mono-8020`**, folder `export-357-dcp/`, at commit `87569a04`. This is the same bytes as `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/export-357`: all 67 sizes match and all 65 LFS sha256 match. `CHECKPOINT_SHA256SUMS` lists every file. |

## Quickstart

```bash
# 1. environment (Python 3.10, CUDA 12.8 driver, 2 x H100 80 GB)
python3.10 -m venv venv && . venv/bin/activate
pip install -r requirements.txt

# 2. checkpoint (88.5 GB, private: needs an HF token with InternalCan access). VERIFY=1 also checks every sha256.
bash fetch_checkpoint.sh ./checkpoints            # -> ./checkpoints/export-357-dcp

# 3a. a studio prompt package: 2 s prefill, then words + audio + pose generated to 60 s (the 10-19 eval path)
python infer.py --ckpt ./checkpoints/export-357-dcp --package PROMPTS/10_27286bdd \
    --words EXTRAS/words.json --voice-clip EXTRAS/voice_clip_mimi.safetensors --out out/10_27286bdd --seed 20261026

# 3b. an Aloe-format row, pose only (the row's words + audio forced for 60 s)
python infer.py --ckpt ./checkpoints/export-357-dcp --row ROW.pkl --mode pose --out out/row-pose --seed 1
```

**On the canopy cluster** everything above is already on disk:
- Checkpoint: `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/export-357`
- Package: `/data/home/amartyavarma/video/demo-data/prompts/10_27286bdd`
- Example extras: `/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/example-10_27286bdd/`
- Aloe rows: `/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/rows-np/R-*.pkl`
- The exact test commands: `/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/e2e2.sbatch` and `det.sbatch`

Run on free GPUs only, never on the PPM pretraining hosts. To keep a run offline, set these environment variables to the local copies:

```bash
export HF_HUB_OFFLINE=1 \
  MIMI_DIR=/data/home/amartyavarma/models/mimi \
  FA3_DIR=/data/home/amartyavarma/models/qwen-s2/vllm-flash-attn3
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh   # Python.h for Triton's launcher builds
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH PATH=/usr/local/cuda/bin:$PATH   # --pose-latents: the raster kernel's C++/nvcc build
```

The cluster venv `/data/home/amartyavarma/venvs/qwen-s2` matches `requirements.txt`.

The prompt package itself is not in this repo: it contains a real person's face and voice. The example extras are not here either: they contain a voice clip of the same speaker. Both live in internal storage only. The package also comes from the private HF dataset `vanshjjw/demo-website-requirements` @ `7c694dfe`, under `prompts/10_27286bdd`.

## Inputs

### (a) Studio prompt package (`--package DIR`)

This is the pose-demo-website `demo-data/prompts/<id>` layout:

| file | used for |
|---|---|
| `meta.json` | `uuid`, `ref_seed_key`, `category`, text |
| `llm_inputs.safetensors` | `prefill_audio` [32, 25] Mimi and `prefill_pose` [32, 25] posev3 tokens (2.0 s), `frame0_pose` [133, 3] (pose latents only) |
| `gt/mimi_local.safetensors`, `gt/pose_tokens.safetensors` | [32, 150] (12 s), used only when `--prefill-s > 2`, or as the forced audio in `--mode pose` |
| `sink_latent.safetensors`, `refs/` | `--video-sample` only |

The package carries no LLM text conditioning, so it needs two **extras**. They can sit in the package dir, or be passed with flags:

1. **Word timings**: `words.json` or `--words`. It holds `[[start_s, end_s, "text"], ...]` in package time, covering at least the prefill: words whose NEW_WORD slot falls before frame `prefill + 2` enter the prefill lane. Lists of `[text, start, end]` and `{"start", "end", "text"}` are also accepted. The lane is built with the training rule: Qwen3 word pieces, first piece on `floor(start x 12.5)`, later pieces on the following frames, NEW_WORD on the PAD frame before each word.
   - The first word is tokenised without a leading space, as at the start of a track. If the window starts mid-track, pass `--first-word-space`.
   - `--no-prefill-words` gives a PAD lane instead. This is off-distribution when the prefill has speech.
2. **Voice clip**: `voice_clip_mimi.safetensors` or `voice_clip.wav` in the package, or `--voice-clip WAV|CODES`. It should be 3-10 s of **the same speaker** (it is cropped to 10 s and must be at least 3 s). Tonight's clips came from a later chunk of the same recording; avoid the prompt window itself. `--no-voice-clip` gives the empty system block.

`tools/package_extras_from_row.py ROW.pkl OUT_DIR` makes both extras from an Aloe-format row aligned to the package. That is how the example extras for `10_27286bdd` were made, from tonight's studio row `rows-studio/S-05.pkl`. The script also checks that the lane rebuilt from `words.json` equals the row's lane exactly.

### (b) Aloe-format row (`--row ROW.pkl|ROW.npz`)

This is the trainer's micro for one monoplex item, a dict with these keys:

| key | shape | content |
|---|---|---|
| `P` | int | system block length (header, voice clip and/or persona, footer) |
| `ids` | [2, T] | text lanes of channels A and B (A must be all PAD) |
| `depth` | [2, T, 32] | Mimi codes as **global** ids (`code + 2048 x codebook`, -1 = none), at code delay 2 |
| `present` | [2, T] | channel present flags (A must be 0) |
| `pdepth` | [T, 32] | pose codes, global ids, code delay 2 |

T = P + n + 2 for n conversation frames. The row's own system block is used unless `--voice-clip`, `--persona` or `--no-voice-clip` is given; any of those rebuilds it. In `--mode all` only the first `prefill` frames of the row are read. In `--mode pose` the row's lane and audio are forced over the whole window.

### What is teacher-forced (the prefill)

With prefill N frames (`--prefill-s 2` gives N = 25), positions `[0, P + N + 2)` are teacher-forced:
- the system block;
- Mimi and pose codes of frames 0..N-1;
- **lane frames 0..N+1**: because of the code delay, the lane runs 2 frames ahead of the codes.

Generation then starts at conversation frame N + 2:
- the lane is sampled from frame N + 2;
- audio and pose are sampled from frame N.

## CLI reference

| flag | default | meaning |
|---|---|---|
| `--mode all \| pose` | `all` | `all`: words, Mimi audio and pose are all sampled after the prefill. `pose`: words and audio are forced to the given track for the whole window, and only the pose is sampled. |
| `--prefill-s` | `2.0` | prefill length in seconds (25 frames). Packages allow up to 12 s, read from `gt/`. |
| `--seconds` | `60` | total length including the prefill. In `--mode pose` it is capped by the forced audio. |
| `--temp` | `1.0` | temperature of every stream. `--text-temp`, `--audio-temp` and `--pose-temp` override per stream. |
| `--audio-topk` | `30` | Mimi cb0 and cb1-31 top-k |
| `--pose-topk` | `30` | pose cb0 and cb1-31 top-k |
| `--text-topk`, `--min-p` | `0`, `0` | text top-k and min-p (both off) |
| `--pose-cfg` | `0` (**OFF**) | pose CFG scale. A value > 0 enables it with the silent-speech uncond: every Mimi input becomes digital silence, while the words and pose history stay the same. |
| `--voice-clip`, `--no-voice-clip` | clip **ON** | system-block voice clip |
| `--persona TEXT` | **OFF** | persona text block |
| `--words`, `--no-prefill-words`, `--first-word-space`, `--force-audio` | | package extras (see Inputs) |
| `--seed` | `0` | sampling seed. The text, audio and pose generators get seed + 1, seed and seed + 2. |
| `--devs` | `0,1` | CUDA devices. The backbone is split over them; the speech branch runs on the last device, and the pose branch on the first. |
| `--pose-latents` | off | also write `pose_latents_electron_v3.safetensors` (package input) |
| `--video-sample` | off | also write `video_sample/` for the DiT (package input; implies `--pose-latents`) |

There is no repetition penalty anywhere. All compute is bf16, and nothing is FP8. With the defaults the sampling dict equals the 10-19 studio eval's (`gen_studio_nocfg.py`) exactly, and `meta.json` records this as `sampling_equals_10_19_studio_eval`.

## Outputs

| file | content |
|---|---|
| `audio.wav` | 24 kHz: the B track's Mimi decoded. This is the prefill audio, then generated (or forced) audio. |
| `voice_clip.wav` | the system-block voice clip, decoded |
| `words.json` | the model's word lane as words with time `t` and `prompt` flag, the text before and after the prefill, and the raw lane |
| `tokens.npz` | `audio_codes` [32, n] and `pose_codes` [32, n] (local codes, int16), `lane` [n], `prefill_frames` |
| `pose_tokens.safetensors` | `{"codes": [32, n] int64}`, the posev3 tokens (same format as `python -m posetok`) |
| `mimi.safetensors` | `{"codes": [32, n] int16}`, the demo-v3 audio input |
| `meta.json` | input, mode, P, n, prefill, sampling, seed, checks, speed, peak memory, audio stats |
| `pose_latents_electron_v3.safetensors` | with `--pose-latents`: electron v3 pose latents `[2n/4 + 1, 48, 30, 16]` bf16, decoded against the package's `frame0_pose` |
| `video_sample/` | with `--video-sample`: `sink_latent`, `refs` (symlink), pose latents, `mimi`, `meta.json`. This is a demo-v3 `SAMPLE_DIR`. |

Pose latents use the vendored pose-demo-website `pose/` module (`posetok/`). Its weights are fetched from pinned private HF revisions, or read from `$ELECTRON_ENCODER` and `$POSE_TOKENIZER_DIR`. A small CUDA raster kernel is JIT-built on first use, which needs `nvcc` on `PATH`, `ninja`, and the Python dev headers (`Python.h`). Triton's kernel launchers also need `Python.h` for the main run.

## Optional: render video with the pose-demo-website DiT

This repo does not vendor the video model. Render with the **non-distilled** 9-sinks bimodal DiT from [`canopy-labs-internal/pose-demo-website`](https://github.com/canopy-labs-internal/pose-demo-website) `demo-v3` (commit `ffbd8fc`). Its weights are `vanshjjw/aloe-bimodal-studio-9-sinks` @ `976713a1`, `run_1/step10424/model_ema.safetensors`, pinned in its `settings.yaml`. Use that repo's own environment, then run:

```bash
python infer.py ... --package PROMPTS/<id> --video-sample --out out/<id>
python <pose-demo-website>/demo-v3/video.py out/<id>/video_sample out/<id>/video.mp4 --seed 2026
```

The 10-19 eval videos used the working copy `/data/home/amartyavarma/video/demo-v3-1019`, which is demo-v3 plus `LOCAL_EDITS.md`. With default arguments those edits are bit-identical to stock demo-v3. Video seeds were 2026, and 2027 for the second sample of the same package.

## GPU and memory

- **2 x H100 80 GB** (Hopper: the FA3 kernels and CUDA graphs need sm90).
  - Measured on the test (1 row + its CFG uncond row, 60 s): the first GPU (37 backbone layers + pose branch) peaks at 50.9 GiB, and the second (27 layers + text head + speech branch) at 41.9 GiB.
  - The weights alone are 88.5 GB in bf16 (the 32B backbone plus two 4B towers and two 0.6B decoders), so one 80 GB GPU is not enough.
- **Host RAM:** ~32 GB. The measured peak RSS was 25 GB: the DCP load streams the weights onto the GPUs.
- **Disk:** 88.5 GB for the checkpoint.
- **Time per run** (test: weights read from Weka at ~1.3 GB/s):
  - load + CUDA-graph capture ~65 s, longer on the first run on a node while the Triton kernels compile;
  - **~61 ms per 80 ms frame**, i.e. faster than real time; 725 steps (58 s of generation) take 44 s;
  - ~110 s wall for a 60 s sample.

## Known limitations

- **The pose under-moves compared with real people.** On 10 held-out 60 s Aloe rows with words and audio forced (pose CFG 3), mean wrist speed is **56.7% of real** (95% CI 42-76%). Gestures per 45 s are 4.3 against 10.0 real, and the still fraction is 0.84 against 0.66 real. On the 4 picked eval rows with everything generated, it was 80% (CI 43-127%). Expect calmer, smaller motion than real.
- **Prefill mechanics.**
  - Only the prefill is real: 2 s of audio and pose, and 2.16 s of lane. Everything after it is the model's own.
  - The prefill audio of a package is the package's own Mimi encoding (chunk gain -23 LUFS, native sliding window). It is not the training stores' encoding: cb0 matches 71% on `10_27286bdd`.
  - The lane needs real word timings (see Inputs). Without them, the model sees 2 s of speech with an empty lane.
  - The words are this model's own lane, not ASR. The repo never runs Whisper or any ASR.
- **App vs studio aspect.** The Aloe training pose tokens are normalised in the full **app frame (9:16)**. Studio prompt packages are normalised in the DiT's **256 x 480 frame (8:15)**.
  - Studio packages (prefill and rendering) need no rescale. `--pose-latents` decodes them as the pose-demo-website does.
  - Tokens of app-framed (Aloe) rows decoded with this path stretch vertically by 1.0547. Scale the decoded y-offsets by **0.9481** = (256/480)/(9/16) before `offsets_to_positions`; the 10-19 app cards did this with `studio-posemono/code/app_pose.py`.
  - The model itself sees studio-normalised tokens in a studio prefill, a mild shift from its app-normalised training data.
- **Monoplex.** Channel A is always empty: the model speaks and moves as one person with no partner audio. The model has no partner input; for two-sided conversations use a duplex model.
- **Prompt.** A voice clip is ON by default, but only 28% of training items actually had one. Persona was 20% in training and is OFF here.
- **One row per run.** The engine supports batching, but this CLI runs one input per process, and every run reloads the checkpoint.
- **Reproducibility.** A fixed seed is bit-reproducible on the same GPU type and software stack (see Test run, item 3). The 10-19 eval videos came from the upstream autotuned engine, so they cannot be regenerated bit for bit; they can only be resampled with the same settings.

## Test run

The test ran end to end on 2026-10-19 on slinky-15, a free node (H100 80 GB, not the PPM pretraining hosts). It was offline, with every weight read from its local copy. Scripts: `/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/e2e2.sbatch` (items 1, 2, 4, 5) and `det.sbatch` (item 3).

All outputs are in `/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer-test/`.

1. **Input parity: PASS.**
   - The package path for `10_27286bdd` (with extras made by `tools/package_extras_from_row.py` from tonight's studio row `rows-studio/S-05.pkl`) builds exactly the teacher-forced prefix that tonight's `gen_studio_nocfg.py` fed the model.
   - That prefix is 116 positions: `P = 89` (an 80-frame voice clip, no persona), then 25 frames of audio and pose and 27 lane frames. `ids`, `depth`, `present` and `pdepth` are all equal (`tests/check_package_inputs.py`).
   - `words.json` rebuilds the row's lane exactly over all 752 frames.
2. **Package, `--mode all`, 60 s, seed 20261026 (`det-p1/`): PASS.**
   - Speed: 725 steps in 44.1 s (**61 ms per 80 ms frame**); load + capture 65 s; 109 s wall in total.
   - Peak memory: 50.9 / 41.9 GiB on the two GPUs.
   - Prefill words "Okay. Hi. Um. Do you"; 136 words generated after the prefill ("have a name? Hi Jackie. I'm Michel. You too. What are you like to do? ...").
   - Generated audio: -29.4 dBFS, no clipping. Pose: 36 distinct cb0 codes.
   - The same input run with `--video-sample` (`t1-package-all/`) gave pose latents `[376, 48, 30, 16]`, all finite, and a demo-v3 `video_sample/`.
3. **Reproducibility: PASS after a fix.**
   - Two concurrent runs with the same seed are **bit-identical** in audio, pose and lane over all 750 frames.
   - Upstream, the fast engine autotuned its small-decoder GEMV kernels by timing in every process. The configs have different reduction orders, so the same seed gave different samples run to run: the original script against itself diverged from frame 27-29 in one of two pairs.
   - `ppm/light_kernels.py` now pins one config per kernel (`PPM_AUTOTUNE=1` restores the upstream autotune). Speed is unchanged.
   - Before the pin, this repo matched the original script's first generated frames exactly (lane to frame 54, audio and pose frames 25-26), then diverged exactly as the original does against itself. Exact equality with tonight's videos is therefore not possible, and it was not possible for the original script either.
4. **Aloe row, `--mode pose`, `rows-np/R-004.pkl`, 60 s, seed 1 (`t3-row-pose/`): PASS.**
   - The forced lane equals the row's at all 871 positions, and the forced audio at all 27,872.
   - Speed: 61 ms per step.
   - The pose has only 8 distinct cb0 codes (see the under-movement limitation).
5. **DiT render (optional step): PASS.**
   - `t1-package-all/video_sample/` was rendered by the pose-demo-website demo-v3 non-distilled 9-sinks model (`video.py`, seed 2026) on 1 GPU.
   - Result: 375 latent frames (59.88 s) in 385 s of sampling (about 5x real time), peak 30.1 GiB, written to `t1-package-all/video.mp4`.

## What gets downloaded

| artefact | where | licence |
|---|---|---|
| checkpoint | `InternalCan/ppm-aloe-pose-mono-8020` `export-357-dcp/` (private) | internal (Canopy Labs) |
| Mimi codec | `kyutai/mimi` @ `89091b3e` (override: `MIMI_DIR`) | CC-BY-4.0 |
| FA3 kernels | `kernels-community/vllm-flash-attn3` @ `905da400`, build `torch28-cxx11-cu128-x86_64-linux` (override: `FA3_DIR`) | BSD-3-Clause |
| pose tokenizer (latents only) | `InternalCan/pose-tokenizer-32cb-200k-noshift` @ `1d6ff4f0` (override: `POSE_TOKENIZER_DIR`) | internal |
| electron encoder (latents only) | `vanshjjw/electron-vae` @ `9f183759` `encoder-v2-pose.safetensors` (override: `ELECTRON_ENCODER`) | internal |

Bundled data is only the Qwen3-32B config and tokenizer files and the Qwen3-4B config (Apache-2.0); no Qwen weights are shipped. `infer.py` asserts that every bundled and fetched data source is under a commercially usable licence (`BUNDLED_DATA`, `RUNTIME_DATA`).

## Repo layout and provenance

```
infer.py                      the CLI (inputs -> trainer-layout arrays -> fast engine -> outputs)
fetch_checkpoint.sh           HF download of export-357-dcp (+ VERIFY=1 sha256 check against CHECKPOINT_SHA256SUMS)
ppm/fast_ppm.py               fast engine (CUDA graphs, bf16): ppm/fastinfer/fast_ppm.py
ppm/model_ppm.py              PPM model: ppm/aloe-m1vox-ft/code/model_ppm.py (the engine's copy; pose maths = aloe-pose-mono-8020/code)
ppm/{heads_s2q,speech_q,sysprompt,persona_text_s3q,vadapter_s4q}.py    trainer lib/ (system block, lane rules, heads)
ppm/{fused_triton,fa3_op,pinned_fa3,dec_attn32}.py                     trainer vendor/ (fused kernels, pinned FA3)
ppm/light_kernels.py          fused 0.6B-decoder kernels (qwen-1ch/infer/s3fast; GEMV configs pinned for determinism)
ppm/ppm_io.py                 Mimi / wav / tokenizer / lane helpers (subset of ppm/code/infer_ppm.py)
ppm/configs/                  Qwen3-32B config + tokenizer, Qwen3-4B config (Apache-2.0)
posetok/                      pose tokens -> electron v3 latents (pose-demo-website demo-v3 pose/, see posetok/SOURCES.md)
tools/package_extras_from_row.py   words.json + voice clip for a package from an aligned Aloe row
tests/check_package_inputs.py      package path == the reference row's teacher-forced prefix (CPU)
```

Each vendored file starts with a `# VENDORED ...` line that names its upstream and any edit. The edits are path handling only: repo-local imports, configs and FA3 / Mimi locations. They also remove the unused Voxtral-front and streaming-Mimi imports from the engine, which asserts both off. There is one numeric edit: `light_kernels.py` pins one Triton config per GEMV for determinism (Test run, item 3). The model maths is unchanged, and the inputs match tonight's run exactly (Test run, item 1).
