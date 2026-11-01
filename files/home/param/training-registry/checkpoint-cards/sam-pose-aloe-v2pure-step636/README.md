---
library_name: transformers
tags:
- speech
- pose
- tts
- audio-generation
---

# SAM pose, Aloe fine-tune v2pure (Aloe v2 release only), step 636

An 8B speech + body-pose generation model. Give it the full text upfront, and it speaks that text as Mimi audio codes (24 kHz) while generating the speaker's body pose as pose-tokenizer codes (55 joints: face, body and both hands, 25 fps).

This model is trained only on the Aloe v2 release. Every part of its fine-tune data comes from `gs://aloe-processed-data/pose-speech-vad/releases/v2/`: the pose codes, the Mimi codes, the words and their timing, and the item selection. It is the release-only rebuild of `InternalCan/sam-pose-aloe-v2-step636`, which used posev3's cache with the v2 Mimi codes swapped in. The architecture, init, recipe and inference code are the same as v2 and posev3.

- **Architecture:** a Qwen3-8B trunk (36 layers, hidden 4096) plus two Qwen3-0.6B-shaped depth decoders, one for audio codebooks 1..31 and one for pose codebooks 1..31. Each 12.5 Hz frame has 32 Mimi codebooks and 32 pose codebooks. The trunk has its own heads for audio cb0 (2049-way, including END) and pose cb0. See `model.py`.
- **Init:** our SAM 8B speech + pose pretrain (stage 2, step 61258), the same init as posev3. Lineage: SAM text-upfront TTS (stage 1), then the speech + pose model (stage 2, step 61258), then this fine-tune.
- **Fine-tune data:** built only from the Aloe pose-speech-vad release v2 (`gs://aloe-processed-data/pose-speech-vad/releases/v2/`). Nothing is taken from earlier caches.
  - **Items:** the release manifest's own selection: segments with `eligible` and `is_active_speaker` both true. These are active-speaker clips only: the speaker on camera is the one talking, and there is no partner channel. After dropping the few clips that have no words or are too long for a training row, that gives 367,496 training clips (2,047 h of speech, 12,384 recordings).
  - **Pose:** the release's pose codes (`confidence-hold-03-frame0-init-v2`, encoded with `InternalCan/pose-tokenizer-32cb-200k-noshift`).
  - **Audio:** the release's Mimi codes. They were encoded with Mimi's native 250-step sliding-window attention, using the same `kyutai/mimi` checkpoint and 32 codebooks.
  - **Text:** the release's words and word timing.
  - **Held-out split:** the release has no eval split. We applied posev3's recording rule to the v2 items: a recording is held out when `blake2b8('aloe-pose-v1|' + recording_id) % 10000 < 100`. That holds out 3,482 clips from 113 recordings, which are never trained on.
- **Recipe:** identical to posev3 and v2. It is 1 epoch over the 367,496 training clips (636 steps) on 24 H100s, with LR 1e-4 for the trunk and 2e-4 for the depth decoders and new embeddings/heads, 50 warmup steps, then cosine decay to 0.
- **Weights:** a bf16 export (`model.safetensors`, 18.8 GB).

## Final eval (step 636)

| metric | v2pure `eval` (step 636) | v2pure `heldout` (step 636) | v2 `heldout` (step 636) | posev3 `heldout` (step 636) |
|---|---|---|---|---|
| audio cb0 CE | 1.5192 | 1.5239 | 1.5239 | 1.7462 |
| audio cb1-31 CE (depth decoder) | 4.0600 | 4.0226 | 4.0226 | 4.0112 |
| audio cb0 acc | 0.5420 | 0.5430 | 0.5430 | 0.4864 |
| END recall | 0.1529 | 0.1198 | 0.1200 | 0.1274 |
| pose cb0 CE | 1.6219 | 1.6305 | 1.6305 | 1.6305 |
| pose cb1-31 CE (depth decoder) | 3.6476 | 3.6700 | 3.6700 | 3.6700 |
| pose cb0 acc | 0.5731 | 0.5714 | 0.5714 | 0.5711 |

CE is in nats per token. `heldout` is the held-out Aloe eval split: the v2 release's eligible active-speaker clips of the held-out recordings (3,482 clips, never trained on), chosen with posev3's recording rule, so the clips are the same as the v2 and posev3 runs' held-out clips. `eval` is a small set of packed rows held back from the training data. All columns compare directly with v2, which used the same codes. Pose numbers also compare directly with posev3. Its audio numbers do not, because posev3's Mimi codes are a different encoding of the same audio.

## Input format

There is one stream, generation only: the active speaker's speech and pose, with no partner audio.

```
SOT  <full text tokens>  EOT  | slot 0 | slot 1 | ...
```

- The **whole transcript comes first**, before any speech.
- After it, speech slot `s` holds `PAD + sum(32 audio code embeddings) + sum(32 pose code embeddings)` of frame `s-3`. The hidden state at slot `s` predicts frame `s-2`.
- **Cold start:** text only. Every frame is generated.
- **Prefill:** the first N seconds of real audio and pose codes are teacher-forced, and the rest is generated. A prefill file is an `.npz` with `audio` `[32, F]` and `pose` `[32, F]` int codes. It can also carry `text`, `global_state` `[3]` (frame-0 root x, y and shoulder width, used for drawing the skeleton) and `orig_hw`. `examples/aloe_eval_2567.npz` is a held-out Aloe eval clip of 10.4 s. All of its fields come from the v2pure cache, which was built from the v2 release.
- Encode your own prefill audio the way the v2 release was encoded: `kyutai/mimi` @ `89091b3e`, 32 codebooks, with Mimi's native 250-step sliding-window attention. Do not use full attention over long clips.
- Generation stops at `--max_s` (total length including the prefill) or when audio cb0 samples END.

## Setup

```bash
huggingface-cli download InternalCan/sam-pose-aloe-v2pure-step636 --local-dir sam-pose   # or git clone
cd sam-pose
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

You need an HF token with access to `InternalCan`, because the pose tokenizer is a private dependency. These dependencies are fetched on first use, each pinned to a commit:

- Pose tokenizer weights: `InternalCan/pose-tokenizer-32cb-200k-noshift` @ `1d6ff4f0de89408cb054da618e309d4334596252`. Its code is vendored in `pose_tokenizer/`, from `canopy-labs-internal/pose-tokenizer` @ `1995b93`.
- Mimi codec: `kyutai/mimi` @ `89091b3e466eb6a9d11e537bf26b144f194978f7`.

## Inference defaults

| | |
|---|---|
| temperature | 1.0 on every stream and every codebook |
| audio | top-k 30, no CFG |
| pose | top-k 250 |
| repetition penalty | none |
| min-p / top-p | none |
| pose CFG (optional) | weight 3 on all 32 pose codebooks. The unconditional branch has the same text and pose history, with every audio input (prefill and generated) replaced by the Mimi codes of digital silence. |
| precision | bf16 everywhere (never FP8) |
| seed | 0 |

These are the defaults of the code in this repo. `infer.py` applies the optional pose CFG (weight 3) by default. To turn it off, pass `--pose_cfg 0` or run `pose_infer_fast.py`.

**A lower pose top-k (e.g. `--pose_topk 30`) gives more stable results**, with less varied pose.

## Run

From text alone (cold). The flags shown are the defaults:

```bash
python infer.py --text "Hey, so I finally tried that new ramen place downtown, and honestly it was way better than I expected." \
  --pose_topk 250 --topk 30 --out out/cold
```

From text plus a 2 s speech/pose prefill. The text defaults to the npz's transcript; pass `--text` to override it.

```bash
python infer.py --prefill_npz examples/aloe_eval_2567.npz --prefill 2.0 --pose_topk 250 --topk 30 --out out/prefill
```

Without pose CFG:

```bash
python infer.py --text "..." --pose_cfg 0 --out out/nocfg
```

Useful flags are `--max_s 10` (the default), `--seed 0`, `--device cuda`, `--no_video` and `--ckpt DIR` (the default is this repo). `pose_infer.py` is the plain HF reference loop, with no CFG and about 1 s/frame. `pose_infer_fast.py` is the same maths with CUDA graphs and no CFG. `infer.py` (`pose_infer_fast_cfg.py`) is the fast path plus the optional pose CFG.

## Outputs (in `--out`)

- `gen.wav`: 24 kHz mono speech.
- `codes.npz`: `audio` `[32, F]` and `pose` `[32, F]` codes (prefill included) and `prefill_frames`.
- `gen.mp4`: the decoded skeleton (25 fps) muxed with the audio. Prefilled frames are labelled "prefill (real)".
- `meta.json`: the settings and timings.

**Speed:** about 0.065 s per 80 ms frame on one H100 with pose CFG, which is about 1.2x faster than real time. Setup adds about 10 s with a warm inductor cache (`TORCHINDUCTOR_CACHE_DIR`), or about 75 s on the first compile. Loading the model takes about 30 s. One GPU with 40 GB or more of memory is enough.

## Files

- `model.safetensors`: the full fine-tuned weights (trunk, audio and pose heads, both depth decoders).
- `source.json`: the export's provenance.
- `config.json`: the trunk architecture.
- `configs/qwen3-0.6b-light/config.json`: the depth-decoder architecture.
- `tokenizer/`: the text tokenizer.
- `config.yaml`: the token ids and layout (code delay 3, acoustic delay 0, text lead 1). Its paths are relative to the repo.
- `model.py`, `sam_lib.py`, `pose_infer.py`, `pose_infer_fast.py`, `pose_infer_fast_cfg.py`, `infer.py`: the inference code.
- `pose_tokenizer/`: the vendored pose tokenizer code, used for decoding pose codes to skeletons.
- `examples/aloe_eval_2567.npz`: an example prefill clip from the v2 release.
