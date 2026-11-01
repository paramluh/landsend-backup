---
pretty_name: Xabi v2pure production cache
tags:
- multimodal
- pose
- speech
- internal
---

# Xabi v2pure production cache

This private dataset preserves the exact compact `sam-cache-active-v2pure` cache used to build the PPM Xabi/v2pure fine-tuning stores. The production run was `ppm-aloe-v2pure-main-17850-j27757`: the 8,1,1 PPM checkpoint at step 17850, followed by 200 fine-tuning steps, producing export 200. Its final model is `InternalCan/ppm-aloe-v2pure-main-17850-final` at commit `7beed89de7ca01803cf84ff2c4bff928322403c7`.

The original cache was copied without transformation from `/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure` into the requesting user's own directory. Every source and destination file passed SHA256 comparison; the source size and modification-time inventory remained unchanged during copying. Original source files were never modified. Original provenance paths in copied documents describe the historical environment; they are not paths to write to.

## Contents and split

`cache/` contains 7,691 original files totaling 12,416,427,207 bytes, including 960 training shards and one evaluation shard. Every shard has the original eight files. The cache derives from the active-speaker selection of `gs://aloe-processed-data/pose-speech-vad/releases/v2/`.

| Split | Items | Frames at 12.5 Hz | Recordings |
|---|---:|---:|---:|
| Train | 367,496 | 92,132,615 | 12,384 |
| Eval | 3,482 | 868,932 | 113 |

The cache retains one one-frame item in each split. The historical PPM row reader excludes those two items, yielding 367,495 training clips and 3,481 evaluation clips. PPM packs the training clips into 12,791 rows and completes one epoch in 200 steps of up to 64 rows. Items are split by recording using the preserved historical hash rule; train and evaluation recordings do not overlap.

Media codes are unchanged local integer IDs in [0, 2047], with 32 codebooks for Mimi speech and 32 for pose at 12.5 Hz. No decoding, normalization change, retokenization, or new filtering was performed. The cache's `text.npy` is the historical SAM text representation, not PPM's timed word lane. Reproduce PPM text from the cached timed `items.parquet` words and the preserved original conversion/tokenizer rules under `provenance/original/`.

Each shard contains `codes.npy` and `pose.npy` (int16, total frames by 32), `codes_off.npy` (item frame offsets), `text.npy` (int32 SAM token IDs), `text_off.npy` (item text offsets), `audio_id.txt` (ordered item keys), `items.parquet`, and `DONE.json`. Timed words are JSON lists of clip-local [start seconds, end seconds, text], with times rounded to three decimals. Items also retain recording/speaker/segment identity, active-speaker flags, source timing, crossbleed and loudness status.

Original encoding pins are pose tokenizer `InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0de89408cb054da618e309d4334596252`, pose confidence processing `confidence-hold-03-frame0-init-v2`, pose coordinate transform `frame0-root-frame0-horizontal-shoulder-width-v1`, and speech encoding `mimi-local-attention-v1`. The preserved extractor pins the Mimi checkpoint revision to `89091b3e466eb6a9d11e537bf26b144f194978f7` and its attention policy to `checkpoint_sliding_window_v1`. A pose token is anchored to the end of its two-frame group, at 0.08-second token stride.

Cached geometry preserves each item's selected 55-joint frame-zero pose as flattened 55-by-3 values in (normalized y, normalized x, detector confidence) order, frame-zero root and shoulder width, original canvas height/width, geometry-valid flag, and `global_state0`. The compact cache does not contain the full 133-joint frame-zero pose or a per-frame global-state sequence.

`provenance/original/` preserves the original cache builders/extraction/verification code; PPM store builder and validator; historical PLAN, STORES and per-store row metadata/pose indices; the production run environment; Qwen3-32B tokenizer assets; the original row converter and silence codes. These are provenance, not a newly edited training recipe. No model weights are included.

## Verification and reuse

`provenance/source-cache-manifest.json` and `source-cache.sha256` record source hashes. `provenance/cache-validation.json` records checks of all 961 shards, all media-code ranges and array shapes, split/membership invariants, geometry, and exact mapping of all 370,978 items to the historical PPM rows. `provenance/hf-cache-verification.json` records the uploaded cache revision and checks against HF object hashes.

`files.sha256` covers every cache file, every provenance file, and this README. `COMPLETE.json` records its digest and successful validation. The hash manifest excludes itself and COMPLETE to avoid recursive hashes. HF-managed `.gitattributes` and local download metadata under `.cache/` are outside the package manifest. Run `sha256sum --quiet -c files.sha256` from the package root to verify a copy. Pin the final dataset commit when downloading.

The preservation package contains the original selected data and its provenance, not the source videos/audio or the padded PPM array stores. No new filtering, train/eval re-splitting, or training-recipe change is implied.
