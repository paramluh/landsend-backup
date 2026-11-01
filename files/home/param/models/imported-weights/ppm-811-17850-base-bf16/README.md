---
library_name: pytorch
tags:
- multimodal
- speech
- pose
- internal
---

# Exact PPM 8,1,1 step 17850 base, bf16 model-only export

This private preservation package contains the pre-fine-tune PPM checkpoint that seeds the Xabi/v2pure production fine-tune, job 27757. It is the successful 8,1,1 delivery-3 continuation, run `ppm-v1-j27653`, step **17850**. It is not the Xabi export-200 fine-tune, the Aloe export-357 fine-tunes, a Voxtral model, or the later pose5 checkpoint.

The exact source was `/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850`, an existing bf16 conversion of `hf://InternalCan/ppm-v1-c@c9b2213751cb3f2ede1dce2c5b9ed275b5cc3ca2/dcp-17850`. It was copied unchanged through the requesting user's own directory. Every copied file passed SHA256 comparison, and the original file-size/mtime inventory remained unchanged. No original source files were modified. The original metadata and source paths are preserved verbatim.

## Weights and architecture

`weights/` is the original `ppm-local-bf16-shards-v1` loader format: `rank0.pt` through `rank127.pt`, `rep.pt`, `meta.json`, and `COMPLETE` (17850). These 131 files total 88,413,726,150 bytes. They contain model weights only, with no optimizer state.

The model has 2,150 unique parameter tensors: 1,522 sharded tensors and 628 replicated decoder tensors. Architecture: a 64-layer, width-5120 backbone; two 36-layer, width-2560 speech/pose towers; and two 28-layer, width-1024 speech/pose depth decoders. The source sharding world is 128, the projection rank is 2048, tower vocabulary is 151682, and code delay is 2.

The learned D33 `decp_bb` weights are present and preserved. The base's recorded `decp_bb_off=1` means this link was disabled during 8,1,1 continuation; it does not mean its weights are absent or zero. The historical Xabi fine-tune deliberately builds the link and reactivates it with alpha 1.0 and learning rate 1e-5. No D32 pose head, S2P embedding, or G0COND projection is present in this base. No architecture flags or parameter values were changed during preservation.

## Using the historical fine-tuning loader

Set the original `train_aloe_pose.py` `load_init` path (`QQA_INIT`) to the package's `weights/` directory. That loader reads the source dim-0 shards and reshards them into the target distributed world. All source/target row coverage has been checked for a 56-rank target (seven nodes with eight ranks each); divisibility of 128 by 56 is not required. Each node may hold the full export; no checkpoint conversion is needed for this loader.

Runtime path mapping:

- `M.QWEN32`: `configs/Qwen3-32B/` (the exact original architecture config and Qwen tokenizer assets).
- `M.QWEN4B_CFG`: `configs/Qwen3-4B/` (the exact original tower architecture config).
- The historical depth-decoder config is constructed in code. `configs/Qwen3-0.6B/config.json` is included for provenance, not as a change to that implementation.

Build the model with `init=False` when loading these complete weights. Original Qwen pretrained weight files, Mimi encoder weights, and pose-tokenizer weights are not included or required to train from precomputed media codes. The custom model uses the original loader; this package is not a standard Transformers `from_pretrained` directory.

## Verification and provenance

`provenance/source-weight-manifest.json` records source paths, lengths and SHA256 hashes. `weight-validation.json` records every tensor header/dtype, architecture keys, shard layout, target-world-56 row coverage, and finite/nonzero learned D33 content with raw-tensor hashes. `hf-weight-verification.json` records comparison of every uploaded weight object against the source manifest. The original model/trainer/run configuration and available historical conversion checks are preserved in `provenance/original/`.

`files.sha256` covers every weight, config, provenance file and this README. `COMPLETE.json` records its digest and successful verification. The manifest excludes itself and that completion record to avoid recursive hashing; HF-managed `.gitattributes` and local `.cache/` download metadata are outside its scope. Verify a copy with `sha256sum --quiet -c files.sha256` from the package root, and pin the final model-repository commit when downloading.
