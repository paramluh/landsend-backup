# Audio conditioning (universal wide)

Frozen Mimi decode + trainable encoder + per-frame cross-attn. Universal wide
gathers and builds the modules in `models/pose/wan_universal.py` and
`models/pose/helpers.py`; encoder/adapter live in `models/pose/audio.py`.
Codec: Mimi (`kyutai/mimi`), 24 kHz / 12.5 Hz.

## On disk

One clip = one `*_mimi.npy` of codes `[codebooks, codes_T]` at 12.5 Hz. The
dataset always ships a fixed `[B, codebooks, 750]` tensor (one row per source
clip in the microbatch, zero-padded past `codes_T`; `AUDIO_CODE_QUANTUM` in
`dataloading/sources.py`). Video and pose are curtailed to
`clip_len = latent_t - 1`, covered by the real codes; no final code is repeated.
Clips longer than one quantum (`2 · clip_len > 750`) are dropped at manifest
time, so a row always covers its usable clip.

## Rate path (`_extract_audio_features`)

First `n` codebooks (`n = num_codebooks`; train draws
`n ~ U{min_codebooks..num_codebooks}` when `min_codebooks` is set, else fixed;
eval uses `num_codebooks`) decode to
512-d, upsample ×2 to 25 Hz:

```
codes [B, n, 750]
  → quantizer.decode   [B, 512, 750]
  → upsample ×2        [B, 512, 1500]  →  [B, 1500, 512]
```

RVQ always sums to 512, so upsample's conv shape is independent of `n`. Both
stages are frozen (`torch.no_grad`); Mimi's own decoder is deleted after the
encoder below is cloned from it.

## Encoder (`MimiAudioEncoder`)

```
[B, 1500, 512] → decoder_transformer[:num_layers] (causal+RoPE, fresh init)
               → Linear(512 → dim) → drop first `drop` → [B, 1500-drop, dim]
```

Architecture only, not Mimi's weights (`reset_parameters`, `MimiLayerScale`
reset to Mimi's init). `dim` is the DiT width (3840 on `wan_5b_wide`). All
tokens run through for causal context; `drop` then aligns token 0 with content
frame 1. The encoder runs once per source clip; windows index into it.

Two learned tokens: `null_token` `[1, 1, 1, dim]` (appended to every window)
and `no_audio_tokens` `[1, 1, K+1, dim]` (replaces a dropped window). Only
`no_audio_tokens` may be missing from an older checkpoint (`_load_module`).

## Gather (`_window_audio`)

Window row `b` reads source row `window_source[b]`. Present frame with clip
index `f ≥ 1` takes `K = cross_attention_codes` contiguous tokens
`{K(f-1) .. Kf-1}`; padding slots (`ofi = -1`) clamp to frame 1 and are masked
downstream. Output `[B, P, K+1, dim]`; `+1` is the null. Causal: `f` reads up
to `Kf-1`.

Training only: with probability `audio_dropout` per window, the whole
`[P, K+1, dim]` block is replaced by `no_audio_tokens`. The realised rate is
logged as `train/audio_dropped`.

## Injection (`FrameCrossAttentionBlock.forward_present`)

Only the `P` present frames query audio; sink and history never do. Each
present frame attends its `K` tokens plus null. The residual is multiplied by
`present_valid`, so padded present slots receive nothing. `AudioAdapter` places
a block at `range(injection_offset, num_dit_blocks, injection_interval)`;
`max_blocks` keeps the shallowest. After self-attn, before FFN.
Self-attention uses FA3; tiny-KV audio cross-attention uses FA2.

`[model.audio].initialization` is required: `zero` zeroes `out_proj` (no-op at
start), `random` is stock `nn.Linear`. Anything else raises. Live keys live in
the experiment toml; don't copy numbers from here.

## Params

`audio_encoder.*` and `audio_adapter.*` carry `original_name` for save/load and
optimizer groups (`[[optimizer.groups]]` matches on that prefix). The Mimi
front-end is frozen, lives outside the DDP module, and is not saved.

Pose is a different path: precomputed latents → patch-embed → **added** to the
token stream for all 1+H+P frames; sink, history padding, and startup padding
are masked (`ofi ≥ 1`).
