# Production dummy step

Ceiling for one sequential cached forward: Initial + 30 blocks + Final, no audio encoder. Dummy weights, dummy KV.

```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python production/benchmark.py --sweep                       # wan_5b_wide, 120 tokens
CUDA_VISIBLE_DEVICES=0 .venv/bin/python production/benchmark.py --tokens 480 --dim 5120 --kv 1  # 2x H/W, 10B model
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 production/benchmark.py --tokens 480 --dim 5120 --kv 1
```

`--tokens 480` = 2× H and 2× W (60×32 latent). `--dim 5120` = ffn 23808, 40 heads × 128. `--kv 1` adds the `k`/`v` projections of the moving frame (the dummy as-is has every key precomputed). `WORLD_SIZE > 1` = tensor parallel. `--profile` buckets device kernel time of the fastest config.

## What is timed

One frame, B=1. Self-attn KV is cached: Q tokens against 8·Q keys (sink + 6 history + self). Eight audio blocks (offset 4, interval 3) do Q×5 cross-attn; the Mimi encoder is skipped (run once in real production, not per step). Every config: eager, eager + CUDA graph, whole-step `torch.compile`, compile + CUDA graph. `--sweep` adds `reduce-overhead`, `max-autotune`, and per-block compile (single GPU only).

## Method

Rounds 4–7 (`docs/benchmarking/`): exclusive H100 80 GB HBM3, warmed CUDA-event means (40 iters), `torch.inference_mode`, `num_threads=1`, bf16. Kernel buckets from `torch.profiler` on the graphed step. Analytical FLOPs: GEMMs on the Q tokens + `4·Q·KV·D` per attention.

## 1. wan_5b_wide, 120 tokens (dim 3840, 5.59B dummy params, 1.31 TFLOP/step)

| setup | ms/step |
|---|---:|
| eager (first dummy) | 15.85 |
| eager (this file) | 12.07 |
| eager + CUDA graph | 7.35 |
| compile whole | 6.23 |
| **compile whole + CUDA graph** | **5.54** |
| compile `reduce-overhead` | 5.71 |
| compile `max-autotune` | 5.78 |
| compile per-block | 11.90 |
| compile per-block + CUDA graph | 5.99 |
| compile per-block `max-autotune-no-cudagraphs` | 13.12 |
| compile per-block autotune + CUDA graph | 6.10 |

Re-run 2026-09-19 with the current script: 5.535 ms, every compiled/graphed row within 2%. Autotune does not beat cuBLAS on these GEMMs; one graph over the full step beats per-block compile.

Kernel buckets of the graphed step: GEMM 4.46 ms, FlashAttention 0.66, fused elementwise (AdaLN, LN, GELU, residual) 0.48 → 237 TFLOP/s, 24% of dense bf16 peak.

## 2. 480 tokens (2× H, 2× W), dim 3840

| setup | 120 tokens (KV 960) | 480 tokens (KV 3840) | ratio |
|---|---:|---:|---:|
| eager | 12.66 | 16.29 | 1.29× |
| eager + CUDA graph | 7.35 | 15.03 | 2.04× |
| compile whole | 6.32 | 12.22 | 1.93× |
| **compile whole + CUDA graph** | **5.54** | **11.78** | **2.13×** |

Work 1.31 → 5.89 TFLOP/step (GEMM 5.04, self-attn 0.85: both Q and KV grow 4×). Achieved 237 → 500 TFLOP/s (51%). Kernel buckets at 480: GEMM 9.00, attention 1.79, elementwise 1.20. Per token 46 → 24.5 µs.

Why not 4×: a GEMM `[M×K]@[K×N]` has intensity M FLOP per weight byte, so the ridge point 989 TFLOP/s ÷ 3.35 TB/s ≈ 295 FLOP/byte sits at **M ≈ 300 tokens**. At M=120 the step is bound by streaming 11.2 GB of weights + 0.4 GB of cache (floor 3.47 ms; measured 1.6× that). At M=480 it is compute-bound (floor 5.96 ms; measured 2.0× that, since 480 rows still fill only one wave of tiles on 132 SMs).

| | 120 | 480 |
|---|---:|---:|
| weights + cache read | 11.6 GB → 3.5 ms | 13.0 GB → 3.9 ms |
| GEMM+attn at 989 TFLOP/s | 1.32 ms | 5.96 ms |
| binding floor | **bandwidth 3.5 ms** | **compute 6.0 ms** |
| measured | 5.54 | 11.78 |

This supersedes the earlier "why not ~2 ms" framing (compute floor 1.62 ms at the round-7 GEMM rate): at 120 tokens the weight stream, not the GEMM rate, is the binding floor.

## 3. dim 5120 (≈10B) at 480 tokens

Parameter count at fixed depth is ≈ `30·(4 + 2·14/3)·d² + 8·4·d² (audio) + 7·d² (time)` ≈ 380·d² → 5.6B at 3840, 9.9B at 5120 (dummy, no `k`/`v`), 11.5B with them.

| setup | dim 3840 | dim 5120 | ratio |
|---|---:|---:|---:|
| eager | 16.29 | 23.15 | 1.42× |
| eager + CUDA graph | 15.03 | 22.17 | 1.48× |
| compile whole | 12.22 | 19.06 | 1.56× |
| **compile whole + CUDA graph** | **11.78** | **18.72** | **1.59×** |

Work 5.89 → 10.07 TFLOP/step (1.71×) for 1.59× the time: wider GEMMs tile better, 538 TFLOP/s (54%). Kernel buckets: GEMM 13.58, attention 3.18, elementwise 1.48. Weights + cache 22.2 GB/step = 1.19 TB/s effective, a third of HBM: firmly compute-bound (floor 10.2 ms; measured 1.84×). Peak memory 37 GiB.

### Real cached step: `--kv 1`

The dummy has no `k`/`v` projections; a real cached step projects the moving frame's k/v each layer (+2·D² per block, +1.6B params at 5120) and appends them to the cache.

| dim, tokens | dummy (`--kv 0`) | with k/v (`--kv 1`) | params |
|---|---:|---:|---:|
| 3840, 480 | 11.78 | 14.47 | 5.59B → 6.48B |
| 5120, 480 | 18.72 | 22.65 | 9.92B → 11.49B |

At 5120 the +3.9 ms is ≈ 2.2 ms of GEMM (60 extra 480×5120×5120, ~690 TFLOP/s) + ≈ 1.5 ms of `torch.cat` materialising the full K and V per layer (2.4 GB written + read). A real cache writes the new k/v in place and does not pay the second part, so ≈ 21 ms is the fair single-GPU number for a real 480-token step of the 10B model.

## 4. Tensor parallel over 2 GPUs (dim 5120, 480 tokens)

Goal is TTFV, so batching is useless; the only way to use a second GPU on one sequential step is to split the step. TP splits heads (40 → 20 per GPU, K/V cache split with them) and FFN (23808 → 11904); every block does an all-reduce of the `[480×5120]` bf16 partial residual (4.9 MB) after `o` and after `w2`, audio blocks one more after `out_proj`: **68 all-reduces per step**, plus 8 tiny fp32 ones for the audio qk-norm, which normalises over all heads jointly and therefore needs the sum of squares from both ranks to stay exact. Biases of reduced projections live on rank 0 only. GPUs on this node are NV18 all-to-all; the pair used was on one NUMA node.

All-reduce kernel choice, measured on the exact message, graphed, 2 GPUs (`torch 2.11 / NCCL 2.28.9`):

| kernel | µs / all-reduce |
|---|---:|
| NCCL `all_reduce` | 40.5 |
| symm-mem one-shot (copy + kernel) | 38.8 |
| symm-mem `one_shot_all_reduce_copy` (fused) | 41.8 |
| **symm-mem two-shot (copy + in-place)** | **27.0** |
| NVLS `multimem_one_shot` | 48.3 |
| NVLS `multimem_all_reduce_` | 34.9 |
| copy into the symmetric buffer alone | 3.8 |

Wire time for 4.9 MB at 450 GB/s per direction is ~11 µs; two-shot is the closest. `two_shot_all_reduce_` is in-place and untraceable by Dynamo (a graph break at every call site, which ran at 14.9 ms), so the script wraps it in a custom op that returns the call site's own symmetric buffer; the whole step then compiles into one graph.

| setup (compile whole + CUDA graph) | ms/step | vs 1 GPU |
|---|---:|---:|
| 1 GPU | 18.72 | |
| TP=2, NCCL all-reduce | 13.30 | 1.41× |
| TP=2, one-shot symm-mem | 12.82 | 1.46× |
| TP=2, two-shot, graph-broken (in-place op) | 14.86 | 1.26× |
| **TP=2, two-shot as custom op** | **12.06** | **1.55×** |
| TP=2, two-shot + Inductor max-autotune | 11.67 | 1.60× |

NCCL / one-shot / autotune rows were measured with the approximate (per-rank) qk-norm; the exact norm costs ≈ 0.1 ms (11.90 → 12.01). Peak memory 19 GiB/GPU.

Kernel buckets of the 12.06 ms step per GPU: GEMM 7.01 (from 13.58: 1.94×), attention 1.55 (from 3.18), collectives 1.96 (76 kernels, ~26 µs each), elementwise 1.25 (LayerNorm/AdaLN run on the full width on both ranks, so they do not halve), copies into symmetric buffers 0.26. Ideal half is 9.36 ms; the 2.7 ms gap is 1.96 collective + 0.26 copy + ~0.5 unsharded elementwise.

With the real k/v projections:

| dim 5120, 480 tokens, `--kv 1` | ms/step | vs 1 GPU |
|---|---:|---:|
| 1 GPU | 22.65 | |
| **TP=2, two-shot** | **13.94** | **1.62×** |
| TP=2, two-shot + max-autotune + coordinate descent | 14.09 | noise |

Correctness of the sharding was checked at small dims against an fp32 unsharded reference with identical weights: TP=2 bf16 rel-L2 error 1.05e-2, single-GPU bf16 1.30e-2, graph replay identical to the eager compiled result (both `--kv` settings).

What is left on 2 GPUs: overlapping the 68 collectives with chunked GEMMs (async TP), at most ~1 ms and uncertain at M=240 per chunk; bf16 floor ≈ 10.5–11 ms (`--kv 0`), ≈ 12.5–13 ms (`--kv 1`). For the real model note that its self-attn qk-norm is also joint over heads, so exact TP adds ~30 more latency-bound all-reduces (≈ +0.3–0.5 ms) unless the norm is made per-head. For 30 denoising steps at 480 tokens the 10B model's first frame costs ≈ 420 ms of model time on 2 GPUs vs ≈ 680 ms on one.

## Summary

| d | tokens | k/v proj | GPUs | params | TFLOP/step | ms/step | TFLOP/s |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 3840 | 120 | no | 1 | 5.6B | 1.31 | 5.54 | 237 |
| 3840 | 480 | no | 1 | 5.6B | 5.89 | 11.78 | 500 |
| 3840 | 480 | yes | 1 | 6.5B | 6.74 | 14.47 | 466 |
| 5120 | 480 | no | 1 | 9.9B | 10.07 | 18.72 | 538 |
| 5120 | 480 | yes | 1 | 11.5B | 11.58 | 22.65 | 511 |
| 5120 | 480 | no | 2 (TP) | 9.9B | 10.07 | 12.06 | 835 agg. |
| 5120 | 480 | yes | 2 (TP) | 11.5B | 11.58 | 13.94 | 831 agg. |
