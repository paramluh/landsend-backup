# Benchmarking rounds 1–6: consolidated record

This is a lossless technical condensation of `round-1.md` through `round-6.md`.
It preserves the measurements, implementation advice, correctness checks, rejected paths, caveats, and reproduction notes while removing repeated narrative.

## Round 1 — base `wan_5b_vidcon` training profile

### Setup and result
- 8×H100, bf16; `wan22_5b_pose_aloe.toml`; audio+pose, text off, microbatch 12, `compile_blocks=true`, `pipeline_stages=1`.
- DiT shape is `x≈(B,1440,3072)`: 12 frames × 120 tokens through 30 layers. FFN + q/k/v/o + cross/audio are ≈60% of CUDA, and activation recompute roughly doubles forward GEMMs.
- RoPE was 9.5% before precomputation; Flex-attention and offline pose-latent precompute were not bottlenecks.
- Wall step was ~560 ms larger than `train_batch`; DeepSpeed `iter time`/`samples/sec` report only `train_batch`, hiding the gap.

### Single-variable runs

All runs: 8×H100, mb=12, global batch 96, audio+pose, text off, 4.2B parameters.

| config | step | peak memory/GPU |
|---|---:|---:|
| compile OFF, skip=4, ZeRO-0 | 1.92 s | 73.8 GB |
| compile ON, skip=4, ZeRO-0 | 1.52 s | 57.4 GB |
| compile ON, skip=0, ZeRO-0 | 1.55 s | 49.5 GB |
| compile ON, skip=0, ZeRO-1 | 1.43 s | 29.5 GB |

- At skip=4, `compile_blocks` is the dominant lever: −21% step and −16 GB by fusing compute and reducing activations; `activation_checkpoint_skip` 4→0 gives +2% step and −8 GB through recompute.
- At skip=0, ZeRO-1 gives −20 GB/GPU (−40%) and the source records approximately −8% step. Replicated optimi bf16 states are recorded as approximately 25 GB; fp32 master/m/v sharded `/8` are ~8 GB/rank and make each optimizer step ~8× smaller.
- Optimizer-state memory scales with DP size (`/8` on one node, `/16` on two). This DeepSpeed path uses bucketed gradient `all_reduce` plus parameter `all_gather`, not literal reduce-scatter.

### CPU thread oversubscription: the largest wall-time win
- The continuation-window `_build_window` runs in main-process `prepare_inputs` before `train_batch`. PyTorch defaults `num_threads` to physical cores, giving 8×64=512 threads; ~1 ms per-sample copies became ~560 ms through OpenMP barriers and GPUs idled ~47%.
- This is a phase transition above the physical-core count, not linear slowdown. `/mnt/somfs` fetch was ~0 ms because prefetch worked; I/O was not the cause.
- Apply after distributed initialization in every `learn_*.py`:

```python
torch.set_num_threads(cpu_count // local_size)
```

- This is byte-identical because only CPU thread count changes. Measured with mb=8, skip=20, ZeRO-1:

| state | wall step | GPU utilization |
|---|---:|---:|
| before | ~1.62 s | ~53% |
| after | ~0.88 s | ~90%+ |

- End-to-end speedup is ~1.8×; `train_batch` becomes ≈96% of wall step. `pin_memory=True`, configurable `num_dataloader_workers`, and `prefetch_factor` in `dataloading/dataloader.py` affect only margins.

### Measurement advice
- DeepSpeed `iter time`/`samples/sec` are `train_batch`, not wall; use step-log timestamp deltas or `nvidia-smi dmon -s u`.
- Temporary memory/op/loop profilers were removed. Re-add `torch.profiler` around `train_batch`; `record_function` may wrap eager call sites only, never compiled regions. Round 2 records the code details.

## Round 2 — compiled transformer blocks and attention/RoPE changes

### Stacking changes
Three numerically checked changes were made to `wan_5b_vidcon`: opt-in per-layer `torch.compile` via `compile_blocks`; cached RoPE multipliers; and two-call self-attention replacing varlen attention plus KV duplication. The last two are always active.

### `TransformerLayer` split

- `forward(inputs)` stays eager for offloader plumbing, `make_contiguous`, and synchronizing `freqs_i`/history count `H`, then calls `self._run(...)`.
- `_forward_impl(...)` contains modulation, norms, self/cross/audio attention, FFN, residuals, and is `torch.compile(dynamic=False)`d when enabled.
- `_split_self_attention` uses history→history and present→full-window `flash_attn_func` calls, removing KV `cat`, `cu_seqlens`, and duplication.
- `_fa_func = torch.compiler.disable(flash_attn_func)` makes FlashAttention opaque while Inductor fuses surrounding qkv/RoPE/casts; cross/audio kernels in `models/wan/model.py` and `audio.py` are likewise opaque.
- `_build_rope_freqs_i` caches `[1440,1,64]` by token count `s`; `_apply_rope` replaces Python looping, rebuilding, and `grid_sizes.tolist()` sync with one batched complex multiply.

| compiled/fused by Inductor | eager or opaque |
|---|---|
| modulation, norms, residuals, FFN | FlashAttention kernels |
| self-attn qkv, RoPE, casts, o-proj | `InitialLayer`, `FinalLayer`, offloader, AdamW |
| cross/audio-attn linears | `freqs_i` and `H` computation in `forward` |

### Correctness measurements

- Two-call attention equals varlen+KV duplication bitwise (`max|Δ|=0`; deterministic FlashAttention, identical key sets), and cached RoPE equals `rope_apply` bitwise in bf16 and fp32.
- Compiled versus eager differs only by rounding: bf16 gradient relative error ~2.6% falls to 0.12% in fp32; no NaN/Inf.
- `compile_blocks=false` is bit-identical to pre-change code.

### Use and hazards

```toml
compile_blocks = true
```

- Default is off and byte-identical. Compilation needs `ninja` plus matching `nvcc`; the first step compiles 30 blocks, and `dynamic=False` recompiles for new shapes.
- Critical hazard: caching `freqs_i` only by `s` is silently wrong for another grid with the same `s` or `frame_offset != 0`.
- Current assumption: one fixed batch-uniform `[12,15,8]` grid at offset 0. Robustness requires key `(F,H,W,frame_offset)`, a uniform-batch assertion, and `rope_apply` fallback.

## Round 3 — resolution-model profile and scaling

### Setup and token regimes
- 8×H100, bf16, ZeRO-1, `compile_blocks=true`; `wan22_5b_pose_resolution.toml`, `combine`, audio+pose, text off; 30 layers, dim 3072, 24 heads × 128.

| regime | pixels (h×w) | grid (F×H×W) | tokens `s` | history `H` |
|---|---:|---:|---:|---:|
| low resolution | 480×256 | 12×15×8 | 1440 | 960 |
| high resolution | 960×512 | 12×30×16 | 5760 | 3840 |

- `combine` runs low then high with GA=2; high has exactly 4× tokens from 4× spatial size.
- Profiling used `activation_checkpoint_skip=0`, mb=4. Production multinode uses skip=10, but one-node ZeRO-1 has ~4 GB/GPU less optimizer headroom and OOMs.
- Skip=0 recomputes all 30 blocks, inflating forward casts/norms/residuals/GEMMs without changing composition.

### CUDA profile

Compiled `combine` measured ~12.9 s self-CUDA over 5 steps. Buckets count device kernels only; `aten::addmm`-style launcher rows were excluded to avoid double-counting `nvjet`/Triton kernels.

| bucket | CUDA share |
|---|---:|
| GEMM, `nvjet_*` forward `addmm` + backward `mm` | 43% |
| fused elementwise: norms, casts, residual add/mul, GELU, RoPE | 28% |
| FlashAttention forward + backward | 23% |
| NCCL communication | ~2.3% |
| other: memcpy and reductions | ~1.5% |

- The model is compute-bound: GEMM + FlashAttention is 66%, all irreducible work including FFN `3072×14336`, qkvo, and `O(s²)` attention.
- The 28% elementwise bucket is maximally fused and memory-bound: fp32 norms ~8–9%, casts ~6–7%, residual add/mul ~7%, GELU ~2%, modulation ~2.4%, RoPE ~1%.
- GEMM is 43%, not 21.6%: forward `aten::addmm` and backward `aten::mm` are distinct operators dispatching the same `nvjet_*` family.

### RoPE and casts are not useful optimization targets

Standalone tests used real shapes, ablated RoPE inside compiled attention, and retained FlashAttention as an opaque leaf.

| measurement | low resolution | high resolution |
|---|---:|---:|
| standalone RoPE eager → compiled | 0.29→0.26 ms | 1.11→0.96 ms |
| FlashAttention forward, 2 calls | 0.28 ms | 3.34 ms |
| block fwd+bwd with-RoPE minus no-RoPE | 0.38 ms | 1.35 ms |

- Across 30 blocks and both passes, RoPE is ~52 ms/step (≈2%). Its in-graph marginal is below standalone compiled RoPE because Inductor folds complex multiply/fp32 cast into qk normalization; `triton_..._mean_pow_rsqrt_view_as_complex` is 0.8%.
- Standalone compilation gives only 1.15×; a custom RoPE/cast kernel would win nothing and is not worthwhile.

### Verified scaling

- Per-block theory: GEMM 4×, attention 16× because both `H` and `s` scale 4×, elementwise 4×, NCCL fixed. Isolation used `batch_mode='random'`, `low_ratio=1.0/0.0`, mb=4, skip=0, GA=1, over 5 self-CUDA steps.

| bucket | low (ms) | high (ms) | measured high/low | theory |
|---|---:|---:|---:|---:|
| GEMM | 1166 | 4582 | 3.93× | 4.0× |
| FlashAttention | 215 | 2787 | 12.97× | 16.0× |
| elementwise | 949 | 2922 | 3.08× | 4.0× |
| NCCL, de-duplicated | ~302 | ~323 | 1.07× | 1.0× fixed |
| compute-only: GEMM+Flash+elementwise | 2330 | 10290 | 4.42× | 4.62× |
| total self-CUDA/step | 618 | 2227 | 3.60× | — |

- GEMM 3.93× matches 4×; NCCL 1.07× is fixed, hence 19.5% of low step but ~3% of high.
- Compute-only 4.42× matches 4.62× theory; single-GPU full-block measured 4.85×.
- Flash 12.97× versus 16× and elementwise 3.08× versus 4× reflect low-resolution launch/tail underutilization; GEMMs saturate at both sizes.
- High resolution is ~82% of `combine` from 4× tokens. Attention's 16× scaling is real but Flash is only ~7% of low budget; nothing is pathological.

### Practical conclusion and reproduction

- Remaining levers: raise `skip` as ZeRO frees memory, or lower the GEMM/Flash floor with precision/quantization; RoPE/cast tuning is exhausted.
- Random mode/`low_ratio` were removed. To isolate a regime, process only low or high in `prepare_inputs`.
- Profile `train_batch`, bucket `key_averages()` by device-kernel self-CUDA, exclude self-CPU rows, and use `record_function` for component splits only with `compile_blocks=false` because in-graph markers break Dynamo. Temporary instrumentation was removed.

## Round 4 — FlashAttention-3, SDPA failure, and GC stalls

### Setup
- 8×H100 80 GB, Hopper sm_90, CUDA 13.0, torch 2.11.0+cu130; `resolution_5b_aloe.toml`, `wan_5b_resolution_variable`, `combine`, `lowres_mode='GT'`, audio+pose, text off.
- mb=4, GA=2, history 8, present 4, ZeRO-1, compile on; non-causal bf16 attention with 24 heads × 128.

| call | low resolution, s=1440/H=960 | high resolution, s=5760/H=3840 |
|---|---|---|
| history→history `out_h` | q=k=v `[4,960,24,128]` | q=k=v `[4,3840,24,128]` |
| present→full `out_p` | q `[4,480]`, kv `[4,1440]` | q `[4,1920]`, kv `[4,5760]` |

- Production adopted FA3 self-attention for all four pose models and `gc.freeze()` in `learn_resolution.py`; cuDNN/SDPA and selective checkpointing were rejected after measurement.

### FlashAttention-3

FA3 is a separate prebuilt-wheel package from FA2:

```bash
pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130 --no-deps
```

- It imports as `flash_attn_interface` alongside FA2's `flash_attn`; its function returns one tensor, not `(out,lse)`, removes `dropout_p`, and is a drop-in for the model's `_fa_func`.
- Against fp32 over identical bf16 inputs at production shapes, including forward and dq/dk/dv, FA3 max-relative error was ~2–4e-3 and identical to FA2 row-for-row.
- CUDA-event forward+backward speedup versus FA2, median of 100:

| shape | FA3 | cuDNN-SDPA | torch-flash, effectively FA2 |
|---|---:|---:|---:|
| low `out_h` | 1.39× | 1.50× | 0.99× |
| low `out_p` | 1.30× | 1.46× | 0.97× |
| high `out_h` | 1.74× | 1.61× | 0.96× |
| high `out_p` | 1.76× | 1.62× | 0.97× |
| low audio, KV=5 | 0.89× | 1.30× | 1.22× |
| high audio, KV=5 | 1.08× | 1.09× | 1.00× |

- FA3 wins dominant high-resolution self-attention at ~1.75× but low-resolution KV=5 audio regresses to 0.89×, so keep `audio.py` on FA2.
- End-to-end 8×H100 `train_batch` median, compile on, skip=0:

| run | FA2 | FA3 | change |
|---|---:|---:|---:|
| combine step | 2.678 s | 2.426 s | −9.4% |
| high-only step | 2.214 s | 1.979 s | −10.6% |

- High-only −10.6% cross-checks 27% Flash share × `1−1/1.73=42%`, predicting −11.4%. Loss matched and training stayed clean.
- A ~1.7× kernel gain becomes ~10% end-to-end because GEMM is now ~53% and elementwise is unchanged.

### Why cuDNN/SDPA was rejected

- Isolated cuDNN-SDPA was faster for low-resolution self/audio and measured ~2.10 s end-to-end versus FA3's 2.43 s, but every compiled `F.scaled_dot_product_attention` route was finite at step 1 then NaN at step 2 from corrupted gradients.
- Failures covered cuDNN forced through `sdpa_kernel` in an eager `compiler.disable` leaf, compiled global forcing, dispatcher choice, and cuDNN-excluded Flash/memory-efficient fallback.
- Contiguous q/k/v and output still failed and cost +0.15 s. cuDNN, aten-Flash, and memory-efficient all failed while package FA2 stayed finite; both traced and leaf paths failed.
- fp32 SDPA was stable, identifying bf16 backward precision: compile fusion yields Inf, while package FA2/FA3 retain fp32 backward accumulation.
- With no grad scaler and global-norm clipping, one Inf makes total norm NaN and all parameters NaN at the step-2 optimizer update.
- fp32 SDPA measured ~2.62 s, about FA2 and slower than FA3; only the broken bf16 path was faster. FA3 is the only tested kernel both fast and bf16-robust in backward.

### GC stalls and the shipped fix

- Production's 0.3–0.4 s stalls were not I/O: prepare/fetch stayed ~50–70 ms with the thread fix.
- They were generation-2 GC pauses of ~280–370 ms every ~16 steps landing in `t_train` mid-`train_batch`, draining the GPU while scanning the huge static graph to collect only ~42 objects. Single-node A/B over 80 steps:

| GC mode | `t_train` median | max | steps >2.9 s | generation-2 pause |
|---|---:|---:|---:|---:|
| default | 2.679 s | 3.118 s | 5 | 264–369 ms |
| `gc.disable()` | 2.671 s | 2.739 s | 0 | — |
| `gc.freeze()` | 2.675 s | 2.738 s | 0 | 20–29 ms |

- Call `gc.freeze()` after setup and before training: it excludes the static model/optimizer graph while still collecting new cycles, makes gen-2 ~13× cheaper, and removes stalls.
- Average single-rank effect is ~1.3%, but at DP=48 asynchronous rank collections block subsequent all-reduces, explaining why stalls appeared frequent in production.

### Selective activation checkpointing

- The tested policy explicitly stores FFN and recomputes only attention—the cheap part—giving theoretical ~2.5 versus ~2.0 GEMM-units saved/GB over binary `skip`; it was finite, exact, and loss-matched.
- At 12 SAC blocks GEMM fell 6303→5875 ms (−6.8%), but storing attention-output→FFN costs ~2.7 GB/block high-res and one node OOMed beyond ~14.
- Iteration improved only ~2.3% and barely scaled from 8→12 blocks due to backward overlap, so SAC was reverted; use multinode ZeRO headroom to raise ordinary `skip`.

### Post-FA3 profile

`combine`, skip=0, device-kernel self-CUDA:

| bucket | FA3 + GC | round-3 FA2 |
|---|---:|---:|
| GEMM, `nvjet_sm90` | ~53% | 43% |
| elementwise/fused | ~25% | 28% |
| attention, cutlass FlashAttn Sm90 | ~14% | 23% |
| NCCL | ~6% | ~2% |
| memcpy/other | ~2% | ~3% |

- FA3 moves attention 23→14%; one FFN GEMM is the largest kernel at ~18%.
- Remaining levers are higher `skip`, subject to the same memory constraint that killed SAC, and fp8 GEMMs—the largest ceiling but accuracy-sensitive; elementwise/attention/NCCL are exhausted here.

### Methodology

- Never benchmark on contended GPUs: the live run pins all 8. Establish parity against fp32 over identical bf16 inputs; use warmed CUDA-event medians and time forward separately from forward+backward because FA3's gains differ.
- `train_batch` is DeepSpeed iter time; wall is log timestamp delta. Bucket device-kernel self-CUDA and drop self-CPU launcher rows (`aten::*`, `flash_attn_3::*`, `nccl:`).
- Bisect NaNs across layout, backend, precision, and traced/leaf execution; only fp32 changed this result. Timers, GC logging, and profiler code were removed.

## Round 5 — audio slowdown, cuDNN plan-cache thrash, and code-length quantization

### Controlled setup and headline
- One 8×H100 80 GB sm_90 node, DP=8, mb=4, GA=2, global 64, ZeRO-1, compile on; verbatim `stage-2-pose.toml`/`stage-3-audio-clean.toml` reduced to `hands=['clean-orig']` with step caps.
- Samples, batch, and sequence length match. Audio alone adds `audio_conditioning=true`, Mimi, and 10 cross-attention blocks over 30 layers (`injection_offset=2`, `injection_interval=3`).
- Extra cross-attention is not the problem: of the 246 ms gap, ~47 ms is real cross-attention and ~162 ms is per-length cuDNN plan rebuilding; code-length quantization removes it bit-exactly.

| config | wall p50 | `train_batch` p50 | samples/s | gap vs pose | stalls >1.2× median |
|---|---:|---:|---:|---:|---:|
| pose | 948.6 ms | 912.2 ms | 67.5 | — | 0/50 |
| audio as configured | 1194.9 ms | 1153.6 ms | 53.6 | +26.0% | 0/50 |
| audio codes rounded to multiple of 750 | 1035.9 ms | 996.1 ms | 61.8 | +9.2% | 0/50 |
| audio codes rounded to multiple of 256 | 1032.8 ms | 992.7 ms | 62.0 | +8.9% | 0/50 |
| padded audio + `overlap_comm=True` | 1037.9 ms | 998.4 ms | 61.7 | +9.4% | 0/35 |

- FLOP/traffic theory predicted +8%; measured audio was +26%, then +8.9% after padding, identifying ~18 points of implementation overhead.

### Video throughput

- Use video-seconds/wall-second. `VAE_TEMPORAL_STRIDE/FPS=4/25=0.16 s`; `attention_sink=true`, `num_history=8`, `num_present=8` form 16 latent frames (1 sink+7 real+8 present).
- Thus 1920=16×120 tokens represent 2.56 s/sample and global batch 64 is 163.84 video-seconds/step.

| run | step p50 | samples/s | video-s/wall-s | per GPU | present-only | video/day |
|---|---:|---:|---:|---:|---:|---:|
| pose | 948.6 ms | 67.5 | 172.7× | 21.6× | 86.4× | 4,145 h |
| audio as configured | 1194.9 ms | 53.6 | 137.1× | 17.1× | 68.6× | 3,291 h |
| audio fixed | 1035.9 ms | 61.8 | 158.2× | 19.8× | 79.1× | 3,796 h |

- Pose/fixed audio are 172.7×/158.2× realtime versus old audio 137.1×; one video hour takes 20.8/22.8/26.3 s respectively.
- Fixed audio is 91.6% of pose, up from 79.4%. Present-only is 8 frames=1.28 s/sample, confirmed by `loss_mask_sum=32.0=4×8`.
- Sink is clip frame 0 and early history is padded, so unique content is at most 15 frames=2.40 s; ×0.9375 gives strict lower bounds 162/129/148×.
- Exposed all-reduce does not shrink with nodes, though later four-node production slightly exceeded the estimate.

### Device profile of the 246 ms gap

Four profiled steps; values are device-kernel self-CUDA/step, with operator rows removed, plus idle estimated as `train_batch − device busy`.

| bucket (ms) | pose | audio | audio+padded | audio−pose |
|---|---:|---:|---:|---:|
| GEMM | 456.6 | 479.5 | 481.8 | +22.9 |
| Triton | 153.3 | 161.8 | 161.9 | +8.5 |
| NCCL | 101.8 | 188.4 | 95.4 | +86.6 |
| eager elementwise | 96.8 | 106.1 | 106.8 | +9.3 |
| FA3 self-attention | 59.5 | 59.2 | 59.3 | −0.3 |
| FA2 audio cross-attention | 0 | 7.5 | 7.5 | +7.5 |
| memcpy | 12.1 | 13.1 | 13.2 | +1.0 |
| other | 10.0 | 10.9 | 11.0 | +0.9 |
| device busy | 889.9 | 1026.4 | 936.9 | +136.5 |
| GPU idle in `train_batch` | ~75 | ~158 | ~94 | +83 |

- NCCL 101.8→188.4 ms for only +10.2% gradient volume, then 95.4 ms after padding at unchanged volume, is rank skew rather than bandwidth.
- GPU idle 75→158→94 ms agrees; padded audio adds +47 ms/step device work for 10 cross-attention blocks.

### Root cause: cold cuDNN plans

- Pose has no cuDNN rows because the DiT calls FA2/FA3 directly.
- Audio CPU profile adds:

```text
209.1 ms  n=32  aten::_cudnn_attention_forward
225.9 ms  n=32  aten::_cudnn_attention_backward
```

- This is ~108 ms/step host work over 16 calls while device kernels total <1 ms; 8 calls/step correspond to 4 Mimi layers × 2 microbatches.
- Four-layer `mimi.decoder_transformer` uses SDPA with non-null mask, producing “Flash Attention does not support non-null attn_mask” and a memory-efficient rejection, then selecting cuDNN.
- Input is `4×clip_len`; batch-max `_collate` padding changes it nearly every microbatch, forcing cuDNN heuristic/plan rebuilds.

| Mimi encoder B=4 fwd+bwd, median of 24 | host | CUDA |
|---|---:|---:|
| fixed T=500, plan-cache hit | 7.81 ms | 7.79 ms |
| varying T, training behavior | 89.67 ms | 89.65 ms |

- Varying is 11.5× slower; equal host/CUDA time proves serialization. Two calls cost ~180 ms varying versus ~16 ms fixed.
- Allocator churn is excluded (`num_alloc_retries=0`, 66.8/80 GiB reserved). Cache is per-process/per-shape; replaying lengths gives median 90.76→6.98 ms.
- In 60 steps/120 draws there were 83 lengths, min 109/max 737=59.0 s. At 12.5 Hz across ~9–59 s, ~600 shapes are reachable, so a 50-step run stays uniformly ~26% slow rather than spiking because shapes almost never repeat.
- `_mimi.npy` can exceed usable clip duration: `clip_len` clamps consumed latents, but the encoder processes the stored array.

### Shipped fix: right-pad to one code-length quantum

- Before the frozen front-end, right-pad `codes.shape[-1]` upward to a multiple of 750; nothing is sliced and the full clip is encoded. Cost is flat:

| fixed encoder input | host | CUDA |
|---|---:|---:|
| 310 tokens | 8.07 ms | 8.06 ms |
| 750 tokens | 7.80 ms | 7.78 ms |
| 1500 tokens, 60 s maximum | 7.89 ms | 8.18 ms |
| 3000 tokens | 8.74 ms | 13.75 ms |

- The largest relevant bucket is effectively free, so one shape minimizes DP=40 tail latency.

| audio variant | distinct shapes | wall p50 | max | max/p50 |
|---|---:|---:|---:|---:|
| as configured | 83 | 1194.9 ms | 1262.1 ms | 1.056 |
| multiple of 256 | 3 | 1032.8 ms | 1164.4 ms | 1.127 |
| multiple of 375 | 2 | 1036.9 ms | 1131.2 ms | 1.091 |
| multiple of 750 | 1 | 1035.9 ms | 1082.5 ms | 1.045 |

- One shape pays one step-0 build and then gives byte-identical rank work. Padding is bit-exact because decode is per-timestep, transformer causal, and gather reads `[0,4·clip_len)`.

| T→padded | encoder output shape | used prefix | max absolute delta |
|---|---|---:|---:|
| 250→256 | `(4,499,3072)`→`(4,511,3072)` | 496 | 0.000e+00 |
| 498→512 | `(4,995,3072)`→`(4,1023,3072)` | 992 | 0.000e+00 |
| 501→768 | `(4,1001,3072)`→`(4,1535,3072)` | 1000 | 0.000e+00 |

### Rejected alternatives and receptive field

- Backend switch, fresh process, 24 distinct lengths, fwd+bwd:

| backend with varying T | host |
|---|---:|
| cuDNN | 90.76 ms |
| memory-efficient | 18.16 ms |
| math | 19.95 ms |
| Flash | unavailable with non-null `attn_mask` |
| any backend at fixed T | 7.75 ms |

- Backend switching helps, but fixed length is 2.3× better. Slicing adds silent audio-history truncation risk while cost is flat (8.07 ms at 310 versus 7.89 ms at 1500), and would need 1000 context tokens rather than 250 because four layer windows compose.
- Mimi's four `sliding_window=250` layers compose to 1000 tokens. Perturbation used T=2400, probe 2000:

| distance behind probe | max delta at probe |
|---|---:|
| 1…900 tokens | 7.8e-03 |
| 1000 tokens | 0.0 |
| 1100, 1500, 1999 tokens | 0.0 |

- Receptive field is `250×4=1000` tokens=40 s at 25 Hz; a 59 s clip's oldest ~19 s is unreachable.
- Using more than the current 4/8 Mimi layers adds 250 tokens=10 s/layer. Perturbing all inputs after the probe leaves every output at or before it bit-identical (`0.0e+00`), formally validating right-padding.

### Cross-attention cost is healthy

- Production block shape: x `[4,1920,3072]`, audio `[4,15,5,3072]`; this is 60 sequences of 120 queries × 5 keys with 24 heads × 128.

| one `FrameCrossAttentionBlock` | forward | forward+backward |
|---|---:|---:|
| eager | 1.221 ms | 3.744 ms |
| compiled | 0.754 ms | 2.514 ms |

- `2.514×10×2=50.3 ms/step`, matching measured +47 and analytical ~54 ms; compile is 1.49× with no eager fallback.
- Fwd+bwd components: two `Linear(3072,3072)` 1.53 ms, attention 0.62 ms, norms+`view[:,1:].reshape` copy+`F.pad` ~0.4 ms.

| tiny-KV attention | forward | forward+backward |
|---|---:|---:|
| FA2, used by `audio.py` | 0.091 ms | 0.621 ms |
| FA3 | 0.103 ms | 0.671 ms |
| FA2 with KV=128, 25× FLOPs | 0.117 ms | 0.747 ms |

- FA2 saves 0.05 ms/call (~1 ms/step), confirming 0.89× FA3. KV 5→128 raises FLOPs 25× but time 1.2×: 0.15% FLOPs still consume ~25% of block time.

### NCCL diagnosis

Standalone 8-GPU all-reduce, median of 12:

| size | time | bus bandwidth |
|---|---:|---:|
| 7.162 GiB, pose 3.845B bf16 | 28.13 ms | 478.5 GB/s |
| 7.892 GiB, audio 4.237B bf16 | 31.28 ms | 474.1 GB/s |
| 0.931 GiB, one 5e8 bucket | 3.93 ms | 444.9 GB/s |
| 0.186 GiB | 0.86 ms | 405.4 GB/s |

- ~475 GB/s is flat; +10.2% volume predicts +3, not +87 ms. In-run pose/audio were 101.8/188.4 ms and varied 88.7→101.8/140.2→188.4, showing rank wait.
- Removing `audio_encoder.`/`audio_adapter.` optimizer groups (3→1) moved four-step NCCL 753.6→661.9 ms and wall 1278.5→1265.4 ms, negligible; padding moved 188.4→95.4 without parameter change.
- The config prints `overlap_comm=False`; forcing true changed 1032.8→1037.9 ms. ZeRO-1 contiguous gradients reduce after backward; overlap is a ZeRO-2/3 lever.
- Pose is still ~2.4× standalone (101.8 vs 28.1 ms): in-run RING_LL rather than Simple/NVLS leaves an uninvestigated, config-independent ~70 ms/step opportunity.

### Scale behavior, correction, and shipped locations

- One node had no stalls: 0/50 above 1.2× median, p99/p50=1.04, max 1262 versus 1194.9 ms; fetch 0.5 ms p50, data 3.5%.
- Rank-specific cold plans inflate NCCL 101.8→188.4 at DP=8 and approach one straggler/step at DP=40. This initially explained production +20% versus one-node +26% as plan thrash plus fixed scale cost; four-node data explicitly retracted the latter—collective cost is flat 8→32, and plan thrash alone caused the idle.
- Fabric was healthy: 8 mlx5 HCAs Active/LinkUp at 400 Gb/s; `ibp*` down; `NCCL_SOCKET_IFNAME=eth0` is bootstrap only and data uses IB verbs. Four-node timing answers the proposed `NCCL_DEBUG=INFO` check: socket fallback is incompatible with the measurement.
- `_extract_audio_features` uses quantum 750=60 s at 12.5 Hz, above max 737; multiples degrade safely to two shapes above 60 s.
- Applied in `models/pose/vidcon_variable.py::Wan5BVidConVariablePipeline` for `stage-3-audio-clean.toml`/`vidcon_aloe.toml`, and `models/resolution/compressed_5b_vidcon.py::Wan5BCompressedPipeline` for `compressed_5b_vidcon` plus subclass `compressed_5b_pose`, which does not override the method.
- `wan_5b_vidcon.py` stays unchanged: `NUM_LATENT_FRAMES=20` gives fixed 73920-sample raw audio. Bidirectional `audio_raw` has no producer; verify length if revived. Inference pays one negligible build per generation.

### Four-node production validation

- Production `stage-3-audio-clean.toml`, `micro_batch_size_per_gpu=5`, `gradient_accumulation_steps=3`, DP=32/global 480 measured steady 267 samples/s=1797.8 ms/step.
- Subtract data 61, optimizer 3, EMA 1.2, misc 1.2 ms: `3×(5c+b)=1731.4−NCCL`.
- Single-node gives `4c+b=444.9` from 996.1−95.4 NCCL−11 optimizer over 2 microbatches; 94 ms idle independently gives `b≈47`.

| assumed NCCL at 32 ranks | implied `c` | implied `b` | comparison to measured b≈47 |
|---|---:|---:|---|
| 80 ms | 105.6 | 22.4 | too low |
| 95 ms | 100.5 | 43.0 | consistent |
| 110 ms | 95.6 | 62.4 | high |
| 150 ms | 82.3 | 115.8 | impossible |

- Three quantities agree: NCCL≈95 matches `b`, and implied `c=100.5 ms/sample` matches profiled 103.8. All-reduce is flat from 8→32 ranks because the hierarchical algorithm's extra inter-node phase hides inside already-paid cost.
- Without padding, 3×~82 ms gives ~2044 ms/step or 235 samples/s. Actual 267 is +13.7%, matching one-node +13.3% (1194.9→1035.9), with no stalls.
- Forecast 253 (band 245–255) was 5.2% low by scaling fixed `b` with samples and charging 60–95 ms false inter-node penalty.
- Size with `step≈GA×(5c+b)+NCCL+66`, `c≈100`, `b≈45`, `NCCL≈95`; independence is observed through 32 ranks.

### Reproduction notes and traps

- Throwaway `learn_pose.py` removed wandb/eval/checkpointing; add synchronized `perf_counter` timers around `prepare_inputs`/`_build_window`/`_extract_audio_features`, profile 4 steps after 15 warmup, and record allocation retries.
- Bucket kernels while dropping nonzero self-CPU rows; use one `clean-orig` bucket. CUDA-event microbench `FrameCrossAttentionBlock`, `MimiAudioEncoder`, and `dist.all_reduce`; check padding parity plus token perturbations.
- Each backend needs a fresh process because warm caches misleadingly show ~7 ms. Profiler teardown lands in the next step and can mimic a 20 s stall.

## Round 6 — wide-model FLOP floor and prepare serialization

### Setup and headline
- `vidcon_audio_clean_masked.toml`, `wan_5b_wide`: 12 frames/1440 tokens, present 4, masking, mb=3, GA=5, ZeRO-1, skip=30, compile on; narrow `wan_5b` control.
- Exclusive 8×H100 80 GB `slinky` nodes; one `clean-orig` bucket of 110,292 clips; DP=8/global 120 samples/step.
- Runs: 42 steps, stats 13–35, 4 profiled with round-4/5 methodology; losses decreased from real 150k/135k EMA checkpoints.

| variant | wall p50 | `train_batch` p50 | samples/s | MFU |
|---|---:|---:|---:|---:|
| narrow control as configured | 1874.2 ms | 1533.2 ms | 64.0 | 29.3% |
| narrow with both fixes | 1594.9 ms | 1588.3 ms | 75.2 | 34.4% |
| wide as configured | 2412.7 ms | 2051.2 ms | 49.7 | 34.9% |
| wide + GPU probe | 2293.8 ms | 2020.7 ms | 52.3 | 36.7% |
| wide + GPU probe + lazy iterator | 2110.0 ms | 2102.2 ms | 56.9, +14.4% | 39.9% |

- The +14.4% is 56.9 versus wide-as-configured 49.7 samples/s. Narrow-as-configured still contains the eager-prepare bubble; only its “both fixes” row removes it.
- Wide memory is 67.0 GiB at DP=8, 57.2 GiB at DP≥32, versus narrow 48.3; best config consumes 1.92 s video/sample at 109× realtime/node or 36× for 4 supervised frames.

### FLOP model validation

- Model: `step≈GA×(mb·c+b)+NCCL+fixed`; 12/16 frames scale linear buckets 0.75 and Flash 0.583 via `(960²+480·1440)/(960²+960·1920)`.
- Wide/narrow GEMM scale `(3840/3072)²=1.5625`; elementwise/Flash 1.25; audio blocks 8 versus 9.

| quantity | predicted | measured |
|---|---:|---:|
| narrow device `c`, ms/sample | 77.6 | 82.6 |
| wide `c` | 110.6 | 116.4 |
| wide `train_batch`, mb=3/GA=5 | 2055–2105 ms | 2051.2 ms |
| wide/narrow compute ratio | 1.425× | 1.41× |

- The wide port has no hidden inefficiency: its ratio exactly matches predicted FLOPs. Both `c` values are ~5% high from fixed kernel cost at 1440 tokens, which cancels in the ratio.

### Microbatch utilization and remaining floor

- Wide GEMMs: 794.7 TFLOP/1069.9 ms=743 TFLOP/s=75.1% bf16 peak; narrow 73.7%, near cuBLAS ceiling. `mb×1440=4320` rows saturates.
- mb=4 needs 6 checkpointed blocks (`skip=24`) and worsens 142.5 versus 136.7 ms/sample; mb=4/5 at skip=30 OOM, so mb=3/skip=30 is optimal.
- MFU gap is structural: ~567 ms memory-bound elementwise, split as Triton norms/adaLN/GELU/RoPE ~290 and eager grad adds/copies ~278 ms.
- Flash is 29% peak, NCCL ~99 ms, residual bubble ~165 ms; complex RoPE is 19.8 ms/step=0.9%, and autograd grad accumulation ~50–100 ms.
- Grad-accum dtype and real RoPE change numerics and were rejected. Compile captures fusible work; uncompiled Mimi is ~40 ms/step and now overlapped.

### Missing 15%: serialized eager preparation

- Eager materialization of every GA microbatch caused ~370 ms prepare idle: `_build_window` 13–147 ms, mask probe ~22, Mimi ~8, plus splits/noise/fp32 conversion.
- Fix 1: single-stage `utils/evaluation.py` returns lazy `itertools.islice`, overlapping prepare i+1 with kernels i and reducing wall−train from ~370 to ~8 ms.
- Fix 2: `models/pose/wan_5b_vidcon.py::_build_loss_mask` runs probes on CUDA and returns masks to latent device; old `device=latents.device` selected 8-thread CPU convolution because training latents are CPU tensors. Savings are ~80 ms/step.
- Item order and RNG calls are unchanged. GPU rounding may change knife-edge 0.5 mask cells, but the new definition starts at step 0 and does not break resume continuity.
- `NCCL_PROTO=SIMPLE` was 3% slower, confirming RING_LL is fine here; mb/skip alternatives also lost.

### Multinode scaling

All runs include both fixes and use production-style `torchrun`.

| nodes | ranks | wall p50 | samples/s | scaling efficiency |
|---|---:|---:|---:|---:|
| 1 | 8 | 2110.0 ms | 56.9 | — |
| 2 | 16 | 2115.8 ms | 113.4 | 99.7% |
| 4 | 32 | 2168.0 ms | 221.4 | 97.3% |
| 5 symmetric | 40 | 2165.5 ms | 277.1 | 97.4% |
| 5 production shape, `slinky-0` masked to 7 GPUs | 39 | 2109.7 ms | 277.3 | 100% |

- One→four nodes grows all-reduce 68→118 ms and ZeRO all-gather 31→77 ms, both RING_LL, but bubbles hide them; all-gather must triple before hierarchical ZeRO matters.
- Five-node slowdown no longer reproduces: 39-rank production shape matches one node. Old `get_data_iterator_for_step` behavior exposed 13–147 ms `_build_window` skew with a larger maximum over 39 than 16 ranks; laziness removes it.
- A ~90 s run cannot see heat soak; if hours-long slowdown remains, exclude `slinky-0` before blaming communication.
- Four nodes/global 480 give ~221 samples/s. Two epochs over 356,277 entries are ~1,484 steps: ~54 minutes on 4 nodes or ~43 on 5.

### Reproduction notes

- `~/bench-runs` contains `bench_learn.py`/`bench_learn2.py` stripped of saver/eval/wandb, TOMLs, `bench_[a-k].sbatch`, logs, and `bench_stats.py`; run harnesses from repo root.
- Switches: `BENCH_PHASE`, `BENCH_GPU_PROBE`, `BENCH_LAZY`, `BENCH_PROF=start:steps`.
- `mask-probe-bench/` uses random weights at real architecture for representative cost, not real probe weights. De-duplicate Kineto `nccl:all_reduce` and `ncclDevKernel`.
