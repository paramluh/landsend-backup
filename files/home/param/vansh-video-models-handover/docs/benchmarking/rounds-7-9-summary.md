# Benchmarking rounds 7–9: consolidated record

This is a lossless technical condensation of `round-7.md` through `round-9.md`.
It preserves the measurements, implementation advice, correctness checks, rejected paths, caveats, and reproduction notes while removing repeated narrative.

## Round 7 — Universal Wide on the torchrun/ZeRO-1 trainer

### Setup
- First benchmark of the `universal-wide` branch after the DeepSpeed removal (`docs/development-chronology.md` §4). Rounds 1–6 profiled `wan_5b_vidcon`; this round profiles `UniversalWidePipeline` (`wan_5b_wide`: dim 3840, ffn 17920, 30 heads, 30 blocks) on `learn_pose.py`'s DDP + `Zero1FP32MasterAdamW` loop.
- 2 nodes of the `r` cluster (r0 master, r1), 16×H100 80 GB, IB between nodes, bf16 autocast.
- Window: 1 sink + 5 history + 5 present = 11 latent frames; `squeaky-clean-orig` buckets, pose v2, audio on.
- mb=3, GA=5, global batch 240 windows/step; `compile_blocks=true`; `activation_checkpointing=true` with `activation_checkpoint_skip=30`, which with 30 blocks means **no block is recomputed**: the peak memory below is stored activations.
- Harness: a copy of `learn_pose.py` without wandb/eval/saving, `torch.cuda.synchronize()` timers around data / `prepare_inputs` / fwd+bwd / optimizer, and an optional `torch.profiler` window on rank 0 (steps 30–31). Wall = mean over steps 13–42 (the first 12 absorb compile). Step-to-step noise ≈ ±1.5%; one run drifted to 1.78 s and repeated at 1.70 s, so single runs do not resolve below ~2%.
- Production config is mb=1 (`stage-3b-universal.toml`) and was not benchmarked; every number here is mb=3.

### Result

| state | step | fwd+bwd | optimizer tail | peak mem | samples/s/node |
|---|---:|---:|---:|---:|---:|
| baseline (`e48105fc`, 3 runs) | 1.795 s | 1.631 | 0.152 | 71.3 GB | 66.8 |
| + real RoPE (`51d82f61`) | 1.766 | 1.595 | 0.152 | 71.3 | |
| + compiled `InitialLayer` (`bd5a0ee0`) | 1.747 | 1.579 | 0.152 | 71.2 | |
| + flat ZeRO-1 / fused AdamW / all-gather (`7ceb4ed2`) | 1.704 | 1.601 | 0.089 | 73.3 | |
| + one graph per block (`b7f3ecac`) | 1.678–1.705 | 1.57–1.60 | 0.089 | 73.3 | 70.4–71.5 |

- Net −5 to −6.5% step time, +6% samples/s. Loss at every step matched the baseline to three decimals in every run (same data order, same seeds), and each change was checked numerically in isolation.
- Round-6 wide was 56.9 samples/s/node at 4 supervised frames; this is ~71 at 5 supervised frames per window, with 11-frame windows instead of 12.

### Where the step goes (profiled, rank 0)

| | baseline | final |
|---|---:|---:|
| span | 1839 ms | 1707 ms |
| GPU compute busy | 1594 | 1575 |
| NCCL total / exposed | 301 / 95 | 283 / 55 |
| idle (no kernel) | 150 | 78 |
| `cudaLaunchKernel` per step | 22169 | 7657 |
| `CompiledFunction` calls per step | 3320 | 2280 → ~300 after `b7f3ecac` |

- Compute is ~93% of the step and did not move: GEMM ≈ 1000–1100 ms (`nvjet` cuBLAS; ≈726 TFLOP/step of linear FLOPs → ~73% of bf16 dense peak, the round-6 efficiency), Inductor pointwise/reduction ≈ 150 ms, eager elementwise ≈ 130 ms (≈ 80 ms of it the `AccumulateGrad` `add_` of 4 accumulation microbatches over 13 GB of grads), FlashAttention ≈ 80 ms (FA3 fwd 18 + bwd 54, FA2 audio 7).
- The gradient all-reduce (207 ms bf16 + 33 ms fp32, `RING_LL`) is hidden under the last microbatch's backward, exposed 10 ms. The remaining exposed communication is the 41 ms parameter all-gather at the end of the optimizer step. The 78 ms idle is thousands of sub-50 µs launch gaps, not a few long stalls (0 gaps > 1.5 ms).
- Everything in this round removed CPU launch work and exposed communication; nothing shortened GPU compute, which is why a 3× cut in kernel launches bought 6%.

### Changes, each verified in isolation
- **RoPE in real arithmetic (`51d82f61`), −1.6%.** `apply_rope` used `torch.view_as_complex` and a complex multiply; Dynamo cannot trace complex ops, so every block's `_forward_impl` was three graphs (before q-RoPE, between q and k, after). `cached_freqs_i` now caches `view_as_real(freqs_i)` = (cos, sin) per grid and `apply_rope` does the 2×2 rotation on `x.float().unflatten(-1, (-1, 2))`. fp32 max |Δ| vs the complex path 4.8e-7 before the bf16 cast; 1e-5 of elements differ by one bf16 ulp after it. Round 6 rejected "real RoPE" for *computing* the frequency table in real arithmetic; this keeps the table and only changes how it is applied.
- **`InitialLayer` compiled (`bd5a0ee0`), −1.1%.** The Mimi-derived audio encoder (4 transformer layers) plus patch/pose/time embeddings ran eager; `_forward_impl` is now compiled under `compile_blocks`. The audio-dropout counters (`audio_dropped`, `audio_drop_seen`) moved out of the compiled function: a mutated Python int attribute is a guard and forced a recompile every call (first attempt 1.896 s, worse than baseline).
- **Flat ZeRO-1 parameters, fused AdamW, one all-gather (`7ceb4ed2`).** `Zero1FP32MasterAdamW.step()` ended with 16 serial `broadcast`s of `torch.cat`'d shards (85 ms exposed, plus the cat/copy). Now:
  - every parameter's storage is a view into one flat `[world, shard]` buffer per dtype (bf16 transformer; fp32 audio encoder / adapter / pose patch);
  - ownership is balanced per dtype, largest first (padding 49.5M bf16 + 15.9M fp32 elements ≈ 160 MB);
  - `torch.optim.AdamW(fused=True)` runs on the masters, and one in-place `all_gather_into_tensor` per dtype publishes the shards.

  Optimizer tail 152 → 88 ms (gather 41, fused Adam ≈ 9, grad casts/copies the rest). The optimizer is built **before** DDP so the reducer sees the re-homed parameters. Verified bit-identical to plain fused AdamW on a 3-rank toy, and the old parameter storage is freed (13.26 → 13.22 GB params). `load_state_dict` must put the AdamW `step` tensors on device (`3701bbe2`); fused AdamW rejects CPU state.
  Unexplained: peak allocated rose 71.2 → 73.3 GB with this change. Not shard imbalance (rank-0 shard 4.46 → 4.53 GB under the new assignment), not leaked storage, not allocator retries (0). Irrelevant at mb=1; would matter if mb=3 were pushed further.
- **One compiled graph per block (`b7f3ecac`), ~1% wall.** The round-6 wrapper `flash_attn_func = torch.compiler.disable(flash_attn_func)` was carried over on the belief that it made FlashAttention an opaque leaf inside the graph. It does not: Dynamo cannot put a disabled function in a graph, so it *breaks* the graph around every call. `torch._dynamo.explain` on a real block: 7 graphs (plain block) / 10 (block with audio cross-attention), matching the 2280 `CompiledFunction` calls per step.
  - Both installed wrappers (`flash_attn_interface` FA3 and `flash_attn` 2.8.3) register custom ops with fake kernels (`flash_attn_3::_flash_attn_forward/_backward`, `flash_attn::_flash_attn_forward`), so removing the wrapper gives one graph per block with FA as a custom-op node.
  - Numerics: same distance from eager as the old split compile (rel 2.8e-3 forward, one bf16 ulp max at |y|≈160; audio block forward bit-identical to the split compile).
  - Only ~1% wall because the GPU was already the bottleneck; it removes the CPU-bound backward and is the prerequisite for Inductor work across attention boundaries.
  - **Correction:** the rounds 1–6 summary's "opaque leaf" line (round 2, `_fa_func`) should be read with this correction.

### Fixes on the way that did not move the step
- `58939f05` `prepare_inputs`: `.to(device, float32, non_blocking=True)` from pinned bf16 is a blocking copy (the dtype change goes through an unpinned CPU temporary); move, then cast on device.
- `e48105fc` `sinusoidal_embedding_1d`: upstream Wan's `torch.arange(half).to(position)` is a pageable H2D copy that synced the stream every microbatch. Max |Δ| = 0.
- `6c3e1420` `sources.py`: the mimi code-length check raised on the 4.5% of clips with `T_mimi = 2L−2` (a 20k-clip scan found only `2L−1` and `2L−2`). Would have killed the first production step.
- Both sync fixes were real (150 ms of idle in the baseline profile, ~15 ms per microbatch traced to the copy), but the idle was overlapped by the launch-bound backward; step time unchanged (1.797 / 1.793). Fixing a stall is not the same as fixing the critical path.

### Rejected

| tried | result | note |
|---|---|---|
| `NCCL_PROTO=Simple` | 1.678 s, no change | all-reduce already hidden; the all-gather is bandwidth-bound either way |
| `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` | 1.746 s, worse | pointwise kernels were not the limiter |
| `mode='max-autotune-no-cudagraphs'` | not run | dropped when the cluster was needed; expected 0–4% |
| compiled `InitialLayer` with counters inside the graph | 1.896 s | recompile per call from the mutated int guard |

### Left on the table (in order of expected gain)
1. Overlap the 41 ms parameter all-gather with the next step's forward: lay the flat buffer out in layer order and issue per-owner broadcasts that early layers wait on. ~2%.
2. Gradient accumulation adds (~80 ms/step): 4 × read/read/write of 13 GB. Only removable by accumulating in the wgrad GEMM epilogue (`addmm_` into the grad view inside a custom linear), which conflicts with DDP's bucket hooks and would need our own bucketed reduce. ~4%, invasive.
3. Present-key K/V gather + its `index_put` backward (~38 ms/step) via a FlexAttention block mask instead of `k_frames[:, present_key_indices]`; FlexAttention is slower than FA3 per FLOP, so the net is uncertain (round 9 later took this path).
4. GEMM autotune (`max-autotune-no-cudagraphs`), 0–4%.

The remaining ~93% is GEMM at ~73% of peak plus the elementwise traffic of a 3840-wide model; a further 10% needs one of items 1–3, not more launch-overhead work.

### Operational notes for the production launch
- `output_dir` must resolve to shared storage. Every rank writes its EMA and optimizer shard into `run_dir`; rank 0 merges the EMA shards and reads `optim_meta.pt` / `model.safetensors` on resume. `/home` is a separate local disk per node, so a genuinely local path fails at the first save on the non-master node. The repo's `pose-checkpoints/` is a symlink to `/mnt/somfs/hands/pose-checkpoints/` on every node, so paths under it are fine.
- `save_q` → exit → `resume_from` with the flat optimizer had not been exercised end to end; only the periodic save path had.
- The two-node launcher's ssh pipe can outlive the run; kill the peer after the master exits.

## Round 8 — Aloe H4 on Universal Wide: compilation silently off after step 1

### Setup
- `learn_aloe.py` + `aloe-h4-universal.toml` (`wan_5b_wide`, sink + H4 + P5 = 10 latent frames = 1200 tokens/window, mb=4, GA=15, `activation_checkpoint_skip=30`, `compile_blocks=true`, DDP + `Zero1FP32MasterAdamW`). Production: 5 nodes, 40×H100.
- Bench: one H100 of param node 1, a fwd+bwd-only harness (model, data and `LayeredModel` built as in `learn_aloe.py`; no DDP, no optimizer; grads preallocated as with `gradient_as_bucket_view`). 72,000 tokens per GPU per step; RTF = wall / (60 windows × 5 present frames × 0.16 s). Steady state only.

### Result

| run | ms/microbatch | tok/s/GPU | RTF/GPU | peak |
|---|---:|---:|---:|---:|
| production `0hbtzog4`, 9.21 s/step median (whole step) | 614 | 7,820 | 0.192 | — |
| harness, as production (step-1 checkpoint, cpu//8 threads, limit 8) | 592–615 | 7,800–8,100 | 0.185–0.192 | 69.8 GiB |
| harness, no step-1 checkpoint | 361–364 | 13,200–13,300 | 0.113 | 60.8 GiB |
| harness, `recompile_limit = 64` (shipped) | 362–363 | 13,240 | 0.113 | 60.8 GiB |
| harness, `set_num_threads(1)` instead | 365–366 | 13,100 | 0.114 | 60.8 GiB |

Production ran its steady state eager: 1.65× slower and 9 GiB more memory than compiled.

### Cause
- Every block's `torch.compile(self._forward_impl)` shares one code object, so all 30 blocks share one Dynamo cache with `recompile_limit = 8`. Variants: plain / audio block × the first block's input stride.
- Dynamo's global-state guard includes grad mode and `num_threads`. Step 1 runs `checkpoint_all` (reentrant): the no-grad forward compiles 4 variants on the main thread (`cpu//8 = 16` threads), the recompute compiles 4 more on the autograd thread (1 thread). The steady-state grad forward on 16 threads is the 9th → `hit config.recompile_limit (8)` → every block eager for the rest of the run, silently. `OMP_NUM_THREADS` does not change the autograd thread's count.
- Fix: `torch._dynamo.config.recompile_limit = 64` in `learn_aloe.py` and `learn_pose.py` (13 graphs after step 1; eval at mb=5 adds its own no-grad variants). `set_num_threads(1)` also works, but round 1's thread count protects main-process CPU work, so it was not used.

### Where the compiled microbatch goes
Profiled (CUPTI), mb=4: 355 ms kernel time of 362 ms wall → idle ≈ 6–8 ms (2%) without the profiler; the profiler itself inflates idle to ~55 ms.

| bucket | ms | share |
|---|---:|---:|
| GEMM (`nvjet` cuBLAS) | 244 | 68.7% |
| Inductor pointwise/reductions (norms, adaLN, RoPE, GELU, residuals, casts) | 69 | 19.4% |
| FlashAttention 3, 90 calls fwd+bwd | 25 | 7.1% |
| eager (`AccumulateGrad` `add_` into 13 GB of grads) | 15 | 4.3% |
| copy/index | 1.6 | 0.5% |
| conv (patch embed) | 0.4 | 0.1% |

Forward per layer: `InitialLayer` (patch/pose/time embeddings + Mimi-derived audio encoder) 2.5 ms, plain block 3.5 ms ×22, audio block 4.05 ms ×8, `FinalLayer` 0.7 ms; forward 113 ms, backward ≈ 2.2× forward. ~170 TFLOP of linear work per microbatch → ~700 TFLOP/s ≈ 70% of bf16 peak; in isolation cuBLAS runs 4800×3840×17920 at 799 and 4800×3840×3840 at 764 TFLOP/s.

### Rejected

| tried | result | note |
|---|---|---|
| mb=5, GA=12 | 450 ms/mb, 13,330 tok/s, RTF 0.112, 69.2 GiB | same per token: 4,800 rows already saturate the GEMMs (round 6: 4,320) |
| `mode='reduce-overhead'` (CUDA graphs) | 360–361 ms | +0.5%; only ~2% idle to recover |
| `mode='max-autotune-no-cudagraphs'` | stopped | cuBLAS won every large GEMM shape; no Triton template faster |
| FP8 GEMMs | not run | ruled out (numerics) |

FSDP would not raise kernel efficiency; it only frees memory (not the limiter: mb=5 fits and does not help) and removes the grad adds (~4%), at the cost of per-microbatch parameter all-gathers and reduce-scatters that 0.36 s microbatches cannot hide across IB (estimate, not measured; round 9 measured it).

### Floor
The fwd+bwd floor is 15 × 362 ms = 5.43 s/step = 13,250 tok/s/GPU, RTF 0.113. Convs and the audio encoder are ~2%; the rest is GEMM at the cuBLAS ceiling plus the memory-bound elementwise traffic rounds 3–7 already called structural. 15k tok/s/GPU (320 ms/mb) is not reachable without changing numerics. What remains is the trainer's overhead above 5.43 s (optimizer, the last microbatch's all-reduce, data, logging), to be measured on the resumed run.

### Reproduction
`/tmp/fb-bench/bench_fb.py` on param node 1 and reserved node 4 (not in the repo):
`torchrun --nproc_per_node=1 bench_fb.py <config> <cycles> [checkpoint_first] [recompile_limit=N] [main_threads=N] [mb=N] [ga=N] [compile_mode=...] [profile]`. Kernel buckets match attention before GEMM (FA3 kernels are `cutlass::device_kernel<flash::...>`).

## Round 9 — 9-sink windows: flex attention, no recompute

### Setup
- `learn_aloe.py` + the 9-sink studio config (`wan_5b_wide`; window = 9 refs + sink + H4 + P5 = 19 latent frames × 120 tokens = 2,280 tokens; mb=2, `compile_blocks=true`, DDP + `Zero1FP32MasterAdamW`).
- Benchmarks Oct 10, branch `speed-opt`, agent transcript a635d85d. `bench_aloe.py` = the `learn_aloe` loop without wandb/eval/saves, GA=16 → 72,960 tok/step/GPU (refs counted, as `data/tokens_per_s`), median wall over post-warmup steps, peak = `torch.cuda.max_memory_allocated`, landsend 8×H100 nodes, placeholder copies of the real reference latents. RTF/GPU = wall / (windows × 5 present × 0.16 s), as round 8.

### Result

| run | nodes | s/step | tok/s/GPU | RTF/GPU | peak |
|---|---:|---:|---:|---:|---:|
| baseline: fixed (FA3) path, `activation_checkpoint_skip=20` | 1 | 6.637 | 10,992 | 0.259 | 66.6 GiB |
| baseline | 2 | 6.666 | 10,946 | 0.260 | 60.6 GiB |
| flex, skip=20 | 1 | 6.289 | 11,602 | 0.246 | 57.0 GiB |
| **flex, skip=30 (no recompute)** | 1 | 5.796 | **12,588** | 0.226 | 63.9 GiB |
| flex, skip=30 | 2 | 5.826 | 12,523 | 0.228 | 57.9 GiB |
| flex, skip=30, `NCCL_PROTO=Simple` | 2 | 5.817 | 12,544 | 0.227 | 57.9 GiB |

+14.5% tok/s over the baseline (target was 11.5k). The flex path frees ~10 GiB, which is what lets skip go 20 → 30. MFU estimate ~40% → ~46% (2.67 PFLOP useful/step); GEMM alone ≈ 74% of 989 TFLOP/s dense bf16.

### Where the step goes
Device-kernel time per step, 1 node, profiled step, with refs:

| bucket | baseline | flex, skip=30 |
|---|---:|---:|
| GEMM | 4008 ms (62%) | 3509 ms (64%) |
| Inductor pointwise/reductions | 1223 (19%) | 798 (14.5%) |
| attention | FA3 673 (fwd 157 / bwd 352 / pre 26, plus copies) | flex 629 (fwd 171 / bwd 437) |
| eager elementwise (bf16 grad-accum adds) | 326 | 338 |
| NCCL (all-reduce 73 `RING_LL` + all-gather 34) | 121 | 122 |
| copy / other | 67 | 67 |
| **device total** (wall) | **6421** (6637) | **5494** (5796) |

−927 ms device time: GEMM −499 and Inductor −425 ms (the recompute of 10 checkpointed blocks and the fixed path's K/V copies, not split further), attention kernels −44, eager +12. GPU ~96% busy.

### The flex path (`flex_attention = true`, `models/pose/wan_universal.py`)
- Fixed path (`_fixed_attention`, flex off): four FA3 calls per layer (refs self-only on B·R, sink self-only, history on B·H, present on B·P), each over explicit K/V copies: the refs' K/V replicated into every history and present frame (9 × 9 frame copies per window), the sink's into every history frame, and `k_frames[:, present_key_indices]` gathers with an `index_put` backward.
- Flex path: one `flex_attention` call per layer. `sink_geometry(freqs, grid, R, H, P)` returns the query and key RoPE tables and a `BlockMask`, built once per geometry in the module-level `_GEOMETRY` and shared by all 30 layers. Queries are the 19 frames `[refs | sink | history | present]` (2,280 tokens). Keys/values are the same 19 frames plus the 4 history frames appended again (23 frames, 2,760 tokens; `_flex_attention` does the `cat`). RoPE temporal positions (offset = `WINDOW_OFFSET` = 10 with refs, 0 without):

| query frame (q at t) | keys it reads (k at t) |
|---|---|
| ref r (0) | ref r only |
| sink (10) | sink only |
| history j (14, the anchor slot H) | all refs (0), sink (10), its own appended copy (14) |
| present i (15+i) | all refs (0), sink (10), the 4 frames before it and itself at true slots (11+i … 15+i) |

- These are exactly the key sets of `_fixed_attention`. The appended history copy (k at the anchor) is read only by its own history query; the true-slot copy only by present queries. A dropped ref slot is the pad token in both paths.
- `visible` is a 19 × 23 frame table; `mask_mod` reads `dense[q_idx, kv_idx]`, the same table expanded to 2,280 × 2,760 tokens. `create_block_mask` uses the default 128 × 128 blocks: 18 × 22 = 396, of which 156 are non-empty and 86 partial. Frames are 120 tokens, so most non-empty blocks are partial and `mask_mod` runs per score in them. Computed density 0.406 vs 0.295 exact (1.38×).
- With `compile_blocks` the flex call is traced inside each block's compiled `_forward_impl` (the `BlockMask` is an argument); without it `flex_attention` is compiled on its own (`dynamic=False`).
- Checkpointing: `LayeredModel` recomputes blocks with index ≥ `activation_checkpoint_skip` (reentrant), so skip=30 with 30 blocks recomputes nothing; the run's first step (`checkpoint_all`, also on resume) recomputes every block. `recompile_limit = 64` (round 8) still applies.
- Numerics: `tests/flex_parity.py` (production shape): flex vs FA3 rel 2.6e-3, each 4.7e-3 from fp32 SDPA. Oct 10 check (`~/eval/history-scratch/segB.md`): fp64 attention forward 8e-16, q/k/v grads 1.7e-12. Full model on trained weights in bf16: dense vs fp32 1.31e-2, flex vs fp32 1.30e-2, dense vs flex 8.95e-3.

### Changes, kept vs not

| commit (`speed-opt`) | change | tok/s/GPU (1 node) | status |
|---|---|---|---|
| `50149c4` | flex path (one pass, frame-level `BlockMask`) | 11,602 at skip=20, +5.5% | on `universal-sinks` |
| config | skip 20 → 30, possible after flex | 12,564 (no refs), +14% | in both studio configs |
| `de5935a` | `mask_mod` = token-table lookup instead of two integer divisions by 120 per score | 12,564 → 12,874 (no refs), +2.5%; flex fwd 303 → 166 ms/step | on `universal-sinks` |
| `1cd9ffc`, `7075e19` | `bench_aloe.py` harness | — | `speed-opt` only |
| `7670365` | `tests/flex_parity.py` | — | `speed-opt` only |
| `8c22353`, `acf25ca` | FSDP2 port of `universal-14b` `learn_bundle` (`utils/fsdp.py`) | 9,777 (−22%); with unsharded grad accumulation + bf16 reduce 10,557 (−16%) | `speed-opt` only, off by default |
| `5185a91` | `flex_block` (BlockMask granularity) + `flex_kernel_options` (tile sizes), configs e1–e4 | e1/e3 `NoValidChoicesError`, e4 shared-memory OOM, e2 flex bwd 768 vs 436 ms | reverted in `6c6cc4e` |
| `09774b1`, `9205b8d` | `experiments/benchmarking/sinks9-*.toml` | — | `speed-opt` only |

- The flex code (including the table lookup) reached `universal-sinks` as part of `d69cb0a` ("main rework done"); no `speed-opt` commit is contained in `universal-sinks` (`git branch --contains`). The flex section of `wan_universal.py` is identical between the two branches. `4436f54` (the flex config committed to `universal-sinks` by the agent) was force-pushed away.
- FSDP2 loses because the per-microbatch all-gather/reduce-scatter is not hidden under ~360 ms microbatches, and StableAdamW's per-parameter RMS clip on DTensors adds ~0.5 s/step.
- `NCCL_PROTO=Simple`: +0.2% at 2 nodes, not adopted.
- Not retried (ruled out in rounds 1–8, unchanged by the new layout): RoPE kernel/cast tuning, cuDNN/SDPA bf16, selective checkpointing, coordinate-descent tuning, max-autotune GEMM, CUDA graphs, mb 4 → 5, FP8.

### Production (landsend 0–8 + 10, 80×H100 80 GB)
Both configs: flex, skip=30, mb=2 × GA=8 × 80 GPUs = 1,280 windows/step, 36,480 tok/step/GPU, 4,774 steps/epoch, 14,322 steps. Step time and tok/s from wandb `canopy-labs/universal-aloe-new-ablations` (`data/step_time`, `data/tokens_per_s`, steps > 20). Memory = NVML used from wandb system metrics, which cover rank 0's node (landsend-0) only, against 81,559 MiB per GPU.

| run | median s/step (p10–p90) | mean | tok/s/GPU median (p10–p90) | RTF/GPU | peak, landsend-0 |
|---|---:|---:|---:|---:|---:|
| `lq38qgq6` studio-9-sinks-proper (`aloe-studio-v2.toml`), finished | 3.116 (3.095–3.530) | 3.244 | 11,708 (10,333–11,786) | 0.243 | 66,685 MiB |
| `uuz7tek9` with-words (`aloe-studio-v2-with-words.toml`), step 14,285 | 3.576 (3.190–3.795) | 3.564 | 10,201 (9,611–11,437) | 0.279 | 79,855 MiB |

- Step time has two modes in both runs and stays flat across training (per-2,000-step medians within ±0.015 s). `lq38qgq6`: 74.9% of steps at median 3.108 s, 24.9% at 3.515 s. `uuz7tek9`: 15.4% at 3.187 s, 84.4% at 3.583 s. 24 / 25 steps take ≥ 5 s (max 14.6 / 21.5 s), among them the step after each eval (every 500). (Round 10 traced the two modes to gen-2 GC collections over the compiled graphs.)
- `lq38qgq6` wall 48,795 s (13.6 h) for 14,322 steps.
- Compared with the bench: 11,708 vs 12,588 tok/s/GPU (−7.0%). The production GA=8 step is 0.218 s longer than half of the GA=16 bench step (2.898 s); not profiled in this round.
- Hidden adapter (`[model.hidden]`, 4 text cross-attention blocks at layers 11, 14, 17, 20; `8b88269`). The two runs also differ in the `llm_hiddens` data path and `audio_dropout` (0.0 vs 0.05). The difference between them is +0.460 s median / +0.320 s mean per step (+14.8% / +9.9%), and +13,170 MiB on landsend-0. The adapter's linear work (q/out projections, 3840², on 1,200 present tokens per microbatch × 4 blocks) is ~0.85 TFLOP fwd+bwd, about 0.5% of the 161 TFLOP linear work per microbatch. The time gap is not profiled.
- Headroom: `lq38qgq6` 14,874 MiB on landsend-0. `uuz7tek9` 1,704 MiB on landsend-0; across all 10 nodes, nvidia-smi at step 13 showed 78,756–79,856 MiB max used, 1,703–2,803 MiB free (`~/eval/launch-words/mem1.txt`).
- The first with-words launch used 8 text blocks (`injection_offset = 5`, `max_blocks = 8` → layers 5, 8, …, 26). Step 1 completed (loss 0.2253), then every rank OOMed in step 2: rank 0 had 76.43 GiB allocated by PyTorch of 79.18 GiB capacity when allocating 156 MiB (`~/eval/launch-words/crash1.txt`). Step 1 runs `checkpoint_all`; with skip=30, step 2 is the first step with no recompute. More adapter blocks need skip < 30 (recompute cost: the baseline-vs-skip=30 rows above).
- Inference (`production/cached.md`, measured on aloe-h4 without refs, not re-measured with refs): cached sampler, compiled + CUDA graph, 7.15 ms/pass on 120 query tokens, 365 ms/frame (50 steps + commit), RTF 2.28 per GPU. Production ~366 ms/frame; a 5-min clip (1,874 latent frames) takes ≈ 11.5 min + ~3 min decode.

### Reproduction
`speed-opt` (worktree `~/eval/speed-opt-wt`, head `6c6cc4e`): `torchrun --nproc_per_node=8 bench_aloe.py --config experiments/benchmarking/<cfg>.toml --steps 12 --warm 4 [--profile START:COUNT] [--trace DIR]`. Configs `sinks9-{baseline,baseline-skip30,flex,flex-skip30,flex-fsdp,flex-fsdp-skip30,flex-fsdp-skip30-acc,flex-fsdp-skip30-mb4}`; `sinks9-baseline-skip30` was not run. Their `[refs] root` predates the actor bank (`[refs] bank`) used by the current configs. `python tests/flex_parity.py [B]` checks the two attention paths against fp32 SDPA.
