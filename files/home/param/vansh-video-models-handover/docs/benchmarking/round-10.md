# Round 10 — stabiliser (sink + H3 + P4, no refs): the gen-2 GC stall, flex as a memory lever

`learn_aloe.py` + `experiments/bimodal/aloe-studio-stabiliser-h3-p4.toml` (`universal_wide_stabiliser`, `wan_5b_wide`;
window = sink + H3 + P4 = 8 latent frames × 120 tokens = 960 tokens, no refs, `sink_pose = true`; dense attention,
`activation_checkpoint_skip = 30`, `compile_blocks = true`, DDP + `Zero1FP32MasterAdamW` + EMA). Benchmarks Oct 11–12,
branch `universal-2-models` @ `c889276`, reserved Nebius nodes r0–r4 (5 × 8 H100 80 GB, 8 × 400 Gb/s NDR IB per node),
real data from `/mnt/paramfs` (`aloe-studio-no-blackbands-high-quality`, 16,342 chunks).

Harness `/mnt/paramfs/vansh/bench/learn_bench.py` (`bench_aloe.py` of round 9 brought up to the `learn_aloe` loop of this
branch: same model, data, DDP + ZeRO-1, EMA, per-step loss all-reduce, `_step_uniques` all_gather, loss buckets and
manual-signal check; no wandb, eval or saves). Instrumentation without extra syncs: CUDA events around every microbatch,
the optimizer and the post-step; CPU timers per phase (data wait, prep, enqueue, `uniq`, opt→item); a `gc.callbacks`
hook counting gen-2 collections and their wall per step; all ranks' records all-gathered into one JSON. `mkcfg.py NAME
k=v …` derives `cfg/NAME.toml` from the real config (wandb off, eval/saves off, out dir on paramfs); `run1.sh` /
`runN.sh` launch one / five nodes with the repo's venv and `PYTHONPATH`; `ana.py out/*.json` prints the summary.

tok/s/GPU is the `data/tokens_per_s` convention (rank-0 window tokens incl. sink + history, per GPU, per second):
mb4 × GA8 = 32 windows × 960 = 30,720 tok/step/GPU; mb6 × GA5 = 28,800; mb8 × GA4 = 30,720. Median wall over post-warmup
steps (6 warm steps dropped; the first two steps compile). Peak = `torch.cuda.max_memory_allocated` (max over ranks) /
NVML used. RTF/GPU = wall / (windows × 4 present × 0.16 s).

## Result

| run | nodes | mb × GA (windows/step) | attention | gc refreeze | median s/step | p10 – p90 | max | tok/s/GPU | RTF/GPU | peak alloc / used |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| m2-flex | 1 | 2 × 16 (256) | flex | – | 2.781 | 2.774 – 3.019 | 3.036 | 11,045 | 0.136 | 50.4 / 52.4 GiB |
| m4-fixed (= production config) | 1 | 4 × 8 (256) | dense | – | 2.540 | 2.534 – 2.690 | 2.875 | 12,094 | 0.124 | 67.4 / 69.4 GiB |
| m4-flex (profiled) | 1 | 4 × 8 (256) | flex | – | 2.543 | 2.533 – 2.744 | 2.774 | 12,079 | 0.124 | 61.1 / 63.1 GiB |
| m6-fixed | 1 | 6 × 5 (240) | dense | – | OOM in step 1 (77.25 GiB allocated of 79.18) | | | | | |
| m4-fixed-gc (production config, gc instrumented) | 1 | 4 × 8 (256) | dense | – | 2.561 | 2.549 – 2.863 | 2.980 | 11,996 | 0.125 | 67.4 / 69.4 GiB |
| m8-fixed | 1 | 8 × 4 (256) | dense | – | OOM in step 1 (77.2 GiB allocated of 79.18) | | | | | |
| m8-flex | 1 | 8 × 4 (256) | flex | – | OOM in step 1 (77.2 GiB allocated of 79.18) | | | | | |
| **N-m4-fixed (= production config)** | 5 | 4 × 8 (1,280) | dense | – | **2.896** | 2.573 – 3.015 | 3.068 | **10,608** | 0.141 | 60.9 / 63.4 GiB |
| **N-m4-fixed-rf** (production config + GC fix, `NCCL_DEBUG=INFO`) | 5 | 4 × 8 (1,280) | dense | 4 | **2.579** | 2.567 – 2.587 | 2.591 | **11,913** | 0.126 | 60.9 / 63.4 GiB |
| N-m4-flex-rf | 5 | 4 × 8 (1,280) | flex | 4 | 2.562 | 2.546 – 2.576 | 2.649 | 11,992 | 0.125 | 54.7 / 57.1 GiB |
| **N-m6-flex-rf** | 5 | 6 × 5 (1,200) | flex | 4 | **2.361** | 2.355 – 2.375 | 2.397 | **12,197** | 0.123 | 65.5 / 68.0 GiB |
| N-m8-flex-rf (10 steady steps) | 5 | 8 × 4 (1,280) | flex | 4 | 2.414 | 2.409 – 2.418 | 2.420 | 12,726 | 0.118 | 76.3 / 78.8 GiB |

"gc refreeze = 4" is the `gc.freeze()` edit below applied to the first 4 steps. Single-node m4 (fixed or flex) is 12.1k; the
production config loses 12% of that on 5 nodes (10.6k) to the stall described next, not to the network. With the GC fix the
5-node step is 2.579 s, 1.5% over the single-node step — the remaining inter-node cost is 60 ms of exposed all-reduce.
mb6 flex at 1,200 windows/step reaches 12.2k; 12.5k is only reached by mb8 flex, which has 0.4 GiB of headroom and OOMs on
one node, so it is not recommended. Target 12.5k: not reached safely (12.2k, −2.4%).

## Where the step goes

Device-kernel time per step, 1 node, m4-flex, 4 profiled steps (device kernels only, launcher rows dropped, round-8 method):

| bucket | ms/step | share |
|---|---:|---:|
| GEMM (`nvjet_sm90` ×6 shapes) | 1,601 | 62.6% |
| Inductor pointwise/reductions (`triton_*`) | 353 | 13.8% |
| attention: flex fwd 61 + bwd 158 (+ 21 flash in the audio adapter) | 246 | 9.6% |
| eager elementwise (`add<bf16>` grad accumulation 119, `add<float>` 19, binary 26) | 221 | 8.6% |
| NCCL (all-reduce `RING_LL` 73 + ZeRO-1 all-gather 34) | 119 | 4.7% |
| copy | 17 | 0.7% |
| **device total** (wall) | **2,556** (2,543) | |

Device total ≥ wall: the NCCL stream overlaps compute, the GPU is ~100% busy. The dense path has the same wall (2.540 vs
2.543), so the attention kernels are interchangeable at 960 tokens. Nothing overhead-shaped is left on one node.

On 5 nodes the CUDA-event spans per rank (N-m4-fixed-rf, medians): 7 microbatches 2.08 s (0.302 each) + last microbatch
0.365 (= 0.302 compute + 0.06 exposed all-reduce tail) + optimizer span 0.101 (loss all-reduce, `_step_uniques`
all_gather, clip, Adam, ZeRO-1 all-gather, EMA) = 2.55 ≈ wall 2.579. Dataloader wait (max over 40 ranks): 2–3 ms median,
14 ms max — paramfs with 8 workers is not a factor. GPU clocks under load 1,980 MHz (max), ~150 W sampled between steps,
throttle reasons 0x0.

## Variance: the gen-2 GC stall

N-m4-fixed (production config as is) is bimodal on 5 nodes: 10 of 34 steps at 2.57–2.61 s, 24 at 2.80–3.07 s. Per rank, all 40:

- compute of microbatches 1–7 is 2.05–2.13 s on every rank and every step; optimizer span 0.103 on every rank; data wait ≤ 14 ms;
- the last microbatch, which carries the DDP all-reduce, is 0.36–0.40 s on fast steps and 0.58–0.85 s on slow steps, on every
  rank at once; the CPU wait at the first sync point (`uniq`, the all_gather after the loss all-reduce) grows by the same
  0.3–0.5 s. No node or rank is consistently slow: the whole group waits for whichever rank is late at the all-reduce.

N-m4-fixed-rf differs only by re-issuing `gc.freeze()` after each of the first 4 steps. It is flat: 2.567–2.591 s over 34
steps (p90 − p10 = 0.02 s), last microbatch 0.35–0.38 on every step. Its gc hook shows every rank running ~10 gen-2
collections in 34 steps (one every ~3.4 steps), each costing ≤ 2 ms once the graphs are frozen. Direct measurement on one node (m4-fixed-gc, production code path, no refreeze, 24 steps): every step without a gen-2 collection on any rank takes 2.53–2.57 s; the five steps in which one or more ranks collect take 2.73–2.98 s, and the collecting rank's gen-2 pause is 0.27–0.52 s (step 11: rank 7 0.31 s → 2.76; step 12: ranks 0/1/2/4 0.35–0.39 s → 2.86; step 13: ranks 5/6 0.52/0.30 s → 2.98; step 14: rank 3 0.33 s → 2.81; step 23: rank 7 0.27 s → 2.73). One rank's pause becomes everyone's step time.

Mechanism: `learn_aloe.py` calls `gc.freeze()` once before the loop, *before* the first step compiles the 12 variants of
`_forward_impl` (`TORCH_LOGS=recompiles`: 12 recompiles per rank, all in steps 1–2, guard `KeyError on self.audio_block`
= audio-block × codebook-count variants; none afterwards, no graph breaks). The dynamo/inductor objects of those graphs
stay in the unfrozen generations, and every gen-2 collection traverses them; the collecting rank's Python thread stops for
0.3–0.6 s, its GPU queue drains, its bucket contributions reach the ring late, and all 40 ranks wait. The 40 ranks collect
at slightly different steps, so most steps catch at least one stall (70% slow here; 25%/84% slow in the two round-9
production runs on 80 GPUs). The same mechanism explains round 9's unprofiled "+0.218 s at GA=8 vs half of GA=16" (fewer
steps per collection cycle ⇒ a larger share of steps stalled), the ≥ 5 s step after each eval (eval allocates and drops
many objects ⇒ a gen-2 collection soon after; the fix also re-freezes after `run_eval`), and why the round-9 bench — which
freezes once after compiling — beat production by 7%.

The fix (unstaged in `learn_aloe.py`, no change to the training math):

```
         if eval_loaders and step % config['eval_every_n_steps'] == 0:
             run_eval(step, examples)
+            gc.freeze()
+        # Compiled graphs are built in the first steps (and the first eval), after the gc.freeze() above; unfrozen,
+        # every gen-2 collection rescans them (0.3-0.6 s per rank, straggling the whole DP group at the all-reduce).
+        if step < first_step + 4:
+            gc.freeze()
```

Effect on 5 nodes: 2.896 → 2.579 s median (−11%), p10–p90 0.44 → 0.02 s, max 3.068 → 2.591. 10,608 → 11,913 tok/s/GPU
(+12.3%).

## Flex on/off and the maximum micro batch

Flex (`flex_attention = true`, the round-9 path: one `flex_attention` per layer over a frame-level `BlockMask` from
`sink_geometry`) vs the dense path (`_fixed_attention`): identical compute at 960 tokens (2.540 vs 2.543 s on one node,
2.579 vs 2.562 on five). Memory: −6.3 GiB at mb4 (67.4 → 61.1 GiB on one node; 60.9 → 54.7 on five, where the ZeRO-1
shards are smaller).

Max micro: mb8 OOMs on one node for both paths (77.2 GiB allocated when the 44–58 MiB allocation fails); the 5-node mb8
flex run survived 16 steps at 76.3 GiB allocated / 78.8 GiB used of 79.18 — 0.4 GiB headroom, with no eval having run.
mb6 flex: 65.5 / 68.0 GiB on five nodes, 11 GiB headroom. mb6 dense OOMs on one node (77.25 GiB allocated); on five nodes the ZeRO-1 shard is 6.5 GiB smaller, so it would likely fit (~72 GiB), untested.
**Max safe micro: 6 with flex, 4 without (6 only with the 5-node shard size, unverified).** Flex is worth turning on only as the memory lever that allows mb6; by
itself it changes neither speed nor variance.

## NCCL transport

N-m4-fixed-rf ran with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV` (`logs/N-m4-fixed-rf.n*.log`): NCCL 2.28.9+cuda13.0,
`Using network IB`, `NET/IB : Using [0]mlx5_0:1/IB … [7]mlx5_7:1/IB [RO]; OOB eth0`. Every inter-node channel endpoint is
`via NET/IB/<hca>/GDRDMA`, spread evenly over the 8 HCAs (274–278 endpoints each across the five node logs); intra-node
`via P2P/CUMEM`; zero `NET/Socket` transports; 16 collective channels. GPUDirect RDMA on all 8 HCAs, no socket fallback —
the network is neither the slowness nor the variance.

## Overlap candidates (not changed)

Measured with the CUDA-event spans above: the GPU timeline is full (span sum ≈ wall on every rank), so there is no idle gap
for a host-side change to fill.

- `_step_uniques` (`uniq` = 52 ms of CPU time on rank 0): that is the CPU waiting for the GPU to drain the last microbatch
  and the loss all-reduce; the all_gather itself moves a few KB. Deferring or making it async saves ~0 GPU time.
- Exposed all-reduce tail (60 ms, 2.3%): NCCL kernels already run on their own stream under the backward; the tail is the
  part of ~11 GB of bf16 gradients that the ring has not finished when the last backward ends. Shrinking it means fewer,
  larger buckets (less overlap) or a reduce dtype change (numerics) — not done.
- Optimizer span (101 ms, 3.9%): Adam on the fp32 master shard + the ZeRO-1 all-gather of the updated bf16 parameters.
  Hiding the all-gather under the next forward needs per-layer parameter prefetch, i.e. the FSDP-style scheme ruled out in
  round 9 (−16 to −22%).

Combined ceiling if all three were fully hidden: < 6% of the step; none clears the ≥ 5% bar on its own.

## Changes, kept vs not

| change | where | 5-node effect | status |
|---|---|---|---|
| `gc.freeze()` after the first 4 steps and after each eval | `learn_aloe.py` (unstaged, r0) | 10,608 → 11,913 tok/s/GPU, p10–p90 0.44 → 0.02 s | **keep** |
| `micro_batch_size_per_gpu = 6`, `gradient_accumulation_steps = 5`, `flex_attention = true` (1,200 windows/step) | config proposal | 11,913 → 12,197 (+2.4% over mb4 dense + fix; 6,373 steps/epoch) | proposed |
| mb8 × GA4 flex | config | 12,726 but 0.4 GiB headroom, OOM on one node | not recommended |
| `_step_uniques` deferral, all-reduce bucket tuning, optimizer/all-gather overlap | — | no GPU idle to recover (above) | not done |

Newly ruled out: NCCL socket fallback or HCA imbalance; a slow node, the paramfs dataloader and clock throttling as
variance sources; flex as a compute speed-up for the 960-token no-refs geometry (0%); mb8 on either path. Still ruled out
from rounds 1–9 and not retried: RoPE kernel/cast tuning, cuDNN/SDPA bf16, selective checkpointing, coordinate-descent
tuning, max-autotune GEMM, CUDA graphs in training, FP8, FSDP2, `NCCL_PROTO=Simple`, flex BlockMask/tile tuning.

## What to expect in production (5 nodes, 40 GPUs)

| config | windows/step | steps/epoch | s/step | tok/s/GPU | epoch wall | 3 epochs |
|---|---:|---:|---:|---:|---:|---:|
| current (dense mb4 × GA8, no fix) | 1,280 | 5,975 | 2.896 (bimodal 2.57 / 2.9–3.07) | 10,608 | 4.8 h | 14.4 h |
| + `gc.freeze` fix | 1,280 | 5,975 | 2.579 ± 0.01 | 11,913 | 4.3 h | 12.8 h |
| + flex mb6 × GA5 | 1,200 | 6,373 | 2.361 ± 0.01 | 12,197 | 4.2 h | 12.5 h |

Plus eval every 500 steps and the end-of-epoch saves. Expect no step-time modes; if production still shows an after-eval
spike, the eval allocates after the re-freeze and a second `gc.freeze()` one step later would be the next thing to test.
Memory at mb6 flex: 68.0 GiB used of 79.18 on the training steps; eval at `eval_micro_batch_size_per_gpu = 2` fits inside that.

## Reproduction

`/mnt/paramfs/vansh/bench/`: `python mkcfg.py NAME [micro_batch_size_per_gpu=6 gradient_accumulation_steps=5
flex_attention=true bench.refreeze=4]` → `cfg/NAME.toml`; `bash run1.sh NAME HOST STEPS 1` (one node) or `bash runN.sh NAME
STEPS 1 [NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV TORCH_LOGS=recompiles,graph_breaks]` (r0–r4, master r0); logs in
`logs/NAME[.nI].log`, per-rank records in `out/NAME.json`, `python ana.py out/*.json` for the table; `--profile START:COUNT
--trace DIR` via `BENCH_ARGS` for the kernel buckets (round-8 bucketing in `learn_bench.py`). The runs of this round:
`m2-flex m4-fixed m4-flex m6-fixed m8-fixed m8-flex` (1 node), `N-m4-fixed N-m4-fixed-rf N-m4-flex-rf N-m6-flex-rf
N-m8-flex-rf` (5 nodes). Full notes: `/home/vansh/bench/results.md`.
