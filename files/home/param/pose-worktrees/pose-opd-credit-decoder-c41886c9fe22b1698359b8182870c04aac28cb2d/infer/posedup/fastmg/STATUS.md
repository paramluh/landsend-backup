# fastmg: multi-GPU live engine (paused 10-25 ~14:33 UTC for the pose fix)
No live file was touched. Dev job 30349 (slinky-18 GPUs 4-7) cancelled. 30343 (slinky-4) cancelled earlier.

## Current engine breakdown (real export-498, slinky-18 H100, 2 GPUs as live; out/prof_cur.json)
- p50 / p95: pose CFG off 63.95 / 64.31 ms; on 67.58 / 67.94 ms. At conversation frame 7300 (~9.7 min): off 73.6, on 77.1 ms (RTF 0.96).
- Critical path: text 1.75 (GPU1), then the speech branch (tower + 31-step 0.6B decoder) 34.5. The pose branch on GPU0 starts only
  after the 2.8 ms front, and runs 34.1 ms (CFG on: 37.6), so it ends after speech and gates seg0. Then seg0 (37 layers) 14.2,
  seg1 (27 layers) 10.3, and mdec 1.4 trailing.
- Late context: the towers and the trunk use FA3 num_splits=1, so attention grows (seg0+seg1 32.5 ms at f7300).

## Built (fastmg/, new files only)
- tpk.py: single-process P2P primitives. Device-side flags; one-shot all-reduce fused with residual add + RMSNorm (fixed rank-order
  fp32 sum, one bf16 rounding); tensor-core GEMV (M<=16) at cuBLAS speed with fp32-out option; hand-off push/wait;
  qkv_post (bitwise = fused_triton qk_prep).
- live_ppm_pose_mg.py: LivePPM subclass of the live engine.
  - Layout: speech+head on GPU0, pose on GPU1, front+Mimi on GPU2 (Mimi on a side stream).
  - Trunk is tensor-parallel over all GPUs (TP 2/4/8), one CUDA graph per GPU, no host cross-device sync. TP prefill. Loads in 47 s.
- fdec.py: 5-kernel-per-layer 0.6B depth decoder (written, NOT yet tested).

## Measured so far (random weights, dev/t_tp.py, out/t_tp.json)
- TP trunk, 64 layers, 3 rows: TP4 8.3 ms (T=100), 9.0 ms (T=9000, attn_splits 0). TP2: 12.9 / 14.2. Today's 2-GPU trunk: 24.5-32.5.
- TP vs unsharded reference: error equals the noise floor of merely reordering the fp32 sums (rel L2 0.94% vs 0.93% after 4
  random layers). Not bitwise: cuBLAS on weight slices is not bitwise either (split-K; dev/t_exact.json).
- FA3 per-head shards and the no-append path are bitwise.

## Not done yet
- Real-weights end-to-end run of the mg engine. The last run crashed in a capture warm-up after 'audio' (pose / codespush jobs);
  rerun with FASTMG_DEBUG=1 to see which job.
- Expected from the parts: about 45 ms off / 48 ms on (decoders 30-34 ms stay the bottleneck), so no deployable milestone yet.
- Next steps:
  1. fdec (decoder 30 ms -> ~13-15 ms target, plus PDL).
  2. Tower num_splits 0.
  3. Speech / pose towers TP-2 on idle GPUs.
  4. Server flag (PPMLIVE_ENGINE=mg).
