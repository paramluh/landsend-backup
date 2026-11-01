# aloe-ft-32-samsep RECIPE status (RECIPE sub-agent, 10-24 PT)

**State: READY TO LAUNCH, not launched.** Waiting on the base init staging (job 28238).
- Launch: `bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/launch.sh '' afterok:28238`
  - `''` = run.env's one-line `QQA_INIT`. Pass a path instead to swap the base for this launch only.
- CPU dry run (slinky-10, `tools/cfg_diff_dry.py`, outputs in `recipe/dry/`): **PASS**.
  - Config diff vs 26232: 44 keys differ, 0 outside the allowed set.
  - Plan digest d1a3e3721328da0a: 15,898 micros / 249 steps, 316 eval micros.
  - All 16,214 A FLACs present; 0 mimidec items.
  - 4 micros match the reference row except the B code values; A depth is all -1.

## Trainer choice
`code/train_samsep.py` = 26467's `train_cotrain.py` in a new aloe-only mode (`COTRAIN_GROUP_MODE=aloe_only`). Why not 26232's `train_aloe_v2vox_raw.py`:
- Its accumulation maps logical node -> physical node at 8 ranks per host (`ALOE_LNODES`), so it cannot run 8 + 6 ranks.
- It has no full-state save/resume.

The cotrain trainer already has both:
- generic slots: physical rank i runs logical ranks i + n s;
- `safety_ppm` STOP saves, tested exact in 26467's T4.

With no zoom ranks, every step is 26232's maths: the same 64 logical micros, the tokens_step normalisers, the decoder all-reduce, the s calibration and the step-0 loss check. The 26467 README states this for its Aloe side.

Edits:
- aloe-only mode;
- uneven hosts (`PPM_HOST_NRANKS`);
- 26232's local-shard `load_init`, ported verbatim (the 16861 init is ppm-local-bf16-shards-v1);
- the sam-sep switch (`data_aloe_v2vox.py`: in-place store re-point + RAWGAIN plan-digest key);
- the `[same-as-v2]` check against the reference row, with code values excluded;
- main-AdamW host offload (`code/optoff.py`);
- export space guard + `NODE_COMPLETE`;
- `safety_ppm.space_check` grouped by hostname;
- report-only config diff vs 26467.

The data modules `data_aloe_v2` / `data_ppm` / `data_qqa` / `data_aloe` are byte-identical to the reference's (sha-asserted at start).

## Layout
World 14: slinky-18 x 8 + slinky-4 GPUs 0-5 x 6.
- `CUDA_VISIBLE_DEVICES=0-5` on slinky-4, so the throttled GPUs 6-7 are never visible. They are allocated but idle.
- Two per-host srun steps (`code/task_samsep.sh`).
- Logical world 64: 5 slots per rank, 6 pad-slot dummies per step, 38 plan dummies on the last step.
- Never slinky-10: the sbatch aborts if it is in the allocation.

## Memory (H100 80 GB = 79.65 GiB)
Peak per GPU at step 2 and later = 38.0 + 574/W GiB, fitted on:
- 26232 (W64): 45.0 measured;
- 25963 (W24, 3 slots): 62.2 measured, 61.9 fitted;
- 23079 (W16): 51.6 at step 1, without AdamW state.

The 574 GiB is the 38.54B FSDP-sharded parameters x 16 B (fp32 master + grad + m + v).

| layout | peak | verdict |
|---|---|---|
| W14, no offload | **79.0 GiB** | OOM |
| W14 + main-AdamW m/v in pinned host RAM | **~58.5 GiB** | main approved 10-24; bit-identical: the same fused AdamW kernel per 2 GB chunk, asserted in process at steps 1-2 |
| W8 on slinky-18 alone | ~110 GiB (~74 with offload) | no |
| W16 | ~74 GiB | needs slinky-4 GPUs 6-7 (throttled) |

Offload pins 176 GB of host RAM on slinky-18 and 132 GB on slinky-4 (747 GB per node, `--mem=540G`).

## Speed / ETA (estimates)
- Per step: about 6.0-6.6 s per slot (25963: 6.0 s per slot at W24) x 5 slots, + Voxtral 1.5 s + offload transfer about 2-4 s = **about 36 s**.
- 249 steps: about 2.5 h.
- Evals: 16 of them (0, every 25, each export step, the end) at 25 micros per rank, about 100 s each = about 27 min.
- Startup about 6 min; exports about 1-2 min each.
- **ETA about 3.1-3.2 h after start.**

## Saves / exports / HF
- **Saves:** STOP-only (main 10-24), written to `/scratch/amartyavarma/aloe-ft-32-samsep/saves`.
  - Full state = fp32 weights + m/v, 12 B per param: **~271 GB on slinky-18, ~205 GB on slinky-4**. That does not fit.
  - `space_check` then SKIPS the save loudly; STOP still writes `export-stop<N>`. Nothing is pruned before a COMPLETE new save.
- **Exports:** node-local bf16 shards, `/scratch/amartyavarma/aloe-ft-32-samsep/exports/export-{41,83,124,166,208,249}`.
  - Size per export: **44.7 GB on slinky-18 (+ rep / meta), 33.5 GB on slinky-4**, 78.2 GB total.
  - Guard: an intermediate export is written only if every host keeps the floor (30 GB) + one more export of its size free afterwards. The final export is never starved.
  - The final also goes to `/data/.../export-249` (consolidated DCP), only if /data stays above 300 GB free; otherwise it is refused, as in the reference.
- **HF** (`hf_watch.sh`): a nice'd overlap step on both hosts, no GPU, no CPU binding.
  - Each export goes up when its `NODE_COMPLETE` appears: intermediates -> private `InternalCan/ppm-aloe-samsep-16861`; final, stop and /data dcp -> `InternalCan/ppm-aloe-samsep-16861-final`. Never GCS.
  - Intermediates are deleted after VERIFY PASS (`HFLX_DELETE_INTER=1`, main approved), through scratch_guard.
  - Each deletion is logged to `setup-logs/hfb/DELETED-1021-hfb-hold.log` and `logs/BACKUP_LOG.txt`.
- **Demo format:** `gather_final.sbatch` (afterok) builds `export-249-full` on slinky-18. slinky-18's own files are hard links; slinky-4's 6 rank files are pulled and sha256-checked.
- **Watchdog:** `watchdog.sh <job>` as a tiny CPU job.
  - On a 15-min stall: py-spy stacks on both hosts (the NCCL flight recorder is also on), then STOP_AFTER_SAVE, then scancel 20 min later if still hung.
  - No automatic relaunch.

## /scratch now (10-24, during init staging) and stale candidates (caches / test leftovers only; deletion needs main's OK)
- **Free now:** slinky-18 156 GB, slinky-4 94 GB. The 16861 init is 62 / 51 of 88 GB staged and its `-dl` raw dirs are 17 / 19 GB.
- **Free after staging:** about 147 / 76 GB if the -dl raw files are dropped. **At 76 GB, slinky-4 skips every intermediate export** (it needs 97 GB).
- Best candidates, on both hosts:
  - `aloe-samsep-17850-init`: 13 GB, partial, no COMPLETE, base-agent leftover;
  - `aloe-samsep-17850-dl`: 13 GB of raw downloads.
  - Deleting these gives +26 GB per host, enough for every intermediate export (each is deleted after its HF verify).
- Other non-checkpoint candidates (older test / output dirs):
  - slinky-4: `tts_audit/clean` 79 GB (10-15), `ac-out` 39 GB (10-15), `s4qqa/prep` 30 GB (10-15), `uv-cache` 6.8 GB.
  - slinky-18: `ac-out` 32 GB (10-15), `s4qqa` 44 GB (10-15; prep / src / test-build), `s3tput` 8.5 GB (throughput tests).
- Not candidates: checkpoints, weights and datasets, e.g. `xabi-samples` (staged w17641 / w17850 weights), `fastinfer/ckpt`, `qwen-s3-ckpt`, `ppm-qqa-scale/export-*`, `aloe-v2pure-ft`, `*-rows`, `tt-v3` / `tt-pw` rows.

## Open questions / uncertainties
1. **ASR adapter:** asrppm-w15973 w-1441 was trained on w-15973's frozen backbone; the base here is dcp-16861. No newer 32B adapter exists. s is re-calibrated at start (26232's rule; it aborts if the ratio leaves 0.25-4).
2. **NCCL with uneven hosts (8 + 6 GPUs):** supported, but never run in this codebase (26467's T3 was 2 + 2). The first steps will show it.
3. **The offload check and memory are verified only in the real run:** the bit-identity assert at steps 1-2, and the peak in the step lines. There was no GPU smoke, by rule.
4. **Step time** is an estimate (no W14 measurement). The first 3-5 steps give the real number and ETA.
