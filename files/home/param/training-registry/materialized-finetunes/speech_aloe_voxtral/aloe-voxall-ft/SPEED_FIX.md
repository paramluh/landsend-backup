# SPEED FIX (10-16, nodediag) — aloe-voxall-ft

## Why the Voxtral trainers were slow (it was the code, not the "slow" hosts)

1. **Decoder never compiled** (fixed earlier by the builder): the Voxtral port dropped `M.compile_decoder`. Effect: 22315 ran 14.3 s/step vs 21980's 7.6 s/step. 22335 (compile restored) ran ~10 s/step.
2. **Remaining ~2-3 s/step = training forward only.** A/B on the SAME 8 hosts (slinky-0,2,3,5,6,8,9,12), with synced phase timing (`nodediag/ab`):

   | job | trainer | forward | decoder nanosteps | backward | micro |
   |---|---|---|---|---|---|
   | 22341 | 21980 `train_aloe.py` | 1.5-2.0 s | 2.0 s | 3.6-4.1 s | 7.1-8.0 s |
   | 22357 | `train_aloe_m1vox.py` | 3.2-13 s (typ. 4-5) | 2.0 s | 3.6-4.1 s | 9.7-12 s |

   - GEMM and attention GPU time were identical (2.88 vs 2.88 s per step); the extra time is the GPU idling on NCCL waits.
   - Eval forward (micros built in advance, no loader activity) was NOT slower.
   - Hosts are fine: 21980 ran at full speed on them. slinky-4/7 bench: matmul 720-800 TFLOPs, NCCL 2-node ag 308 / rs 371 / ar 454 GB/s. The static probe of all 20 nodes was identical.
3. **Cause (strong circumstantial evidence; no direct GPU confirmation was run because PPM holds the hosts):**
   - During every forward pass, the DataLoader workers prefetch the next row with `audio_s4q.ffmpeg_row_audio`.
   - That runs ffmpeg on the WHOLE ~23 min 48 kHz stereo WLT segment FLAC to cut a 10 min row: ~0.85 s wall and ~1.9 CPU-s per row (slinky-4), plus a ~134 MB pipe read and a 57.6 MB IPC copy.
   - This happens on the rank's own 5 logical CPUs = 2.5 physical cores (HT pairs 0-1, 2-3, ...), already ~190% busy with the main thread and NCCL proxy threads.
   - 21980's loader only reads small precomputed arrays.

## What changed (engineering only; the training data is bit-identical)

- **Row cache** — `code/rowcache_vox.py` + `tools/stage_rowcache.py`:
  - Staging runs the trainer's exact `ffmpeg_row_audio` once per row (same binary, command, slice, pad and gain) and stores the float32 result as `.npy` under `ALOE_ROWCACHE` (node-local).
  - The data module's `conv` now calls `RC.row_audio`, which returns the `.npy` when present and otherwise decodes on the fly. Either way the array is bit-identical: dtype and shape are asserted, and `--verify` re-decodes cached rows and compares bytes.
  - The cache key covers FLAC path + size, frame0, n_frames, gain and the ffmpeg binary, so a stale hit is impossible.
- **Disk guard (important):**
  - A full node needs **~120 GB** (m1vox, measured from the plan: ~2,980 rows of up to 57.6 MB, ~41 MB on average); vox / voxall should be similar or larger.
  - The builder stops at `ROWCACHE_MIN_FREE_GB` (default 120; the user's rule keeps >= 100 GB free for the local exports).
  - On disks as full as today (140-185 GB free before exports), expect **partial coverage**. Uncovered rows decode on the fly (correct, slower). Freeing /scratch first raises coverage.
  - Coverage is in `<ALOE_ROWCACHE>/LEDGER_node<i>.json` and in the training sbatch pre-flight line (`rowcache coverage`).
- **CPU headroom:**
  - The training sbatch now has `#SBATCH --cpus-per-task=8` (4 physical cores per rank, = PPM main) and ABORTS if launched with fewer.
  - The launchers pass 8.
  - DataLoader `worker_init_fn` calls `os.nice(19)`; ffmpeg children inherit it, so the main/NCCL threads win the CPU.
- `ALOE_ROWCACHE=/scratch/amartyavarma/aloe-voxall-ft/rowcache` is set in the run's env file.
- `launch.sh` here was a byte-identical STALE copy of aloe-vox-ft/launch.sh (V=aloe-vox-ft), so running it would have launched the aloe-vox-ft run. It now aborts with that message. Launch voxall with `aloe_voxall.sbatch`.
- The decoder compile is already restored (`code/train_aloe_vox.py:766`; POSE_B=0, so dec_s only).

## How to use

`tools/prep_node.sh` now runs `tools/stage_rowcache.py --node K` after the store and FLACs are staged (log `logs/prep/rowcache-p<K>-<host>.log`). To build it alone on node K: `ALOE_ROWCACHE=/scratch/amartyavarma/aloe-voxall-ft/rowcache python -I tools/stage_rowcache.py --node K --workers 32`. Then launch as before; the pre-flight line prints each node's cache coverage.

## Verification (10-16, slinky-4, CPU only)

- `py_compile` passes on every edited .py file; `bash -n` passes on every edited sbatch/sh file; `pyflakes` is clean on the new tools.
- `tools/stage_rowcache.py` (m1vox) was run end-to-end on the first 3 real node-0 rows (FLACs fetched from GCS at the manifest generation): built 3, `--verify` 3/3 bit-identical.
- Each run dir's own `rowcache_vox` + `audio_s4q` copy: cache hit == on-the-fly decode, bit for bit, on all 3 rows (14.4 M / 7.06 M / 14.4 M samples). Load takes 0.01-0.02 s vs 0.49-0.55 s to decode. The no-cache miss path is also identical.
- Not run: any GPU / training run (PPM holds the hosts). The first real launch will show it: forward should drop to ~1.8 s and step time to ~7.6 s.
- Script: `/data/home/amartyavarma/ppm/nodediag/speedfix/cpu_check.sh` (log `cpu_check.log`).

Backups of every edited file: `*.pre-speedfix` next to it. A/B logs: `/data/home/amartyavarma/ppm/nodediag/ab/out/` (ab-22341.log, ab-22357.log, prof-*.txt).
