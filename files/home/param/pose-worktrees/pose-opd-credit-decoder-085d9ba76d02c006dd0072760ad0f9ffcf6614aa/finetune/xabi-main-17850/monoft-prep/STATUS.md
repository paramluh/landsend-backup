# MONOFT: monoplex FT on 8 main hosts (prep only)

Updated 10-22 22:18 PT by MONOFT (tmux `monoft`). User 10-22 ~21:55 PT: "fine-tune on the monoplex data on 8 nodes ... pause the main run and use 8 of its nodes".
Prep only. Nothing has been paused or launched.

## State: DONE 23:36 PT. FT 27757 exit 0; xfree / HF final / cleanup all COMPLETED 0:0.
- **Result (step-200 eval):** pose cb0 **1.5788**, pose light **3.6384**.
  - 17641 FT: 1.5811 / 3.6438. 17352 FT: 1.5849 / 3.6530.
  - Best point of the series so far.
- Final export-200: private HF `InternalCan/ppm-aloe-v2pure-main-17850-final`, VERIFY PASS on 8/8 hosts. It is also kept node-local (11 GB/host).
- The five intermediate exports are on private HF `InternalCan/ppm-aloe-v2pure-main-17850`; their node copies are freed.
- The node-local init copies were deleted (dcp-17850 is still COMPLETE everywhere).
- The decay branch 27810 then started on all 16 hosts.
- Earlier: launched on the user's go via the main session, 22:19 PT.
- Main run 27653 PAUSED at step **17850** (22:29:38 PT). The stop save WAS the hourly save (save-mode alignment worked: one save, dcp-17850, no extra).
- Init job 27756: rc 0, 8/8 FT hosts load_init-checked COMPLETE (14 GB each), 22:29:49 → 22:42:43 PT.
  - It took 13 min instead of ~4: torch oversubscribed threads inside the -c 16 cpuset (60 s per load_init rank vs 6 s in the dry run). Fixed in tools/init_dist.sh (OMP/MKL threads = cpus per task) for any next launch.
- Run dir aloe-v2pure-main-17850. Chain: xfree 27758, HF final 27759, cleanup 27760. The decay branch 27810 (decay agent) waits afterany on the whole FT chain, then takes all 16 hosts.
- samdata_ft point p17850 registered. Its x (pretraining eval at exactly 17850) arrives with the decay branch's / main run's resume eval at 17850.

## What it is
- The FT is aloe-v2pure-17641 (job 27622) run VERBATIM; only the init changes. That is the 26089 v2pure recipe: 200 steps x 64 rows, world 64 = 8 hosts x 8 ranks, LR 1e-5 schedule, alpha 1.0, the 1/3 clip rule, link on.
- Init: the main run's (ppm811 job 27653) save dcp-S at the pause.
- Run dir: `ppm/aloe-v2pure-main-<S>`. launch.sh creates it from `ppm/monoft/template` and a byte-identical copy of 17641's `code/`.
  - `CFG_DIFF.txt`: run.env differs from 17641's only in QQA_RUN / LOG_ROOT / PLAN_DIR / CONTROL_DIR / EXPORT / ALOE_LOCAL_EXPORT_ROOT / ALOE_SAVE_ROOT / QQA_REF_CFG (paths and names). launch.sh aborts on any other difference, or if code/ differs.
- Final resumable save: OFF. No FT host has the 144 GB (+7) it needs after the init. Same as the 17352 / 17641 FTs; ALOE_SAVE_EVERY_MIN is 100000 as in 17641.
- Chain: as ftb_chain.sh, all with `--kill-on-invalid-dep=yes`.
  - init job → FT (afterok)
  - xfree (after): the 5 intermediate exports go to private HF `InternalCan/ppm-aloe-v2pure-main-<S>`, and the node copies are freed
  - HF final export (afterok): `InternalCan/ppm-aloe-v2pure-main-<S>-final`. Never GCS.
  - cleanup (afterok): the FT's own saves, plus the node-local init copy, deleted only while dcp-S is still COMPLETE on that host.

## /data problem: solved without touching /data
- No 83 GB `ckpt-infer/local-<S>` copy is made. Nothing on /data is written except KB-size manifests and logs. No deletion is needed.
- New init layout: node-local on each FT host, `/scratch/amartyavarma/aloe-v2pure-main-<S>-init`.
  - Each FT host holds only the source shards its 8 FT ranks actually read: 18-20 of the 128 files plus rank0, rep.pt and meta.json. That is 13-14.4 GB per host.
  - The needed set is computed exactly: torch's Shard(0) rows × load_init's own source loop.
- Tools in `ppm/monoft/tools`, run from a frozen copy in the run dir:
  1. Every one of the 16 hosts converts its own 8 ranks to bf16 (the ckpt_local_export maths verbatim, including the slinky-7 → slinky-10 slot alias) into a node-local stage dir, with sha256 sums.
  2. Each FT host pulls the shards it needs over HTTP on the pod network (ssh between nodes is refused) and verifies each sha256.
  3. Each FT host runs the trainer's **real `load_init`** on CPU for all 8 of its FT ranks. It also checks that the key set and every global shape equal local-17641's, which 27622 loaded.
  4. Each FT host writes COMPLETE; then every stage dir is deleted.
- dcp-S itself is only read, never moved or linked. It stays intact on all 16 hosts as the main run's resume point.

## Hosts (/scratch free measured 22:00 PT with 2 saves + dcp-16861 + keep-ft present)
- **FT (8):** slinky-7, 8, 9, 12, 14, 16, 17, 19 (`FT_HOSTS.txt`).
  - Free: 119 / 103 / 94 / 125 / 135 / 99 / 92 / 154 GB.
  - Rank 0 is slinky-7 (119 GB): slinky-5 was swapped out for slinky-17 so the rank-0 host, which needs +7 GB, is a roomy one.
  - After the 14 GB init, the minimum free is 78 GB (slinky-17), against a need of 65 (72 on slinky-7).
- **Other 8 (LR-decay branch):** slinky-0, 1, 2, 3, 5, 6, 11, 13. Free: 81 / 82 / 83 / 92 / 94 / 91 / 85 / 90 GB.
- If the decay branch already holds the other 8 hosts at launch time, launch.sh detects it. The conversion of their shards then runs as an `--overlap` CPU step of that job, so no wait.

## How the main session runs it (on the user's go)
- `bash /data/home/amartyavarma/ppm/monoft/pause_main.sh [save|now|eval] --launch default 0`. Run it in the background.
- Or in two steps: `pause_main.sh <mode>`, which prints S, then `launch.sh <S> default 0`.
- pause_main.sh uses STOP_AFTER_SAVE (never scancel). The trainer saves dcp-S at the step boundary and exits cleanly, and the watchdog does not relaunch after a deliberate stop. Modes:
  - `save` (default; "right after an hourly save"): the stop save is timed to BE the hourly save, so no extra 43 GB/host save and no extra HF backup. Waits ≤ 60 min.
  - `now`: S = the next step boundary, about 1-2 min.
  - `eval`: the stop is timed with the 30-min eval timer, so the main run logs "EVAL step S" right before the save. The scaling page needs that x value at exactly S; without it the point waits for the eval at the main run's resume from dcp-S. Waits ≤ 30 min.
- launch.sh returns in seconds. Logs: `<run dir>/logs/init_convert-<job>.log`, then `logs/aloev2pm<S>-<job>.log`. Job ids: `<run dir>/LAUNCH.json`.

## ETA from launch
| step | time |
|---|---|
| init job (16 hosts free) | ~3-4 min |
| FT (27622 took 51 min wall) | ~51 min |
| HF final export | ~3 min |
| **launch → final eval** | **~55 min** |
| **launch → HF-verified final** | **~60 min** |

The pause itself adds ~2 min (`now`), ≤ 30 min (`eval`) or ≤ 60 min (`save`).

## Scaling page (samdata_ft.html)
- Points are NOT auto-discovered.
- `tools/samdata_register.py <S> aloev2pm<S>` (run by launch.sh) registers point p<S> in the live builder and template.
- It is idempotent, keeps backups and restores them on failure. Tested in dry mode: the patched builder runs end to end and `node --check` passes.
- The y value comes at the FT's step-200 eval. The x value needs a main-run eval at exactly S (see the `eval` mode above).

## Dry run (10-22 22:05-22:15 PT, main run training, overlap step on 27653; CPU only, nice / ionice idle, fetch capped)
- Source: the current save dcp-17798. FT hosts: as above.
- Conversion: 14-22 s per host. Fetch (capped at 2-3 × 100 MB/s): 59-132 s.
- `load_init` CHECK PASS on all 8 FT hosts × 8 ranks (64/64). Key set and global shapes are identical to local-17641's (1,522 sharded + 628 replicated tensors).
- Stage and init were deleted afterwards. /scratch free is back to its starting values on all 16 hosts.
- The main run kept 68.2-68.8 s/step throughout.
- The first pass was rerun because the script was edited mid-run; launches now run a frozen copy of the tools.
- Second, clean pass (dryrun/17798/dry2.log): **rc 0** in 3 min 40 s even with the throttle; 8/8 FT hosts FETCHED and CHECK PASS; every stage and init deleted; /scratch free identical before and after on all 16 hosts.
- launch.sh rehearsal (`MF_TEST=1`, nothing submitted): the run dir rendered, CFG_DIFF is clean, and the full chain of sbatch commands is in `dryrun/launchtest/launch.log`.

## Log
- 10-22 21:57 PT: started.
- 22:12 PT: dry run 1. All 64 load_init checks passed; it ended in a shell error caused by the mid-run edit.
- 22:12 PT: dry run 2 started (frozen tools). launch.sh rehearsal OK.
- 22:16 PT: dry run 2 rc 0. The `save`-mode pause window rehearsal (`PAUSE_DRY=1`, touches nothing) is running until the 22:28 PT hourly save timer: dryrun/pause_dry_save.log.
- 10-22 22:29:48 PT: LAUNCHED aloe-v2pure-main-17850 (S 17850): init job 27756 (all16), FT slinky-7,slinky-8,slinky-9,slinky-12,slinky-14,slinky-16,slinky-17,slinky-19, final save 0; chain xfree / HF final / cleanup submitted (ids in /data/home/amartyavarma/ppm/aloe-v2pure-main-17850/LAUNCH.json)
- 22:19 PT: main session started pause_main.sh save --launch default 0 (user's go). 22:23 PT: my PAUSE_DRY rehearsal stopped (main's request; it touched nothing).
- 22:29 PT: paused at 17850; launched. 22:42 PT: init rc 0, FT 27757 started.
- 23:34 PT: FT step-200 eval pose cb0 1.5788 / light 3.6384; 23:36 PT chain complete (HF final verified 8/8, cleanup done); decay branch 27810 running.
