# /scratch deletion candidates for the monoplex SAM FT (monosam, 10-26 ~05:50 PT)

READ-ONLY survey. Nothing has been deleted, moved or modified, and no job was touched. This list is for main to approve. Deleting needs main's relayed approval; the protected list and the user's path-specific rule still apply.

**Units.** Every number is **GiB (2^30)**, the same unit as `df -BG`. Sizes for SAFE and CHECK paths are **hard-link-aware freeable bytes**: allocated blocks of files whose every link lies inside the deleted set. They were measured on each host. du totals overstate some dirs; see the traps below. Targets are taken in GiB, which is stricter than the coordinator's GB: 240 GiB = 258 GB and 300 GiB = 322 GB.

## Answer
- **Option A, 12 hosts at ≥ 240 GiB:** slinky-15, 14, 5, 7, 1, 2, 17, 3, 8, 0, 19, 6. All 12 reach the target with SAFE deletions only. The minimal pick is 1,553 GiB; all SAFE on those hosts is 2,624 GiB.
  - **slinky-3 is tight:** 109 → 243.7, and it needs every SAFE row on the host.
  - **Swap with more margin:** slinky-13 (73 → 328.5 with all SAFE; needs 167). It was not chosen because it needs more deletion.
- **Option B, 8 hosts at ≥ 300 GiB:** slinky-15, 14, 5, 7, 1, 2, 17, 6. The minimal pick is 1,322 GiB.
  - slinky-13 also reaches 328.5 and is the only other feasible host.
- **Cannot reach 240 GiB with SAFE only:** slinky-11 (239.8, short by 0.2), slinky-12 (237.4), slinky-16 (226.6). All three get there if some CHECK rows are approved.
- **For 300 GiB,** only 9 hosts are feasible: the 8 above plus slinky-13.
- **SAFE total:** 3,295 GiB over all 16 hosts.

## Method
1. **Inventory.** Tiny probes inside MoT 30289: `srun --overlap -N16 --ntasks-per-node=1 -c1 --mem=256M --cpu-bind=none`, du under `nice -n 19 ionice -c3` with `timeout 60` per dir.
   - Protected dirs were skipped entirely: s3-rows*, pose-s3-samdata*, ppm, ppm-next, ppm-ilp, ppm-mot, *keep*, hfb/ppm-s3-rows-* and si20k-podcast-flac.
   - MoT stayed at about 21.0 s/step throughout.
   - Five read-only passes: top-level du, depth-3 du with mtimes, a large-file hard-link scan, the hard-link-aware freeable pass, and a pass that runs `scratch_guard.sh guard_check` on every SAFE and CHECK path on its own host.
   - guard_check results: 434 OK and 1 refused (slinky-3 `triton-cache-q1ch`, whose hash-named entry contains "KEEP"; now KEEP).
2. **Classification.** Each path was checked against run dirs and logs, HF verify JSONs (setup-logs/hfb/verify, run-dir hf_verify and backup logs, ppmvox companions/logs/hfstate-1025), /data copies, squeue and scontrol, and the envs and scripts of the running services. Main's earlier deletion ledgers were used as precedent: ppmvox/launch/DELETED-1025.log and control/OPS_PLAYBOOK.md.
   - **SAFE:** a redundant copy (/data or HF VERIFY PASS), a cache, or a finished smoke.
   - **CHECK:** likely redundant, but the proof is missing or the backup is GCS-only.
   - **KEEP:** unique, protected, or in use.
3. **Host choice.** From MoT's 16 hosts (never 4, 9, 10 or 18), take the hosts whose free plus SAFE meets the target, ranked by the deletion needed (target − free).
   - **Minimal pick per host:** non-state items first, largest first, then full training states (saves and dcps), largest first, stopping once the target is met. Hard-link pairs are always kept together.
   - The `pick` column of the TSV marks the rows: `12` and/or `8`.

## Deletion traps (read before approving anything)
- **qwen-s3-ckpt/q32-s3-exp1-duplex-sp-split2-glu-r2-j8333/dcp-750:** about 29 GiB on 12 hosts, KEEP.
  - guard_rm would NOT refuse it. The KEEP_dcp-750 marker is a sibling file, and its text says "Main run resumes from dcp-750 here (protected ... do not delete)". Only w-750 weights are on Weka; the full state is not on HF.
- **Hard links of protected keep dirs free 0 GiB.** du credits their size to the non-keep path on some hosts because of traversal order:
  - `qwen-s3-ckpt/q32-s3-exp1-lr0.3-from750-j8424/dcp-1250` is hard links of the protected `keep-lr0.3-dcp-1250`.
  - `ppm14b/ckpt/ppm14b/dcp-4304` is hard links of the protected `ppm14b/ckpt-keep/dcp-4304`, except on slinky-7, where it is a real copy that is SAFE.
- **Hard-link pairs (delete both or neither):**
  - slinky-8: `aloespk-init/export-249` + `aloe-ft-32-samsep/exports-vox17127/export-249`. 9.0 GiB is shared, counted once.
  - slinky-11 and 13: `si20k-tin-s3/stores` + `si20k-vox/stores`. 12.1 GiB is shared.
  - slinky-7: `conv-pool-v2` frees only 5.0 GiB alone; 28 GiB of it is hard links of the CHECK tt-s3/tt-v3 rows.
- **`ppmvox` and `ppmvox/ckpt` cannot be deleted as units:** they contain the protected `v5-18100-merged-keep`. Use the listed subpaths.
- **`new-pose-llm-data` (~31 GiB per host) and `npl-append` (~35 GiB per host) are KEEP.** They are unique per-host shards (Weka has only a 71 GB partial) and the row source for paused ppmvox v5b and the 8,1,1 decay env. The 10-25 approved clean list kept them.
- **`inductor-cache-ppm` and `triton-cache-ppm` are KEEP:** MoT's `task_ppm.sh` uses them, and the FT uses the same defaults.
- **Full training states in the SAFE set.** Each has a sha256-verified full-state HF copy, so deleting turns a local resume into an HF download:
  - ppmvox v5b `dcp-18163`: HF `InternalCan/ppmvox-v5b-18163-state`, 8/8 slots; the slinky-8 slot is already HF-only.
  - samsep `dcp-249`, both runs: per-host HF VERIFY PASS.
  - poseduplex `dcp-498`: HF `ppm-aloe-poseduplex-v5-498-state`, sha256all on 8/8.
  - The minimal picks use these only where the host needs them.
- **`hfb/xet-cache`:** delete only when no HF upload is in flight on that host. scratch_cleanup.sh skips it if an upload process is alive.
- **CHECK, needing main's decision:**
  - `ppm14b/ckpt/ppm14b/dcp-4502` (20.7 GiB per host on 12 hosts): no export or HF copy (rule: never delete unexported ckpts). However, the set is already non-resumable, because the s5, s8 and s19 slots were deleted 10-25 under the "ppm14b old dir" approval.
  - GCS-only backups: `aloe-pose-ft/exports` (s12, s16: 30 GiB each), `aloe-uncond-ft/exports`, `qqa-gen` and `ac-out`.
  - Built rows with no copy: `ppmvox/vpod3/rows`, `ppmvox/v5b-vpod`, `tt-s3`, `tt-v3`, `tt-n1`.
  - Copies whose origin was not verified: `ttconvo-ft/stores/slinky-18` and `ttconvo-ft/stores/slinky-4`.

## Files
- **`go/scratch_candidates.tsv`:** one row per host and path (601 rows: 280 SAFE, 154 CHECK, 166 KEEP). Columns: `host, path, GiB, class, proof, pick`.
- **`go/scratch_cleanup.sh`:** staged and never run. Defaults to `--dry-run`, which prints only and runs no srun.
  - `--option 12|8`, `--hosts`, `--all-safe`, `--jobid`.
  - `--execute` needs `MAIN_GO=1`.
  - It deletes SAFE rows only, through `scratch_guard.sh guard_rm`, with 1 CPU, `--mem=512M` and nice/ionice idle on each host. It re-checks the guard on the node first and logs to `go/SCRATCH_CLEANUP.log`.
  - Example: `bash go/scratch_cleanup.sh --option 12` (dry run).
- The FT itself needs nothing on /scratch except its own `aloe-pose-mono-8020-sam` dir. Its init and stores are on /data.

### Option A: 12 hosts (world 96), target >= 240 GiB free

| host | free now | SAFE available | free after all SAFE | minimal pick | free after pick |
|---|---:|---:|---:|---:|---:|
| slinky-15 | 630 | 315.7 | 945.7 | 0.0 | 630.0 |
| slinky-14 | 221 | 208.6 | 429.6 | 29.5 | 250.5 |
| slinky-5 | 172 | 196.1 | 368.1 | 135.8 | 307.8 |
| slinky-7 | 143 | 376.3 | 519.3 | 103.0 | 246.0 |
| slinky-1 | 122 | 208.0 | 330.0 | 147.7 | 269.7 |
| slinky-2 | 118 | 205.9 | 323.9 | 145.6 | 263.6 |
| slinky-17 | 113 | 225.7 | 338.7 | 164.7 | 277.7 |
| slinky-3 | 109 | 134.7 | 243.7 | 134.7 | 243.7 |
| slinky-8 | 103 | 166.0 | 269.0 | 166.0 | 269.0 |
| slinky-0 | 101 | 175.5 | 276.5 | 175.5 | 276.5 |
| slinky-19 | 95 | 170.0 | 265.0 | 170.0 | 265.0 |
| slinky-6 | 91 | 241.2 | 332.2 | 180.9 | 271.9 |
| **total** | | **2623.7** | | **1553.4** | |

### Option B: 8 hosts (world 64), target >= 300 GiB free

| host | free now | SAFE available | free after all SAFE | minimal pick | free after pick |
|---|---:|---:|---:|---:|---:|
| slinky-15 | 630 | 315.7 | 945.7 | 0.0 | 630.0 |
| slinky-14 | 221 | 208.6 | 429.6 | 148.3 | 369.3 |
| slinky-5 | 172 | 196.1 | 368.1 | 135.8 | 307.8 |
| slinky-7 | 143 | 376.3 | 519.3 | 158.0 | 301.0 |
| slinky-1 | 122 | 208.0 | 330.0 | 208.0 | 330.0 |
| slinky-2 | 118 | 205.9 | 323.9 | 205.9 | 323.9 |
| slinky-17 | 113 | 225.7 | 338.7 | 225.0 | 338.0 |
| slinky-6 | 91 | 241.2 | 332.2 | 241.2 | 332.2 |
| **total** | | **1977.5** | | **1322.2** | |

### All 16 MoT hosts (GiB)

| host | free now | SAFE | CHECK | free after SAFE | free after SAFE+CHECK | in A | in B |
|---|---:|---:|---:|---:|---:|:-:|:-:|
| slinky-0 | 101 | 175.5 | 49.5 | 276.5 | 326.0 | Y |  |
| slinky-1 | 122 | 208.0 | 50.7 | 330.0 | 380.7 | Y | Y |
| slinky-2 | 118 | 205.9 | 35.3 | 323.9 | 359.2 | Y | Y |
| slinky-3 | 109 | 134.7 | 58.5 | 243.7 | 302.2 | Y |  |
| slinky-5 | 172 | 196.1 | 27.7 | 368.1 | 395.8 | Y | Y |
| slinky-6 | 91 | 241.2 | 39.1 | 332.2 | 371.3 | Y | Y |
| slinky-7 | 143 | 376.3 | 74.3 | 519.3 | 593.6 | Y | Y |
| slinky-8 | 103 | 166.0 | 29.3 | 269.0 | 298.3 | Y |  |
| slinky-11 | 100 | 139.8 | 79.2 | 239.8 | 319.0 |  |  |
| slinky-12 | 91 | 146.4 | 78.4 | 237.4 | 315.8 |  |  |
| slinky-13 | 73 | 255.5 | 68.5 | 328.5 | 397.0 |  |  |
| slinky-14 | 221 | 208.6 | 32.7 | 429.6 | 462.3 | Y | Y |
| slinky-15 | 630 | 315.7 | 13.2 | 945.7 | 958.9 | Y | Y |
| slinky-16 | 97 | 129.6 | 87.5 | 226.6 | 314.1 |  |  |
| slinky-17 | 113 | 225.7 | 33.3 | 338.7 | 372.0 | Y | Y |
| slinky-19 | 95 | 170.0 | 36.2 | 265.0 | 301.2 | Y |  |

### SAFE paths (summed over hosts; per-host rows in scratch_candidates.tsv)

| path (under /scratch/amartyavarma) | GiB total | hosts (slinky-N) | proof |
|---|---:|---|---|
| `aloe-ft-32-samsep/saves-poseduplex/ppm-aloe-poseduplex-v5-j30074/dcp-498` | 584.8 | 1,2,6,7,13,14,15,17 | End full save of finished pose-duplex job 30074 (498/498, training exit 0); private HF InternalCan/ppm-aloe-poseduplex-v5-498-state tag dcp-498-slinky-N, VERIFY PASS + sha256 of all 21 files == local, 78.45 GB per host (/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498/*dcp-498-slinky-N*.json); no hardlink farm left in hfb. |
| `ppmvox/ckpt/ppmvox-v5b/dcp-18163` | 520.8 | 0,3,5,11,12,16,19 | v5b full state (step 18163, all files nlink 1) HF-verified incl. sha256 of every file on all 8 slots: InternalCan/ppmvox-v5b-18163-state (/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025/InternalCan__ppmvox-v5b-18163-state__dcp-18163-slinky-*{,-sha256all}*.json, PASS, ok-* markers); s8 slot already HF-only; v5b STOPPED clean 10-25 05:10 (ppmvox/logs/ppmvox16-30104.log), no job uses it. CAVEAT: newest r ... |
| `aloe-ft-32-samsep/saves-persona/ppm-aloe-samsep-vox17127-persona-j29444/dcp-249` | 361.8 | 6,7,13,14,15,17 | End full save of finished job 29444 (249/249, exit 0, persona/STATUS.txt DONE); per-host private HF InternalCan/ppm-aloe-samsep-vox17127-persona-saves/dcp-249__slinky-N VERIFY PASS (/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_verify/InternalCan__ppm-aloe-samsep-vox17127-persona-saves__dcp-249__slinky-N__slinky-N.json); s16/s19 copies already deleted 10-25 (GUARD_LOG.md:165-167). |
| `aloe-ft-32-samsep/saves/ppm-aloe-samsep-vox17127-j29345/dcp-249` | 301.5 | 0,1,2,5,8 | End full save of finished job 29345 (249/249, exit 0, logs/aloesamsep-29345.log); per-host private HF InternalCan/ppm-aloe-samsep-vox17127-saves/dcp-249__slinky-N VERIFY PASS n_missing 0 (/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_verify/InternalCan__ppm-aloe-samsep-vox17127-saves__dcp-249__slinky-N__slinky-N.json); same dir already deleted on s3/11/12 10-25 under main GO (mot/GUARD_LOG.md:159-163). |
| `ppmvox/vpod3/stems` | 142.2 | 0,1,2,5,6,7,13,14,15,17,19 | rolling user-stem cache range-read from HF bucket InternalCan/podcast-raw-audio-data-mass by data_tools/vpod3_host.py / v5b_vpod_daemon (re-fetchable); s11 copy deleted 10-25 05:08 as 'rolling stem cache, main -f5 go' (DELETED-1025.log); no daemon running |
| `poseduplex-init/w-18100` | 119.3 | 1,2,6,7,13,14,15,17 | Staged init subset (host-to-host copy, sha256 == SHA256SUMS) of PPMVOX v5 export w-18100 for finished job 30074; private HF InternalCan/ppmvox-v5-17850 w-18100 VERIFY PASS + sha256all on 16/16 hosts (setup-logs/hfb/verify/InternalCan__ppmvox-v5-17850__w-18100*); no hardlinks (checked slinky-7). |
| `aloevox-init/w-17127` | 88.7 | 6,7,13,16,17,19 | Staged init subset for finished 29345/29444, pulled from private HF InternalCan/ppmvox-v3-16861 w-17127 (sha == LFS; aloe-ft-32-samsep/persona/stage_init.sbatch); HF VERIFY PASS + sha256all 16/16 (setup-logs/hfb/verify/InternalCan__ppmvox-v3-16861__w-17127*); s14/s15 copies deleted 10-24 (GUARD_LOG.md:78). |
| `aloe-ft-32-samsep/exports-poseduplex-v5/export-498` | 82.2 | 1,2,6,7,13,14,15,17 | Node-local final export shards; per-host private HF InternalCan/ppm-aloe-poseduplex-v5-final/export-498 VERIFY PASS (/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/hf_verify/InternalCan__ppm-aloe-poseduplex-v5-final__export-498__slinky-N.json) + consolidated export-498-dcp PASS; live posedup demo reads its own copy /scratch/amartyavarma/aloe-posedup-live/export-498-dcp on slinky-10 (aloe-posedup-live/L ... |
| `ppmvox/exports/ppmvox-v5/w-18100` | 64.8 | 0,1,2,6,7,13,14,15,17 | per-host bf16 export w-18100 HF VERIFY PASS + sha256all PASS on all 16 hosts -> InternalCan/ppmvox-v5-17850/w-18100 (/data/home/amartyavarma/setup-logs/hfb/verify/InternalCan__ppmvox-v5-17850__w-18100*__slinky-N.json); same copies already deleted on s5/s11/s19 with main go (ppmvox/launch/DELETED-1025.log); no job reads it |
| `aloe-zoom-cotrain/exports/export-253` | 59.6 | 0,1,2,3,5,6,8,11,12,14,16,17,19 | Node-local final export; per-host private HF InternalCan/ppm-aloe-zoom-cotrain export-253 VERIFY PASS on all 16 hosts (ppm/aloe-zoom-cotrain/logs/hf_verify) + gathered copy (logs/GATHER_VERIFY.json). |
| `ppmvox/vpod4` | 59.6 | 0,1,2,3,6,8,11,12,13,14,15,19 | partial (13 batches) podcast store built 10-24 for the superseded ppmvox-v3x run only (ppmvox_16n_v3x.env); not referenced by v5/v5b envs; derived from HF bucket via data_tools/vpod4_host.py; no job uses it |
| `ttconvo-ft/exports/export-285` | 57.2 | 0,1,2,3,5,11,12,13,16,17,19 | [freed GB, hard-link aware: 0:7.7 1:5.4 2:5.4 3:5.4 5:5.4 11:5.4 12:5.4 13:5.4 16:5.4 17:5.4 19:5.4] final export of finished PPM FT ttconvo-ft (job 26510): private HF InternalCan/ppm-ttconvo-ft/export-285 VERIFY PASS on all 16 hosts (ppm/ttconvo-ft/hf_backup/HF_BACKUP-export-285.json: 131 files, 0 missing/size/sha mismatch; local sizes match); same dir deleted on slinky-6/8 10-22 per ledger |
| `aloe-v2pure-17352/exports/export-200` | 50.0 | 3,5,7,12,13 | Node-local final export; per-host private HF InternalCan/ppm-aloe-v2pure-17352-final/export-200 VERIFY PASS (ppm/aloe-v2pure-17352/logs/hf_verify). |
| `hfb/xet-cache` | 49.8 | 0,1,2,3,5,6,7,8,11,12,14,15,16,17,19 | hf_xet client chunk cache (only https___cas_serv*/shard-cache + logs); no running/pending job on these hosts except MoT (MoT uses HF_HOME on /data, state-backup uses /data xet caches); delete only when no HF upload step is in flight on the host |
| `si20k-tin-s3/stores` | 48.4 | 11,13,14,19 | HARDLINK GROUP: pair with si20k-vox/stores (same 44 files); delete both, 12.1 GiB counted here. [freed GB for the PAIR (both dirs deleted together), counted on this line: 11:13.0 13:13.0] delete TOGETHER with si20k-vox/stores (partner line frees 0 alone); hard-link pair (44 files, 13.0 GB on disk; deleting only one frees 0): node-local copy of /data/home/amartyavarma/qwen-1ch-data/si20k-vox/stores (13.01 GB, STORES.j ... |
| `aloe-ft-32-samsep/exports-vox17127/export-249` | 46.0 | 1,2,3,5,8,11 | Node-local final export of 29345; per-host private HF InternalCan/ppm-aloe-samsep-vox17127-final/export-249 VERIFY PASS (/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_verify/InternalCan__ppm-aloe-samsep-vox17127-final__export-249__slinky-N.json) + /data copy aloe-ft-32-samsep/vox/export-vox17127-249 (export-249-dcp HF PASS); s0/s12 copies already deleted 10-24 (GUARD_LOG.md:78). |
| `aloe-ft-32-samsep/exports-vox17127-persona/export-249` | 46.0 | 6,7,16,17,19 | Node-local final export of 29444; per-host private HF InternalCan/ppm-aloe-samsep-vox17127-persona-final/export-249 VERIFY PASS (/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_verify/...persona-final__export-249__slinky-N.json) + /data copy persona/export-vox17127-persona-249 (that /data copy is the one aloedemo loads; s13/14/15 node copies already deleted 10-24). |
| `ppmvox/init-17850-slots` | 42.9 | 15 | staged copy (STAGED_slinky-9.json) of 8,1,1 dcp-17850 slot slinky-9 = HF InternalCan/ppm-v1-c dcp-17850/slinky-9 VERIFY PASS (/data/home/amartyavarma/setup-logs/hfb/verify/InternalCan__ppm-v1-c__dcp-17850__slinky-9.json, 46.1 GB); only a fresh ppmvox init reads it (PPMVOX_INIT_FROM), resumes ignore init |
| `aloe-v2pure-main-17850/exports/export-200` | 42.2 | 7,8,12,19 | Node-local final export; per-host private HF InternalCan/ppm-aloe-v2pure-main-17850-final/export-200 VERIFY PASS (ppm/aloe-v2pure-main-17850/logs/hf_verify) + consolidated /data copy ppm/aloe-v2pure-main-17850/export-200 (66 files, COMPLETE). |
| `aloespk-init/export-249` | 39.2 | 8 | HARDLINK GROUP: delete together with aloe-ft-32-samsep/exports-vox17127/export-249 (8 rank files shared, 9.0 GiB counted here). Init staged 10-24 08:08 for the Ashley 2-node speaker FT that was cancelled (speaker/STATUS.txt 08:04/08:28; never run); 27 files from private HF InternalCan/ppm-aloe-samsep-vox17127-final export-249 (sha == LFS, STAGE.json) + 8 hardlinks of exports-vox17127/export-249 + meta.json (copy on / ... |
| `aloe-m1vox-ft/exports/export-366` | 36.0 | 3,6,8,12 | Node-local final export; per-host GCS VERIFY PASS (ppm/aloe-m1vox-ft/logs/gcs_verify) + consolidated /data copy ppm/aloe-m1vox-ft/export-366 (COMPLETE); (aloe-m1vox-ft/flac is a symlink to /data: do not target it). |
| `s4adapter` | 31.8 | 3,6,7,8,11,12,13 | [freed GB, hard-link aware: 3:3.4 6:3.4 7:3.4 8:3.4 11:6.8 12:6.8 13:6.9] node-local staged copy of /data/home/amartyavarma/qwen-1ch-data/s4adapter (mls 23G + own_kl 176G on /data); stage-4 finished; main deleted the same on slinky-0/1/2/16 10-23 |
| `ttconvo-ft/stores/slinky-7` | 27.0 | 5,6,8 | copy (10-21) of slinky-7 tt-v3 rows for finished ttconvo-ft 26510; origin /scratch/amartyavarma/tt-v3/rows still on slinky-7 (+ two other copies); no job/env after 10-21 uses it; main deleted ttconvo-ft stores on s0/1/2/17 10-23 |
| `qwen-s3-ckpt/q32-s3-exp1-lr0.3-from750-j8424/dcp-1500` | 27.0 | 7 | [freed GB, hard-link aware: 7:29.0] Qwen stage-3 full-state DCP step 1500 (10-06): weights on private HF InternalCan/qwen-1ch-s3-exp1-lr0.3-from750-ckpts j8424-w-1500-bf16 (155/155 files, sha sample ok, hf-backup-staging/s3-lr0.3/state/done.json); lineage superseded by j8815 w-5900; shard set exists only on slinky-7 among the 16 MoT hosts so the DCP is already not resumable; no KEEP marker for 1500 |
| `s2-1ch-smoke` | 25.2 | 5,8 | [freed GB, hard-link aware: 5:12.0 8:15.0] Qwen stage-2 smoke DCPs (s2-1ch-smoke-*-jsmoke dcp-4/6, 10-01/10-03); finished smoke |
| `tt-pw` | 24.0 | 7 | [freed GB, hard-link aware: 7:25.7] tt-pw/ttjson = verbatim GCS turn-taking JSON download cache (dl_ttjson.py) whose product (turns/plans) is on /data qwen-1ch-data/tt-pw; ttpw run finished 10-16; on other hosts tt-pw is only symlinks (0 B) |
| `aloe-v2vox-raw/exports` | 24.0 | 15 | node-local export-253 shards (24 GiB) = /data/home/amartyavarma/ppm/aloe-v2vox-raw/export-253 (consolidated DCP, COMPLETE) + HF InternalCan/ppm-aloe-v2vox-raw export-253 VERIFY PASS (/data/home/amartyavarma/setup-logs/hfb/verify/InternalCan__ppm-aloe-v2vox-raw__export-253__slinky-15.json); on s7 the same path is a symlink to /data (do not touch) |
| `aloe-v2vox-raw-w14503/exports` | 24.0 | 15 | node-local export-253 shards (24 GiB): HF InternalCan/ppm-aloe-v2vox-raw-w14503/export-253 VERIFY PASS from slinky-15 (ppm/aloe-v2vox-raw-w14503/logs/BACKUP_LOG.txt 10-20 02:59) + /data export-253 consolidated DCP (HF_BACKUP.json) |
| `aloe-ft/stores` | 22.8 | 13,15 | Digest-verified copies of Weka /data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores/m1n0..m1n3 (ppm/aloe-ft/stage.sbatch); no running/pending job reads them. |
| `qwen-s4-ckpt/q32-s4-adapter-asr-j9345/dcp-935` | 21.5 | 7 | [freed GB, hard-link aware: 7:23.1] finished Qwen stage-4 adapter run: Weka export qwen-1ch/stage4/ckpt-weka/q32-s4-adapter-asr-j9345/w-935 COMPLETE=935; same dcp deleted on slinky-8 10-24 under main go with that record; only this host still holds a shard |
| `aloe-pose-build` | 21.0 | 11,12,13,16,17 | Per-conversation .pkl temp files of the finished 10-16 pose-store build (build_poseB_rows.py tmp dir); built stores live on /data ppm/aloe-pose-ft/stores (STORES.json digests). |
| `ppm14b/ckpt/ppm14b/dcp-4304` | 20.7 | 7 | slinky-7's slot (slinky-10 alias, ranks 64-71, nlink 1, 21 GiB): sha256-verified pinned copy at slinky-19 ppm14b/ckpt-keep/dcp-4304-slot-slinky-10 (+SHA256SUMS, README: 'slinky-7 had no room to keep it'); s7 ckpt-keep is empty; so ppm14b/ckpt on s7 can go whole (42 GiB) |
| `aloe-v2pure-17641/exports/export-200` | 20.0 | 11,19 | Node-local final export; per-host private HF InternalCan/ppm-aloe-v2pure-17641-final/export-200 VERIFY PASS (ppm/aloe-v2pure-17641/logs/hf_verify). |
| `asr14b/stream` | 17.0 | 15 | MLS stream cache (17 GiB) of asr14b, stopped by user 10-18 'Do not resume' (asr-adapter-14b/README.md); derived data, no job uses it |
| `ttconvo-ft/stores/slinky-13` | 17.0 | 16,19 | copy of slinky-13 tt-v3 rows for finished ttconvo-ft 26510; origin tt-v3/rows still on slinky-13; unused since 10-21 |
| `qwen-s3-tcache` | 11.9 | 7 | [freed GB, hard-link aware: 7:12.8] Qwen stage-3/4 text-KL teacher-logit cache 17afc394d0bffd2c (10-15), regenerable (manifests on /data/home/amartyavarma/data/qwen-s3-tcache); only slinky-7 slice left; no Qwen job running/pending |
| `inductor-cache-ppmvox` | 10.1 | 0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,19 | Compile cache of paused ppmvox runs; not used by MoT or the FT; deleting only costs a recompile on ppmvox resume |
| `aloe-v2pure-16610/exports/export-200` | 10.0 | 3 | Node-local shards of finished job 26089's final export; same weights consolidated on /data ppm/aloe-v2pure-16610/export-200 (COMPLETE) and private HF InternalCan/ppm-aloe-v2pure-16610 export-200 VERIFY PASS (setup-logs/hfb/verify/InternalCan__ppm-aloe-v2pure-16610__export-200__slinky-18.json); no per-host HF record for the shard set itself. |
| `hera-ft-vp8-rows` | 10.0 | 11,12,13,14,19 | rsync copy of Weka /data/home/amartyavarma/qwen-1ch-data/hera-ft-hov-vp8/node<N> (hera_ft_vp8.sbatch:10,50); MiB match exactly (node0 2014, node1 2027, node3 2010, node6 2023, node7 2017); 10-11 run finished |
| `s4ft-gen` | 8.8 | 3,6,7,8 | [freed GB, hard-link aware: 3:2.4 6:2.4 7:2.4 8:2.4] node-local copy of /data/home/amartyavarma/qwen-1ch-data/s4ft-gen/stores/slinky-N (2.3G each on /data); gen_usertext runs finished 10-11; slinky-2 copy sha256-verified identical and deleted 10-22 |
| `ttconvo-ft/stores/slinky-11` | 8.6 | 12 | copy of slinky-11 tt-v3 rows for finished ttconvo-ft 26510; origin tt-v3/rows still on slinky-11; unused since 10-21 |
| `asrppm/stream-nonorm` | 8.5 | 7 | MLS train stream cache (9 GiB) of asrppm-nonorm, FINISHED at 1972 by user 10-16 (asr-adapter/README.md:43); derived data, no job uses it |
| `s4asr2` | 8.5 | 7 | [freed GB, hard-link aware: 7:9.1] stage4-asr2 MLS streaming clip cache (stream_mls.py: deterministic conversion of public parler-tts/mls_eng, trainer-windowed and self-deleting); asr2 runs finished 10-13 |
| `triton-cache-q1ch` | 6.8 | 2,5,6,8,11,12,13,14,16,17,19 | Triton compile cache of old Qwen 1ch runs; not set by MoT/FT task_ppm.sh; recompile-only cost |
| `venvs/audiocorrupt` | 6.7 | 7 | [freed GB, hard-link aware: 7:7.2] uv venv of the finished audiocorrupt job (10-15); recreatable with audiocorrupt-work/scripts/setup2.sh; no job uses it |
| `hera-ft-rows` | 5.8 | 11,12,13 | rsync -a --delete copy of Weka /data/home/amartyavarma/qwen-1ch-data/hera-ft-hov8/node<N> (hera_ft8.sbatch:48); MiB match exactly (node1 2003, node2 1995, node7 1993); 10-09 hera FT long finished |
| `s2h-m1-rows` | 5.6 | 16,17 | Staged copy of Weka store /data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores/m1n<i> (stage_node.sh asserts digest==source; Weka m1n* 2.9G each present); 10-09 mtime; no running/pending job references it |
| `abl_textemb` | 5.1 | 7 | probe feature cache feat_0..3.pt (5 GiB) of the finished 10-16 text_emb ablation (ppm/abl_textemb/abl.py; logs/probe.log 'probe done', report.log written); regenerable from /data ckpt-infer local-10386 |
| `conv-pool-v2` | 5.0 | 7 | frees only 5.0 GiB alone: 28 GiB of rows are hard links of tt-s3/tt-v3 rows (CHECK, kept). [freed GB, hard-link aware: 7:5.3] Qwen pose-s3 conversation pool (10-10, runs finished); user said clear it and it was deleted on slinky-13 10-25 (ledger); ALONE frees only ~5.3 GB here (29.8 GB is hard links of tt-s3/tt-v3 rows; deleting conv-pool-v2+tt-s3+tt-v3 together frees ~35 GB) |
| (smaller SAFE items, each < 4 GiB summed) | 17.6 | | caches / small copies; see TSV |

### CHECK paths (summed over hosts; per-host rows in scratch_candidates.tsv)

| path (under /scratch/amartyavarma) | GiB total | hosts (slinky-N) | proof |
|---|---:|---|---|
| `ppm14b/ckpt/ppm14b/dcp-4502` | 248.4 | 0,1,2,3,6,7,11,12,13,14,16,17 | ppm14b (OPS_PLAYBOOK: 'WRONG run, stopped 10-23 02:00 at dcp-4502, kept for reference') stop save; same dcp-4502 already deleted on s5/s8/s19 under main's 10-25 'ppm14b old dir' approval (DELETED-1025.log) so it is no longer resumable; files nlink 1 (21 GiB freed/host); resume point dcp-4304 stays pinned in protected ppm14b/ckpt-keep; no job uses it // MONOSAM: no export/HF copy of dcp-4502 (rule: never delete unexpo ... |
| `ppmvox/vpod3/rows` | 126.4 | 0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,19 | PPMVOX v5 podcast row stores (8 GiB/host) built by data_tools/vpod3_host.py prep from HF bucket + Weka selections (launch/vpod3_sel); no HF/Weka copy of the built rows; needed only for a v5/v5b resume (v5b-vpod index points into own vpod3 on B hosts); rebuild = hours |
| `ppmvox/v5b-vpod` | 84.0 | 0,3,5,8,11,12,16,19 | v5b merged vpod index + partner host's vpod3 rows/fresh pose copies (PPMVOX_VPOD_INDEX / FRESH_POSE_ROOTS_PARTNER in ppmvox_8n_v5b.env); only needed if v5b resumes (stopped 18163; user 10-25 05:04 'assume Voxtral unfreeze isn't working'); no backup of the built rows |
| `aloe-pose-ft/exports` | 60.2 | 12,16 | export-287/358/430 node-local shards (10-16 run); per-host GCS VERIFY PASS (ppm/aloe-pose-ft/logs/gcs_verify, md5+crc32c) and export-430 consolidated GCS PASS (logs/gcs_verify_weka); backup is GCS not HF; precedent: main deleted aloe-pose-ft on s11/s13/s19 10-25 on the same proof (OPS_PLAYBOOK.md:562-563). // MONOSAM: backup is GCS only (no HF record, no /data copy) -> CHECK; main deleted same dirs elsewhere 10-25 un ... |
| `qqa-gen` | 59.1 | 0,1,2,3,5,6,7,8,11,12,13,14,16,17,19 | 10-15 QQA CC-BY TTS shard copies (generated/ft/pt); backup = GCS gs://podcast-raw-audio-data-mass/amartyavarma/quick-qa-100k-tts-ccby/ 224/224 verified size+md5+crc32c (qqa-gen/GCS_STAGED, STATUS.md) but HF repo InternalCan/quick-qa-100k-tts-ccby has 0 tars (HF quota blocked upload; checked via HfApi 10-26) and no Weka copy -> deleting leaves GCS as sole copy: needs user OK |
| `tt-s3` | 38.1 | 7,11,13 | alone frees ~0 (hard links of conv-pool-v2 rows); with conv-pool-v2 frees ~28 GiB more. [freed GB, hard-link aware: 7:0.1 11:20.6 13:20.2] built Qwen tt-s3 duplex rows (10-10) for finished Qwen tt runs / ttconvo-ft / conv-pool-v2; NO /data or HF copy of the built rows (rebuild = GCS source re-download + build_tt_rows.py); last copies on 7/11/13; on slinky-7 hard-linked into conv-pool-v2 (frees 0 unless conv-pool-v2 a ... |
| `aloe-uncond-ft/exports` | 32.2 | 1,16,19 | export-430 node-local shards; per-host GCS VERIFY PASS (ppm/aloe-uncond-ft/logs/gcs_verify) + consolidated export-430 GCS PASS (logs/gcs_verify_weka); backup is GCS not HF (HF scratch-backup-1024-aloevox has only the s11/s13 copies). // MONOSAM: backup is GCS only (no HF record, no /data copy) -> CHECK; main deleted same dirs elsewhere 10-25 under user "clear old stuff" |
| `ac-out` | 30.1 | 7 | [freed GB, hard-link aware: 7:32.3] audiocorrupt (quick-qa-100k-tts-ccby-corrupt) slinky-7 worker output, 57 shards (10-15); dataset verified complete on GCS only (audiocorrupt-work/verify_gcs.json 224/224 shards); HF upload blocked (hf_status.json org storage limit) -> no HF copy; user rule is HF-only backups |
| `ttconvo-ft/stores/slinky-18` | 27.0 | 3,5,6,8,11,12,13,16,19 | copy of slinky-18 tt-v3 rows (9 identical copies on MoT hosts); origin on slinky-18 not verified (not probed: off-limits/demo node); finished run; deleting all copies could remove the last one |
| `tt-v3` | 17.1 | 7,11,13 | alone frees ~0 (hard links of conv-pool-v2 rows); with conv-pool-v2 frees ~28 GiB more. [freed GB, hard-link aware: 7:0.0 11:9.2 13:9.1] built Qwen tt-v3 rows (10-10); NO /data or HF copy; origin of ttconvo-ft/stores/slinky-N copies; on slinky-7 hard-linked into conv-pool-v2 |
| `aloe-pw` | 12.7 | 0,1,2,3,5,6,7,8 | 10-14 node-local row store (rows-*-of-00012, ~1.4 GiB, 3 GiB on s7); no Weka/HF copy found (/data qwen-1ch-data/aloe-pw holds only plan/logs, 83 MB); not referenced by any live script. |
| `tt-n1` | 9.7 | 11,13 | [freed GB, hard-link aware: 11:5.2 13:5.2] built Qwen tt-n1 rows (10-13) for finished ttpw run; NO /data or HF copy; tt-pw/rows symlinks point here |
| `ppmvox/pose-slot` | 9.6 | 7,15 | ppmvox v2/v3 re-split pose stores for slots slinky-10/slinky-9 built from slinky-4's pose shards (launch/prep/poseslot_build.py; source HF record InternalCan__ppm-pose-s3-samdata-rows__slinky-4.json); only ppmvox v2/v3 envs reference it (v5 never opens it); rebuildable but no direct backup |
| `ttconvo-ft/stores/slinky-4` | 8.8 | 3 | only MoT-host copy of slinky-4 tt-v3 rows; origin on slinky-4 not verified; finished run |
| `aloe-tt3` | 7.7 | 0,1,2,3,6,7,8 | Node-local rows (~1.1 GiB); Weka qwen-1ch-data/aloe-tt3-rows holds only READY markers; no copy proof; not referenced by live scripts. |
| `hf-hold` | 7.2 | 0,1,3,6,7,8,11,12,13,16,17,19 | ~0.6 GiB per host, contents not inspected (possible backup hardlink hold); not worth deleting. |
| `asrppm/ckpt` | 7.0 | 0,1,7 | adapter resume states (r-1920/r-1972 .pt, ~0.7 GB each) of asrppm/asrppm-nonorm; run finished 10-16 and final export on /data asr-adapter/ckpt-final/asrppm-nonorm/final-s1972 (+GCS), but resume states are training state; small (1.5 GiB/host) |
| (smaller CHECK items, each < 4 GiB summed) | 8.1 | | caches / small copies; see TSV |

### KEEP paths (summed over hosts; per-host rows in scratch_candidates.tsv)

| path (under /scratch/amartyavarma) | GiB total | hosts (slinky-N) | proof |
|---|---:|---|---|
| `ppmvox/ckpt/v5-18100-merged-keep` | 584.8 | 0,3,5,8,11,12,16,19 | PROTECTED by */*/*keep* glob (guard_rm refuses; also makes ppmvox and ppmvox/ckpt undeletable as units). FYI 73 GiB/host, HF-verified InternalCan/ppmvox-v5-18100-merged-state sha256all PASS x8 (/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025); deleting needs the USER's path-specific OK |
| `npl-append` | 536.7 | 0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,19 | Pose deliveries 2+3 raw parquet (npl2/jobs, npl3/jobs; HF buckets InternalCan/pose-llm-data/new-pose-llm-data-delivery2/3), node-local per-host shards = row source for ppm-fresh-pose-npl2/npl3 plans used by paused ppmvox v5b (PPMVOX_FRESH_POSE_ROOTS) and prod_811_decay.env; no Weka copy; 10-25 approved clean list explicitly KEEPs npl-append; restaging = HF re-download -> only with user OK |
| `new-pose-llm-data` | 455.8 | 0,1,2,3,5,6,7,8,11,12,13,14,16,17,19 | Node-local unique shard (PLACEMENT.json job_to_host, ~31 GB/host of the 517 GB HF bucket InternalCan/video-pretraining-data/new-pose-llm-data; Weka holds only a 71 GB partial) = on-the-fly row SOURCE for ppm-fresh-pose plans used by paused ppmvox v5b (ppmvox_8n_v5b.env PPMVOX_FRESH_POSE_ROOTS, resumable from dcp-18163) and 8,1,1 decay env; 10-25 main/user-approved clean list (ppmvox/code/tools/v5b/clean_b_host.sh) ex ... |
| `qwen-s3-ckpt/q32-s3-exp1-duplex-sp-split2-glu-r2-j8333/dcp-750` | 320.0 | 0,2,3,6,7,8,11,12,13,16,17,19 | [freed GB, hard-link aware: 0:29.2 2:29.0 3:29.0 6:29.0 7:29.0 8:29.0 11:29.0 12:29.0 13:29.0 16:29.0 17:28.3 19:25.3] sibling file KEEP_dcp-750 (node + Weka ckpt-weka/.../j8333/KEEP_dcp-750): "Main run resumes from dcp-750 here (protected ... do not delete)"; only w-750 weights on Weka, full state not on HF; NOTE guard_rm would NOT refuse this path (29 GB/host), so it must stay off any list |
| `qwen-s3-ckpt/keep-lr0.3-dcp-1250` | 266.0 | 0,2,3,6,7,8,12,13,17,19 | [~25-29 GB/host, but the two dirs share the same hard-linked files] protected *keep* glob + KEEP_dcp-1250 marker; the two dirs are hard links of the same files (deleting j8424/dcp-1250 frees 0 B) |
| `ppm14b/ckpt-keep` | 186.4 | 0,2,3,5,8,13,17,19 | PROTECTED (*keep*): pinned ppm14b dcp-4304 (+ slinky-10 slot copy on s19) |
| `aloe-v2vox-asr-w15973/exports/export-253` | 136.9 | 5,6,7,8,11,16,17,19 | PROTECTED path in PROTECTED_SCRATCH.txt (asr16k demo weights); it is the only sizeable content of aloe-v2vox-asr-w15973 (s7 copy 73 GiB) |
| `ppm14b/ckpt/ppm14b/dcp-4304` | 124.2 | 1,6,11,12,14,16 | hard links (nlink 2, checked s0) of protected ppm14b/ckpt-keep/dcp-4304: deleting frees ~0 GB; leave (du may attribute its 21 GiB here on s1/6/11/12/14/16 because of traversal order) |
| `qwen-s3-ckpt/q32-s3-exp1-lr0.3-from750-j8424/dcp-1250` | 54.0 | 11,16 | [~25-29 GB/host, but the two dirs share the same hard-linked files] protected *keep* glob + KEEP_dcp-1250 marker; the two dirs are hard links of the same files (deleting j8424/dcp-1250 frees 0 B) |
| `inductor-cache-ppm` | 22.0 | 0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,19 | In use: MoT 30289 task_ppm.sh:12 TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/inductor-cache-ppm/$H (the FT run uses the same default) |
| `triton-cache-ppm` | 7.2 | 0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,19 | In use: MoT 30289 task_ppm.sh:11 TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm (the FT run uses the same default) |
