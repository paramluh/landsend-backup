# pose5 independent audit, round 3 (delta)

Auditor: an independent read-only agent. It edited nothing and submitted no jobs. Audited 10:12-10:48 PT.

Files at audit time (UTC mtimes):
- `kit/stage_host.sh` 17:26 (with C's Weka symlink step);
- `kit/exact_check.py` 16:57;
- `kit/hold_chain.sh` 17:08;
- `kit/export_drop.py` 17:09;
- `kit/ppm811s2p_watchdog.sh` 17:05;
- `kit/gate_rules.py` 17:00;
- `c/disk/build_disk_need.py` 17:00;
- `switch_to_pose.sh` 17:11;
- `npl5/bin/npl5_roll.py` v2.1 16:58, md5 07ac8be8;
- Weka tests `npl5/wekatest/run-31062`, `run-31073`;
- sim 31074 still RUNNING at 10:47 PT (0 misses so far; see R3-2).

**Verdict: every round-2 BLOCKER and SHOULD-FIX is fixed in code, and I found no new BLOCKER.**

Two SHOULD-FIX items remain:
- R3-1: the hourly NOSPACE behaviour moved from slinky-13 to slinky-6.
- R3-2: there is no passing end-to-end run yet of the Weka fetch path at production pace; job 31074 is pending.

---

## Round-2 findings: verification

| R2 | Item | Status | Evidence (read and, where cheap, replayed) |
|---|---|---|---|
| 1 | Exactness check used wall-clock keys at tol 0 | **FIXED** | `kit/exact_check.py` uses the trainer's rule (`eval_` / `codependence/` / `xdep/`, minus `*time_s`); EXACT→0, ≤ 1e-6 rel→10 (LIVE + WARN), else 1. **Replayed:** 27653 vs 27204 @17641 gives 184/184 EXACT, rc 0; decay vs itself @17850 gives 184 EXACT. Wired at `switch_to_pose.sh:271-272` and `325-330`. All 7 python heredocs in the switch compile on the login node's Python 3.10.12, so C's py3.10 f-string fix holds; `bash -n` passes. D's `eval_g0cb0.json` shows extra = [], so a correct launch reads EXACT, not WARN. |
| 2 | MoT caches in the slinky-13 delete list | **FIXED** | `stage_host.sh` step 2 skips `*mot*`, `inductor-cache-ppm`, `triton-cache-ppm`, and any path a MoT `.sh/.sbatch/.py/.env` names (grep over `ppm/mot`). `build_disk_need.py:47-60` excludes the same from the lists. Today only slinky-6 and slinky-12 delete caches. I grepped every one of their paths (triton-cache-q1ch, inductor-cache-ppmvox/-s3perf/-samft(-lr3), triton-cache-ppm-fast/-tt3/-samft-lr3, aloe-pose-build, triton_hera_ft) across `ppm/mot`: **0 hits**. MoT's own cache dirs (inductor-cache-ppm, triton-cache-ppm/-gen14/-motlive/-motsamples, hfb/xet-cache, next-arch/*, ppm-mot-test/*) are all excluded. Deletion goes through `guard_rm "${p:?}"`. |
| 3 | Hold-chain membership broke with 2 holds | **FIXED** | `hold_chain.sh:59-90`: HELD counts only saves held on all hosts (`$1==n`) with `grep -qx` membership; partial holds are retried and timestamped (HELDANY), released and alerted; a hold dir with another inode is MISMATCH (never touched, alerted); `mv -T`; a released save is never re-held. Every `rm` is `"${HD:?}/dcp-${s:?}"` (s regex-checked as digits). |
| 4 | slinky-13 hourly NOSPACE | **FIXED for s13** (Weka-served) | Moved to slinky-6: see R3-1. |
| 5 | Export auto-drop never drops | **FIXED** | `export_drop.py:135-176` matches against the bundle's `_backup/MANIFEST.w-N.tsv`: export rows == the on-disk export, MANIFEST == the verified totals, live HF `w-N/` == the MANIFEST, plus sha256. **Replayed (read-only HF):** `--explain 15973` gives "would match but REFUSED: 15973" (325 / 90.80 GB bundle, 131 / 88.69 GB export, 323 HF files). A crash means KEEP + ALERT (`:204-208`). hold_chain runs it with `venvs/duplex/bin/python` under flock (`:159`). Hard refusals: 15973, steps ≤ 17850, other jobs' exports, MoT, realpath outside `ckpt-weka/ppm-v1`. |
| 6 | Empty job id failed open | **FIXED** | `switch_to_pose.sh:206-212` `need_id` after every submit: a hard failure cancels all numeric ids and STOP_ROLL, then dies with MoT paused and `--after-pause` advice. Watchdog `:139` refuses a non-numeric JOBF at start; `:144-149` ignores and alerts on one at runtime. |
| 7 | No `PPM_EXPORT_KEEP=0` assert | **FIXED** | `switch_to_pose.sh:55-56`. |
| 8 | Kept dcp-17798 later pruned | **FIXED** | `stage_host.sh:62-70`: if 17798 is still in CKPT after step 1, ALERT + ABORT. |
| 9 | Heartbeat numbers stale | **FIXED** | `npl5_roll.py:405-423`: every 30 s it recomputes live, coverage, ahead and alerts and rewrites markers mid-pass; SLOWPASS after 15 min. Evidence: `run-31073/heartbeats.txt` shows BEHIND → CRIT → STARVED raised inside a fetching pass. The fetcher-S3 NIT is also fixed: the resume key is (log, resume count), `:360-362`. |

---

## Findings

### R3-1 SHOULD-FIX: slinky-6 now has the hourly NOSPACE pause, with only a 40-step lead
**Where:**
- `c/DISK_NEED.json`: s6 ahead 40 (autoshrink), +0.8 GB after margin;
- `kit/stage_host.sh` writes `MIN_FREE_GB-slinky-6 = 83`;
- `npl5_roll.py` NOSPACE check.

**What is wrong.**
- After its first save, s6 has about 89 GB free, so it fetches.
- From each save N+1's COMPLETE until hold N is released (12-14 min HF verify, measured on 17746 and 17798), free drops to about 43 GB, which is below 83. Every fetch is then NOSPACE.
- A 40-step lead is about 45 min. Normal operation only gives hourly NOSPACE / BEHIND alerts.
- If one HF verify runs more than ~40 min late, s6 starves: FileNotFoundError, relaunch, repeat until the verify lands.
- The same hourly NOSPACE alert also fires on the other near-gate hosts (s0, s19, s16, ... at ahead 100), but they do not starve. That is still about 10 alert lines per hour in ALERTS.log (`hold_chain.sh` alerts on every NOSPACE marker once per host per hour), against the user's no-chatter rule.

**Fix (main decides).**
- For s6: serve its window from Weka like s13 (`gate_rules.WEKA_HOSTS`, the same code path).
- For all hosts: alert on NOSPACE only when it lasts more than 30 min or ahead drops below low-water.

### R3-2 SHOULD-FIX (evidence): no passing Weka-path sim at production pace yet
**What is wrong.**
- `wekatest/run-31062` and `run-31073` sims both return rc 3 (`sim_trainer.py:88`, misses > 0). run-31073: 242 misses from step 18056.
- The cause is the test pace, not the code: 0.5 s per simulated step, about 136× production's 68 s, with 30 % fake 429s. One fetch pass took 204 s while the sim advanced ~400 steps.
- The parts these runs do prove all pass:
  - the refuse test (rc 4, nothing written);
  - prep through the symlink (254 files on Weka, 0 on node-local);
  - nothing touched outside the Weka dir;
  - heartbeat.
- 31074 (the production-ratio sim, slower pace) is still running. At 10:47 PT it had reached sim step 18230 with **0 misses** at every checkpoint, across the relaunch from the held save (~18150).
  - The daemon dropped files inside the Weka test dir only (1,330 + 766).
  - Peak present was ~7.7 GB, and the 429 counter was rising.

**Fix.** Treat slinky-13 as READY once 31074 exits rc 0 (sim_trainer exits 0 only with no misses). If it ends rc 3 at the production ratio, that is a real BLOCKER for the Weka path.

### R3-3 NIT: the end-of-pass status can move `live` backwards
**Where:** `npl5_roll.py:432-449`.

**What is wrong.** `one_pass` writes the `live` it read at the START of the pass into ST at the end. That overwrites the heartbeat's newer value: `run-31073/heartbeats.txt` shows 18013 then 17982. ahead/alerts can flip back briefly. Cosmetic.

**Fix.** Re-read from the log at the end of the pass.

### R3-4 NIT: the slinky-13 disk-trend alert now reads /data numbers
**Where:** `hold_chain.sh` trend block.

**What is wrong.** s13's `scratch_free_GB` is now statvfs of the resolved dest (/data). The trend alert compares it with s13's /scratch floor of 20 + 3.5, so it can never fire. The fetcher's 350 GB /data floor and its NOSPACE marker alert are the effective guard. Fine, but the page label is misleading.

### R3-5 NIT: the remaining literal-path test `rm`s
**Where:**
- `npl5/bin/safety_test.sh:49` (`rm -rf /scratch/amartyavarma/npl-append/npl5` in a srun on its dummy job);
- `npl5/bin/probe.sbatch:18` (a /data probe dir).

**What is wrong.** Both are literal paths (allowed by the standing rule), on spares only.

`e2e.sbatch`, `rehearse.sbatch` and `weka_test.sh` delete canonical paths only after aborting when those paths pre-exist, so they cannot run destructively on a production host.

---

## Checked and found correct in round 3
- **C's Weka symlink step** (`stage_host.sh:160-182`). It runs only for `gate_rules.WEKA_HOSTS` (`slinky-13`).
  - It creates the symlink only if the canonical `jobs` path is absent.
  - An existing symlink to a different target means ABORT.
  - A real dir or file there means ABORT. It is never replaced, moved or deleted.
  - It writes `MIN_FREE_GB-slinky-13 = 350` and `AHEAD-slinky-13 = 100`, and the later generic writes skip Weka hosts (`:187-193`).
  - `AHEAD`/`WGB` are validated as numbers (the round-2 fail-open NIT is fixed).
  - The live gate's `du` of the npl5 dir does not follow the symlink, so s13's /scratch window term is correctly 0.
  - `bash -n` passes.
- **Fetcher v2.1 dest safety** (`npl5_roll.py:156-185`).
  - If `jobs` is a symlink, it must resolve exactly to `realpath(npl5-weka/<phys>/jobs)` under `npl5-weka/`, else exit 4 (refuse test passed).
  - If it is a real dir, it must be under `/scratch`.
  - Every delete goes through `path_ok` (a plan file AND its realpath inside the resolved root) or `safe_remove` (an assert).
  - The disk floor uses statvfs of the resolved dest, so s13 keeps /data ≥ 350.
  - The lock and state files stay node-local.
  - Headroom = max − (current − (file − shmem)).
- **Delete-guard sweep** over `kit/`, `npl5/bin/`, `slot9/`, `switch_to_pose.sh` and `c/`: every variable `rm` uses `"${VAR:?}"` per component, or a regex-validated step, or a literal path. Python deletes are confined (export_drop realpath check; fetcher `safe_remove`; `stage_dcp17850` only its own `.dl`).
- **Nothing MoT can be deleted.**
  - The prune / space_check touch only `ckpt/ppm-v1` (MoT is `ppm-mot`).
  - The cache lists exclude every path MoT names.
  - `export_drop` refuses MoT paths.
  - Nothing else in the kit deletes outside pose5's own paths.
- **ARMED** at 10:11 and 10:13 PT was green per STATUS. The `--dry-run` writes `c/DISK_NEED.dryrun.json`, so the live file is left untouched.

## Not re-checked / pending
- Sim 31074 (production-ratio Weka sim): still RUNNING at 10:47 PT; 0 misses through sim step 18230 (R3-2).
- C's `--armed` after the 17:26 UTC `stage_host.sh` edit (expected green; not observed by me).
- The live host state and any runtime path on the 16 hosts (by design).
