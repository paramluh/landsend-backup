# pose5 independent audit, round 2 (final state)

Auditor: an independent read-only agent. It edited nothing and submitted no jobs. Audited state: 10-27 09:42-10:00 PT.

Two read-only sub-auditors helped: one on npl5 / the fetcher / the data, one on the switch / kit / ops. I re-checked every BLOCKER and SHOULD-FIX below myself, by reading the code or replaying the logic. Exceptions are marked "(sub-audit)".

E2E has landed: job 31048 COMPLETED at 09:41:46 PT, and `npl5/e2e/E2E.json` reads `{"slots_ok": 16, "end": 19382}`.

**Verdict.**
- The maths is correct and well evidenced.
- Both round-1 maths/ops BLOCKERs (#1 watchdog rewind, #3 w-15973) and both fetcher BLOCKERs (#2, #4) are really fixed.
- Two new BLOCKERs need small fixes before `--go`:
  - R2-1, a bug in the switch's go/no-go check;
  - R2-2, a decision that conflicts with a binding user rule.
- Everything else is SHOULD-FIX or NIT.

Severity: **BLOCKER** = fix before --go; **SHOULD-FIX** = fix or have main accept in writing; **NIT** = low risk.

---

## BLOCKERS

### R2-1. The switch's exactness check fails on every correct launch
**Where:** `switch_to_pose.sh:270-276`.

**What is wrong.**
- `ks` takes every numeric key common to the new run's step-17850 record and `runs/ppmdecay-j27810`'s, except step/time/t/elapsed.
- That includes the wall-clock keys `eval/time_s`, `eval/time_s_base`, `codependence/time_s` and `xdep/time_s`, with tolerance `EXACT_TOL` = 0.

**Evidence.** I replayed the exact logic on the pair known to be bit-exact, 27653 vs 27204 at step 17641. Result: 192 keys, worst |diff| 1.49 (`eval/time_s`), 188 exact.

**Failure scenario.** A correct launch prints "NOT VERIFIED", writes an ALERT, and dies without "POSE RUN LIVE" (the run keeps going). Main is pushed to stop a correct run, or learns to ignore this check.

**Fix.**
- Use the trainer's own rule (`train_ppm.py:1675`): keys starting with `eval_`, `codependence/` or `xdep/`, excluding keys ending in `time_s`.
- Slot 9 now runs on slinky-15 instead of slinky-9, and the eval sums are float64 all_reduces (`allsum`, `train_ppm.py:414`). So report the worst diff, and fail only above ~1e-9 relative or on a key mismatch.

**Cross-check that the reference is right.** `code-811-decay`'s `evaluate()` is byte-identical to code-811's, and its model and data files are identical. Its `[811 rows]` lines are logged by ranks 0 and 127, matching the 4 lines the check expects.

### R2-2. The slinky-13 cache deletion includes MoT's compile caches, against the user's 06:16 rule
**Where:**
- `c/DISK_NEED.json` hosts.slinky-13: `delete_caches` is true and the list includes `/scratch/amartyavarma/inductor-cache-ppm` and `/scratch/amartyavarma/triton-cache-ppm`;
- `kit/stage_host.sh:69-75` deletes them on slinky-13 (`guard_rm`).

**What is wrong.**
- `ppm/mot/code/task_ppm.sh:11-12` sets `TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm` and `TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/inductor-cache-ppm/$H`. These are the same directories, shared with MoT.
- The user's binding rule (06:16) says NOTHING MoT-related on the 16 hosts is deleted, and lists caches explicitly. Main approved this deletion (06:48, 08:05) on the belief that they were "the run's own" caches.
- The `*mot*` name filter does not catch them.

**Failure scenario.** The switch deletes MoT state on slinky-13. It is regenerable (a recompile when MoT resumes), but the user's rule is broken.

**Fix (main/user decision).**
- Re-ask main with this fact, or drop the two paths from slinky-13's list.
- Dropping them costs about 1.9 GB. slinky-13 is at +0.4 GB over need, so it may then fail the pre-pause gate. Main then needs a smaller s13 window or another regenerable source; deciding that now is better than discovering it at the switch.
- Found by the sub-audit; I verified the paths and the env lines.

---

## SHOULD-FIX

### R2-3. The chained hold mis-tracks once two saves are held (every hour)
**Where:** `kit/hold_chain.sh:54-56`.

**What is wrong.**
- `HELD` is newline-separated, so the membership test `echo " $HELD " | grep -q " $s "` only matches when exactly ONE save is held. I verified this with bash: `HELD=$'17902\n17954'` gives no match for 17902.
- Two saves are held every hour from save N+1's COMPLETE until N's release (~13 min: 17746 and 17798 took 12-14 min to HF-verify).

**Failure scenario.**
- In that window each poll re-holds N+1. The `cp -al` goes to `.tmp`, and `mv $HD/dcp-N+1.tmp $HD/dcp-N+1` moves the tmp dir INTO the existing hold, nesting it. That costs 0 B but leaves junk in a dir the watchdog may resume from.
- Later polls fail the mv, and after 3 polls the script raises a false "NOT held (KEEP=1 safety net down)" ALERT. That repeats every hour, so a real alert will be ignored.
- `HELD` also counts a save held if ANY host holds it (`uniq -c | awk '{print $2}'` with no count filter). A 15/16 partial hold is never retried and never timestamped, so the 2 h "no HF verify" alert never fires for it. That is exactly the silent KEEP=1 failure round 1 asked to catch.

**Fix.**
- Build `HELD` with `awk -v n=$NH '$1==n{print $2}'`.
- Test with `grep -qx`.
- Use `mv -T`.
- Alert on partial holds.

### R2-4. slinky-13's fetcher pauses about 15-25 min every hour, so CRIT/BEHIND alerts fire hourly and it starves if HF verify is late
**Where:**
- `kit/stage_host.sh:152`: `MIN_FREE_GB = 53 + floor`, i.e. 73 GB on s13 and 83 GB elsewhere;
- `npl5/bin/npl5_roll.py:275` (NOSPACE);
- s13 ahead 40 ≈ 45 min of steps.

**What is wrong.** slinky-13 passes the gate by only +0.4 GB (DISK_NEED: 131.4 vs 131.0).
- With one save present, it has ~77-85 GB free and fetches.
- Between save N+1's COMPLETE and N's release (hold N plus CKPT N+1), free drops to ~31-36 GB. That is below 73, so it hits NOSPACE and fetches nothing.
- The same happens on the other tight hosts (s19 +0.8, s6 +1.6), but ahead 100 covers it there.
- On s13 the lead drops to ~18-20 steps each hour, so CRIT/BEHIND alerts fire. If an HF verify runs more than ~40 min late, it hits FileNotFoundError and a relaunch loop until the verify lands. hold_chain's "no verify" alert only fires after 2 h.

**Fix (main decides).**
- Either s13's MIN_FREE = floor + margin + window (rely on the trainer's loud skip-save fail-safe);
- or reserve the 53 GB write room only until the previous hold is released;
- or raise s13's "no verify" alert to ~30 min.
- Also suppress NOSPACE alerts shorter than ~30 min to avoid hourly chatter.
- Found by the sub-audit (S1); I independently derived the same numbers.

### R2-5. The export auto-drop can never drop anything, so main's 07:42 decision is not delivered
**Where:** `kit/export_drop.py:105-110`.

**What is wrong.** It compares the export dir with hfb's `done/w-N` `local_files`/`local_bytes`. Those describe the uploaded BUNDLE, which includes `run_state/`.
- `done/w-15973`: 325 files, 90,804,619,469 B.
- The dir on disk: 131 files, 88,691,316,063 B (verified).

**Failure scenario.**
- No pose export ever qualifies. It fails closed (KEEP), so it is safe, but it alerts hourly after 3 h.
- /data gains about 89 GB every 6 h (not the 81 GB in the ledger), about 445 GB over the run.
- That is fine only once the 507 GB Weka copy is dropped; otherwise the trainer refuses exports below 300 GB free.

**Fix.**
- Compare against the bundle spec, or exclude `run_state/` on both sides.
- Add a self-test case that includes a bundle.
- Update the ledger to ~89 GB per export.

### R2-6. An empty job id fails open in the switch and the watchdog
**Where:** `switch_to_pose.sh:228-233`; `kit/ppm811s2p_watchdog.sh:138` (`[ -s JOBF ]`).

**What is wrong.**
- If the stage `sbatch` fails at submission, `SJ` is empty, the `afterok:` submit fails, and `J` is empty.
- The switch still writes a newline into `CURRENT_JOB_811`, which passes the watchdog's `-s` test, and starts the watchdog.
- The watchdog sees "job gone". The scan finds no save > 17850, so it relaunches ppm811 from the resume link with NO stage. The trainer then fails at load, because the resume link and the npl5 plans are absent. Nothing is damaged, but the gate is bypassed and the relaunch cap is burned. (sub-audit; verified by reading)

**Fix.**
- After every `csb`/`sbatch`, require `[[ $X =~ ^[0-9]+$ ]]` or stop everything and die.
- The watchdog refuses a non-numeric JOBF.

### R2-7. ARMED does not assert `PPM_EXPORT_KEEP=0`
**Where:** `switch_to_pose.sh:51-58` (the env checks).

**What is wrong.** w-15973's safety now rests on one env line (`env:351`), which is not a cfg key, so the cfg diff cannot see it. If that line is lost, the inherited `PPM_EXPORT_KEEP=2` (`env:104`) deletes w-15973 at the second pose export.

**Fix.** Add `[ "$(ev PPM_EXPORT_KEEP)" = 0 ] || fail "PPM_EXPORT_KEEP"`.

### R2-8. dcp-17798 that is KEPT after a failed live HF re-check is still pruned by the trainer
**Where:**
- `kit/stage_host.sh:54-60`: the re-check fails, so it logs "KEPT" and continues;
- `kit/stage_host.sh:~199`: the older-saves check explicitly exempts 17798.

**What is wrong.** At the first pose save, keep-1 `prune` deletes `CKPT/dcp-17798`, which is not in PROTECT_STEPS. Main's approval was conditional on the live re-check passing; an HF API error also lands in "KEPT".

**Fix.** If 17798 still exists after step 1, ABORT the stage (main decides), or hard-link-pin it outside CKPT like 17850.

### R2-9. The fetcher's heartbeat hides a stuck or long fetch pass (sub-audit)
**Where:** `npl5/bin/npl5_roll.py:208-214`.

**What is wrong.** The 30 s heartbeat rewrites `<host>.json`, but `ahead_steps`, `covered_until` and the markers only change at the end of a pass. A pass slowed by 429s (12 × 60 s per file) can run for an hour while hold_chain sees a fresh file with stale numbers. The trainer then dies loudly at the missing file, so the problem is late detection, not silent maths.

**Fix.**
- The heartbeat recomputes covered/ahead from `present()` and the log.
- Or alert when the end-of-pass time is more than 15 min old while phase == fetching.

---

## NITs
- **Watchdog partial newer save** (`ppm811s2p_watchdog.sh` `resume_decision`). It relaunches from the newest save complete on all hosts even when a newer save is COMPLETE on some hosts only (it only prints a NOTE to stderr). With keep=1, that stale, numerically newer save wins `prune` on those hosts and deletes the relaunched run's next save there. Rare; ALERT instead of relaunching.
- **Fetcher floor lowering** (`npl5_roll.py:360-364`). The floor is lowered only when the resume step changes value, not when the job changes. It is unreachable now that the watchdog never pins once a newer save exists anywhere.
- **Gate fail-open** (`stage_host.sh:154-166`). If `gate_rules.py ahead` errors while `eval` works, `AHEAD`/`WGB` are empty: `AHEAD-<host>` is written empty, so s13 falls back to 100, and PEND = 0. Validate both values as numbers.
- **Memory headroom** (`npl5_roll.py:247-250`). Headroom subtracts memory.stat `file`, which includes shmem/tmpfs (DataLoader /dev/shm). Subtract `file - shmem`.
- **Rehearsal override.** Rehearsal 31047 passed under an unrecorded MIN_FREE override (31041 had failed NOSPACE at 26.7 GB free). Record it.
- **What e2e 31048 tested.** It checked fetch, digests and 24 trainer micros per slot, but never ran `npl5_roll.py`. The sim (07:53) and the kill/OOM test (08:22) did, after the final roll.py mtime (07:46). The micro records do not say whether the micros were built with g0cond/s2p. Those paths are covered by D's tests.
- **Switch timing and greps.**
  - It waits 1 h for the stage, but the stage limit is 2 h.
  - It cancels `$J` before `$WD`.
  - It requires 180 eval keys, not 184.
  - It accepts a "GPUS FREE … (BACKUP FAILED" line, which mot_pause.sh writes only with `--free-on-backup-fail`.
  - It rewrites `c/DISK_NEED.json` in `--dry-run`.
- **Hold-chain edges.**
  - `hold_chain.sh:80` sends the python block's stderr to /dev/null, so the trend and marker alerts can die silently.
  - An existing pin's inodes are not re-checked (`stage_host.sh:100`).
- **Small export_drop and test-script items.**
  - `export_drop.py:113` does not byte-compare small files, although its docstring says it does.
  - `npl5_roll.py:182` runs an `os.remove` without `path_ok`.
  - `safety_test.sh:49` runs an unguarded `rm -rf` on a test node.
- **Trainer START EVAL log.** It still globs `runs*/ppm*-j*` and will compare against ppmvox-v5 (69 text keys). That is cosmetic, because the switch's own check supersedes it once R2-1 is fixed.
- **G0 coverage of npl2.** It rests on scan stats only (no local copy). Missing fields are now LOUD (`[g0cond] WARNING` plus a startup count), which is acceptable.
- **Open decisions for the user.**
  - **s2p variant:** board t1027-24 still shows "YOU: full vs cb0" (asked 06:35). The env runs main's default, cb0. Confirm.
  - **codependence probe:** it is still blind to s2p (round-1 #17). It is recorded as a note for the user and not changed (eval only).

---

## Round-1 findings: fixed / not fixed

| R1 | Item | Status | Evidence (re-read) |
|---|---|---|---|
| 1 | Watchdog rewinds to 17850 | **FIXED** | `ppm811s2p_watchdog.sh:66-108` `resume_decision`: fewer than 16/16 answering → ALERT; newest complete in CKPT, or the hold → explicit dir; split or partial → ALERT; the pin is used only if nothing > 17850 exists anywhere. Partial-newer edge: NIT. |
| 2 | Fetcher LOWMEM silent | **FIXED** | `npl5_roll.py:230-252` subtracts page cache; LOWMEM writes the JSON and a marker; hold_chain alerts; safety test v2 read the real cgroup. |
| 3 | Export rotation deletes w-15973 | **FIXED** (safety) | env:351 `PPM_EXPORT_KEEP=0`; `safety_ppm.py` rmtrees only if keep > 0; `export_drop.py` hard-refuses 15973 and steps ≤ 17850. ARMED does not assert it (R2-7). |
| 4 | Fetcher floor re-download | **FIXED** | Floor persisted in `.roll-state.json` and monotonic; candidates CKPT + hold + pin; MIN_FREE NOSPACE floor (now too strict on s13: R2-4). |
| 5 | Resume link into CKPT | **FIXED** | `stage_host.sh:101`: SRC = pin. |
| 6 | Hold / HF-watcher alerts | **PARTLY** | Alerts exist (`hold_chain.sh:41-45,57,64-66,143`; watchdog :145-151), but they are broken by R2-3. |
| 7 | Clip coupling of new params | **FIXED** (main: own groups) | `model_ppm.py` GROUPS + JOIN + `group_of`; `train_ppm.py` copies lr/clip, uses the same lambda, appends the LambdaLR entry at the JOIN lambda on resume. Evidence: `existing_group_norms_identical` on 5 row types × real and tiny; resume combos `first_step_lr_equals_join`. |
| 8 | Exactness vs ppmdecay | **IMPLEMENTED BUT BROKEN** | R2-1. |
| 9 | evaluate() path untested | **FIXED** | `tests/g0/out/eval_g0cb0.json`, `eval_g0full.json`: 188 metrics == flags-off, PASS. |
| 10-12 | Fetcher resume / heartbeat / lock / evidence / guard | **MOSTLY FIXED** | Lock + per-pid temps; STEP_DIED; stale STOP_ROLL; dry-run s2p guard FATAL (GUARD-s2p ok); sim + safety after the final code. Heartbeat content stale (R2-9); floor-by-value NIT. |
| 13 | Markers bound to code | **FIXED** | `S2P_PASS` and `G0_PASS` `core_md5` (17 files) == the current files (I recomputed: 0 mismatches); ARMED enforces it. |
| 14 | Switch abort / timeout | **FIXED** (gap) | No timeout while mot_pause.sh is alive; 10-min alerts; `--after-pause`; abort stops RS_J / STOP_ROLL / end_of_run. Empty-id gap: R2-6. |
| 15 | Old launcher runnable | **FIXED** | `launch_pose5.sh:20-22` exits 2. |
| 16 | Older saves pruned unchecked | **FIXED** (17798 gap) | `stage_host.sh:194-209`; R2-8. |
| 17 | codep blind to s2p | **NOT FIXED** (deferred to the user, eval only) | STATUS 07:39 note. |
| N1 | cb0 confirmed / labels | **PARTLY** | Env comment fixed; the old CFG_DIFF is marked SUPERSEDED; the user's choice is still open on the board. |
| N2 | Resume-line greps | **FIXED** | `switch:262-263` (POSE-S2P / G0COND ADDED, append_at); `load_local` asserts the exact missing set. |
| N3 | G0 field coverage / loud | **FIXED** | plan1 11,321 / npl3 12,863 / npl5 16,055 clips, 0 invalid; WARNING + startup count. |
| N5 | Corrupt local 17850 pruned silently | **FIXED** | `stage_host.sh:84` ALERT. |

---

## Checked and found correct

**Merged code (`code-811-s2p` vs `code-811`):**
- Only model / train / data / sbatch / watchdog paths differ. The four core md5s match both markers.
- s2p:
  - unchanged from round 1 (same position T, unshifted, zero init, pose rows only);
  - it now sits in its own group 's2p' with tower_p's LR (3e-4) and clip (1.0);
  - it is in TW_GROUPS and in the 8,1,1 pose-side set (`train_ppm.py:1687`).
- G0COND:
  - SLOT position (asserted `KIND_SLOT`), f = 7 user terms + log(H/W), invalid → 0, bias-free, `index_add` at unique positions;
  - own group 'g0cond' with audio's LR (2e-4) and clip (1.0), in BB_GROUPS, so the frozen-micro check sees no gradient under no_grad.
- Groups:
  - New groups are trailing in M.GROUPS order, so `assert names[-len(added):] == added` holds.
  - The saved schedule's groups exclude them (they are not in the checkpoint's cfg lr). The appended `_last_lr` uses the JOIN lambda at `last_epoch` (= 1.0 at 17850).
  - `warmup_groups` is unchanged (cfg diff shows no change). AdamW betas (0.9, 0.95), wd 0 and eps are shared by all groups.
- Resume: the missing set is exactly `{inp.g0.weight, tower_p.s_emb.weight}` (`load_local` asserts names == params − missing), with fresh AdamW state for them.
- Export side info carries `s2p` and `g0cond` in both branches (decp_bb or not). The ckpt-roundtrip build passes both flags.

**Final env and config:**
- Lines 1-308 are verbatim append3.
- Added lines: npl5 root2, `PPM_S2P=cb0`, `PPM_MAX_STEPS=19382`, the host alias, `PPM_G0COND=1`, `PPM_KEEP_CKPT=1`, `PPM_EXPORT_KEEP=0`.
- `kit/state/CFG_DIFF_switch.txt` (09:35 PT, newer than the env and code) lists exactly 13 keys, all expected:
  - `fresh.pose_root2`, `max_steps 19382`, `resume_from` (link), `run_tag`, `keep_ckpt 2→1`;
  - `s2p cb0`, `s2p_from_step`, `g0cond`, `g0cond_from_step`;
  - `lr.s2p 3e-4`, `lr.g0cond 2e-4`, `clip.s2p 1.0`, `clip.g0cond 1.0`.
- GROUPS added: exactly [g0cond, s2p]. Every other LR, clip, schedule and data-rule key is identical to 27653.

**npl5 (sub-audit, spot-checked):**
- 16 slot keys, with slinky-9 served by slinky-15 and slinky-7 by slinky-7. Dedup is against the plan 1-3 scans, and the counts are recorded.
- Dry run:
  - guard ok for code-811 and code-811-s2p (`GUARD-s2p.json`), append_at [17352, 17641, 17850], 4/4 refusals;
  - ends [17352, 17641, 18022, 19382] == env max_steps;
  - every host has ≥ 74 unused rows, so the "rows short of max_steps" assert cannot fire;
  - history before 18022 equals the 3-plan stream (asserted per host and rank).
- WINDOW files exist; s13 a40 = 8.41 GB. The roll dir has no stale status or markers.
- Every rm in the fetcher, prep and hook is confined. Nothing targets slinky-9; the supervisor runs on slinky-4.
- No pose5 job runs on slinky-10/18 now (squeue).

**Ops (sub-audit, spot-checked):**
- Pre-pause and live gates share `kit/gate_rules.py` (s13: floor 20, margin 3.5, ahead 40); `test_gates.py` agrees.
- The pause path waits on GPUS_FREE / PAUSE_ABORT with no timeout while mot_pause.sh is alive.
- `export_drop.py` refuses w-15973, any step ≤ 17850, other runs' exports and MoT paths, and fails closed.
- `end_of_run.sh` deletes nothing.
- All scripts pass `bash -n` / `py_compile`.

## Could not check
- Live host state: CKPT contents, free space, existing dirs on slinky-15. This is visible only after the MoT pause, so a surprise aborts the stage with MoT paused (accepted by design).
- That eval exactness holds bit-for-bit with slot 9 on slinky-15. See R2-1's tolerance advice.
- The fetcher daemon's peak memory against its step `--mem=8G`. The probe showed step `memory.max` = max.
- Runtime behaviour of any path on the 16 production hosts.
