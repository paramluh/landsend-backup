# pose5 independent audit, round 1

Auditor: an independent read-only agent. It edited nothing and submitted no jobs.

Audited state: files as of 10-27 07:15-07:55 PT. The data and rolling-fetcher part was done by a read-only sub-auditor; I re-read the code behind its two BLOCKERs myself.

Not present yet, so not audited:
- G0COND merged into code-811-s2p;
- S2P_PASS (G0_PASS in code-811-g0 appeared at 07:31);
- the npl5 plans, npl5/dryrun and SWITCH_HOOKS.env;
- kit/state/CFG_DIFF_switch.txt.

Severity:
- **BLOCKER**: fix before --go.
- **SHOULD-FIX**: fix before --go, unless main accepts the risk in writing.
- **NIT**: cosmetic or low risk.

Paths are relative to `/data/home/amartyavarma/ppm/pose5/` unless they are absolute.

---

## BLOCKERS

### 1. The watchdog's RESUME_PIN can silently rewind the run to dcp-17850
**Where:** `kit/ppm811s2p_watchdog.sh:84`

```bash
if [ -n "$RESUME_PIN" ] && { [ -z "$s" ] || [ "$s" -le "$PIN_STEP" ]; }; then pin=$RESUME_PIN; s=$PIN_STEP; fi
```

**What is wrong.**
- `s=""` means the scan FAILED, not that no save exists. `scan_complete` returns "" whenever fewer than 16 hosts answer: a NODE_FAIL host, the `--immediate=300` allocation failing, or a timeout.
- The original `ppm811f_watchdog.sh` answers "" with "ALERT, no relaunch". This copy relaunches from dcp-17850 instead.
- The scan also ignores `ckpt-keep-pose5/hold`.

**Failure scenario A.** At step ~19,000 a node fails.
1. The scan returns "", and the watchdog submits a job with `PPM_RESUME_FROM=<pose5-resume/dcp-17850>`.
2. That job resumes about 1,100 steps back.
3. Each of its new saves (17902, ...) sits next to the real newest save dcp-19xxx. `prune(keep=1)` keeps the numerically newest save, so the new saves are deleted.
4. The run then needs npl5 files the fetcher has already dropped. It crash-loops into the same-step cap.

**Failure scenario B.** A host is short of /scratch at save time (see #4 and #6).
1. `safety_ppm.space_check` with keep=1 deletes every unprotected CKPT name (`len(s) > max(keep-1, 0)`). That frees 0 bytes, because the hold hard-links the data, so the save is skipped.
2. CKPT now holds nothing newer than 17850. The next relaunch "succeeds" from 17850.

**Fix.**
- Pin ONLY when the scan returned 16/16 END lines AND no `dcp-N` with N > PIN_STEP exists in CKPT or in `/scratch/amartyavarma/ppm/ckpt-keep-pose5/hold` on any host.
- On a failed scan: ALERT, no relaunch (the original behaviour).
- If a newer save exists only in the hold: ALERT for a human re-link.

### 2. The npl5 fetcher's memory guard will most likely stop fetching for good, with no alert
**Where:** `npl5/bin/npl5_roll.py:251-270`: `job_headroom_gb()` = the job cgroup's `memory.max - memory.current`; `one_pass` returns early when it is below 6 GB.

**What is wrong.**
- In cgroup v2, `memory.current` includes page cache. Every save writes about 46 GB per node into the trainer job's 560G cgroup, and the job also reads parquet. Cache is reclaimed only near the limit, so headroom is likely to stay under 6 GB after a few hours.
- The early return happens BEFORE the status JSON and the BEHIND/CRIT/STARVED markers are written. Only `npl5/roll/<host>.LOWMEM` and a log line appear.
- `kit/hold_chain.sh:85` alerts only on a JSON younger than 900 s, so a stale JSON means silence, and the live page hides the host.

**Failure scenario.** Fetching stops on all hosts. Each trainer dies at its first missing npl5 file, losing up to 1 h of steps since the last save. After a relaunch the new cgroup lets fetching resume until the cache refills. With MAX_RL=3 per 6 h, the watchdog then stops relaunching.

**Fix.**
- Compute headroom as max - (current - file - inactive_file) from `memory.stat`, or rely on the step's own `--mem=8G` cap.
- ALWAYS write the status JSON and a marker on LOWMEM.
- Add LOWMEM and stale-JSON alerts in hold_chain.

**Evidence gap.** The safety test's status JSON (`npl5/safety/run-1793109929/roll/slinky-0.json`) has no `job_mem_headroom_GB` key. So the guard claimed in STATUS ("NEW job-cgroup headroom guard") never ran in any test.

### 3. The run's export rotation will delete the shared export w-15973
**Where:**
- the env, verbatim from append3, lines 98-104: `PPM_EXPORT_EVERY_MIN=360`, `PPM_EXPORT_ROOT=/data/home/amartyavarma/ppm/ckpt-weka`, `PPM_EXPORT_KEEP=2`;
- `safety_ppm.py` `export_weights`: `for w in ws[:-keep]: shutil.rmtree(...)`.

**What is wrong.**
- `ckpt-weka/ppm-v1/` holds only `w-15973`.
- This run is about 31 h on run_base ppm-v1, so it exports about every 6 h.
- At the second pose export (~12 h in), the rotation rmtrees `w-15973`.
- 27653 never lived 6 h, so it never hit this.

**Who uses w-15973:**
- `ppm/aloe-zoom-cotrain/*.env`;
- `ppm/code-frozenbb/*.env`;
- `ppm/aloe-v2vox-asr-w15973*/`;
- `ppm/asr-adapter/w15973/*`;
- the Aloe reference FTs (memory).

This breaks "never delete checkpoints / no deletion without main" and cannot be undone.

**Fix.**
- Add one ops line in [POSE5-OPS]: `PPM_EXPORT_KEEP=0`. In this function 0 means delete nothing. The value is read from os.environ, not CFG, so there is no cfg-diff change and no maths change.
- Main prunes pose exports by hand. Alternatively, `PPM_EXPORT_EVERY_MIN=0` if the user wants no exports from this run.
- Add the exports (~80 GB per 6 h) to the /data ledger. The trainer refuses an export that would leave less than 300 GB free, so the floor holds, but exports can be refused late in the run.

### 4. A host that loses its CKPT save makes the fetcher re-download all consumed npl5 data, with no disk floor
**Where:**
- `npl5/bin/npl5_roll.py:276`: `floor = min((min(sv) ...) if sv else a.start, live)`;
- `saves()` at line 211 scans CKPT only.

**What is wrong.**
- With no COMPLETE save above 17850 in that host's CKPT, `floor` falls back to 17850 while `live` comes from the log (say 18,500).
- The pass then re-fetches every npl5 file read since ~18022, tens of GB, onto a host with no free-space check.

**Trigger.** It is exactly the space_check scenario in #1B: keep=1, plus the hold, plus a short host. slinky-13 has a ~6 GB margin.

**Consequence.** /scratch fills, the trainer's save or write hits ENOSPC, and then #1 rewinds the run.

**Fix.**
- Make `floor` monotonic: persist it, and lower it only on a "resumed ... at step N" log line.
- Include hold and pin dirs in `saves()`.
- Refuse to fetch below a /scratch floor (one save + 20 GB).

---

## SHOULD-FIX

### 5. The resume link on slinky-1/2/7/14 points INTO CKPT, so keep-1 pruning leaves it dangling
**Where:** `kit/stage_host.sh:92` (`SRC=$D`, `D=$CK/dcp-17850`) and line 131 (`ln -sfn $SRC $RL`).

**What is wrong.** At the first COMPLETE pose save, `prune(CKPT, keep=1)` rmtrees `CKPT/dcp-17850` on those hosts. The data survives through the pin, but the link dangles.

**Failure scenario.** Any later relaunch pinned to 17850 fails on 4 hosts. Examples: #1's paths, a deliberate rollback, or a save failing its digest.

**Fix.** After a successful pin, set `SRC=$PIN/dcp-17850` (the same inodes, outside CKPT).

### 6. The chained hold, a condition of main's KEEP=1 approval, can fail silently, and nothing watches it or the HF watcher
**Where:**
- `kit/hold_chain.sh:55`: an incomplete HOLD is logged only in `kit/state/hold_chain/hold_chain.log`;
- `hold_chain.sbatch` is a single job on slinky-4 with no restart;
- hfb-watch 30966 runs on slinky-10, which goes to motasr2 at the switch.

**What is wrong.** The trainer prunes N when N+1 is COMPLETE, held or not.

**Failure scenarios.**
- If the chain job is dead, KEEP=1 runs with no safety net.
- If hfb-watch dies, nothing is verified, holds pile up at 46 GB each, saves get SKIPPED, and #1B and #4 follow.

**Fix.** Write ALERTS.log and overnight_monitor lines for each of these:
- a save > PIN_STEP is COMPLETE on 16 hosts but not held after 2 polls;
- no HF verify within 2 h of a hold;
- the hfb-watch job is not RUNNING;
- the HC job is not RUNNING (the watchdog can check this from LAUNCHED_RUN).

### 7. The new parameters probably dominate their clip groups; G0 can then change speech-embedding updates
**Where:**
- `train_ppm.py:1731`: per-group `clip_grad_norm_`, clip 1.0 for every group;
- `PPM_TOWER_CADENCE=step`.

**What is wrong.**
- `inp.g0` joins group 'audio', together with `inp.aud.*` (the speech code embeddings of both channels, used by podcast speech rows).
- `tower_p.s_emb` joins group tower_p.
- **The numbers:**

  | measurement | source | value |
  |---|---|---|
  | gn_audio, whole group, per step | 27653, steps 17848-17850 | 0.082-0.088 |
  | gn_tower_p, per step | 27653, steps 17848-17850 | 0.037-0.040 |
  | inp.g0 grad norm on ONE real-width pose_full micro | D, `code-811-g0/tests/g0/out/CMP_real_orig_vs_on0.json` | 0.275 |
  | s_emb grad norm on one micro | tiny config | 0.039 |

  So g0 alone carries more than 3x the whole audio group's norm.
- If the step's audio-group norm passes 1.0, the clip rescales `inp.aud`/`inp.pose` updates. That would be G0 touching speech maths, which the user's rule forbids.
- It does not bind at the current scale, but nothing guarantees that at full depth or after the zero-init phase.

**Fix.**
- Before the switch, get the real-width s_emb norm from A's run.
- In switch step 6, also assert `train/gn_audio` < 0.5 and `train/gn_tower_p` < 0.5 at steps 17851/17852, and alert during the run if either passes 0.8.
- **Main/user decision:** give inp.g0 (and s_emb) their own optimizer group with the SAME LR, schedule and warmup but their own clip. The trainer already supports trailing added groups (`pose_head`, `decp_bb`).

### 8. The cheap in-production proof is lost: the START EVAL compares against the wrong run
**Where:**
- env line 241: `PPM_811_EVAL_REF=/data/home/amartyavarma/ppm/runs*/ppm*-j*/metrics.jsonl`;
- `train_ppm.py:1606-1611`: the last matching file wins.

**What is wrong.**
- For 27653 the START EVAL gave "184 common metrics, max |diff| 0, exact matches 184".
- At step 17850:
  - 27653's own record has no eval keys;
  - `runs/ppmdecay-j27810` is the right reference: 184 keys, same dcp-17850, same slots, code-811 maths;
  - `runs/ppmvox-v5-j29813`, a different model, sorts last and wins.
- With PPMVOX only 69 keys are common, all text. Pose and codependence keys are never compared, yet the log will say "exact matches 69".

**Fix (no env or maths change).** In switch step 6, compare:
- the new step-17850 eval with ppmdecay-j27810's 17850 record over all 184 keys (expect max |diff| 0);
- the new `[811 rows] ... step 17851/17852` lines with `ppm/logs/ppmdecay-27810.log` (same slots, flags off).

This proves "zero-init = today's model" and "rows from 17850 = 27653's rows" in the deployed layout.

### 9. The evaluate() path with the flags on has never run, and it runs at every (re)start before step 1
**Where:** `train_ppm.py:1602` `d_start = evaluate(start)`; D's `codep_g0()` (asserts clip ids equal `cdp["info"]["clips"]` and reads `cdp["base"]["gpos"]`).

**What is wrong.** All A and D proofs drive `run_micro` only.

**Failure scenario.** An assert or KeyError in the start eval kills every launch before step 1, with MoT already paused, and the watchdog hits its same-step cap.

**Fix.** On the merged code, run evaluate() once in the CPU harness with PPM_S2P=cb0 and PPM_G0COND=1 on real samdata eval rows (`slot9/posestore/rows` is on /data).

### 10. Fetcher: a resume from a held or pinned save crash-loops, and the slinky-13 "newest" claim is wrong
**Where:** `npl5_roll.py:211-217`, `234`, `276`; STATUS.md "B rolling fetcher sim PASS" (lines 271-272).

**What is wrong.**
- `saves()` ignores the hold dir.
- Until the new job logs a step, `live` = the newest CKPT save.

**Failure scenario.** Resuming from held save N (because N+1 is bad, or N+1 is COMPLETE on slinky-13 only):
- the rows for [N, N+1) are already dropped;
- the first step crashes before logging, so the fetcher never refetches;
- the run loops into the same-step cap.

STATUS's "rows <= 53 steps refetched in ~1-2 min" is therefore false. With keep=1, "newest" and "oldest" give the same floor anyway.

**Fix.**
- Include the hold and pin dirs in `saves()`.
- Parse "resumed ... at step" for `live`.
- Drop BACKMODE newest.

### 11. Fetcher: other ways it goes quiet, and two copies can run on one host
**Where:** `npl5_roll.py:176`, `191-198`, `283-298`; `npl5_roll_super.sbatch:2`, `20-38`.

**What is wrong.**
- Status is written only after a full pass. Under 429s (60 s pauses, 12 attempts per file) that can be hours, and the stale JSON mutes the alert.
- A crashed daemon step (`--kill-on-bad-exit=0`) stays dead with no signal.
- A leftover `npl5/state/STOP_ROLL` makes the supervisor exit at once.
- There is no lock or singleton. Every writer uses the same temp name `dst + ".roll"`, so two daemons (a second switch run, or an orphaned step) can rename a partial file into the trainer's name.
- `wait $SP` can hang forever, so STOP_ROLL never works.

**Fix.**
- A timed heartbeat.
- Alerts on a stale or missing JSON.
- Fail the switch if STOP_ROLL exists.
- `flock` on the npl5 dir, pid and thread ids in temp names, and `--dependency=singleton` on the supervisor.

### 12. Fetcher evidence does not cover the code that will run, and the dry run's guard check does not fail it
**What is wrong.**
- **Code newer than every test.** `npl5_roll.py` was modified at 14:06:52 UTC. The sim ran 13:37-13:52 and the safety daemon started 14:06:19.
- **Sim gaps.**
  - The sim record still holds the false STARVED marker; the fix was never re-simulated.
  - It used keep=2, where production uses keep=1.
  - It never covered a relaunch with no step line yet, newest mode, or a real Slurm step.
- **Safety-test gap.** "npl5-roll steps left: 1" proves nothing, because `sacct -s R` also lists finished steps.
- **Dry-run guard.** In `dryrun_offhost.sh:37,41` the code-811-s2p guard result is non-fatal, so `dryrun.ok` is still written.

**Fix.**
- Re-run the sim and the safety test on the final `npl5_roll.py`, including keep=1, a relaunch from a held save, and a LOWMEM case.
- Make the guard fatal in the dry run.

### 13. The PASS markers are not bound to the code they certify
**Where:** `switch_to_pose.sh:58-61`; `code-811-s2p/tests/s2p/make_pass.py`; `code-811-g0/tests/g0/G0_PASS` (no hashes).

**What is wrong.** S2P_PASS is computed before the G0 merge. The merge rewrites model_ppm.py, train_ppm.py and data_ppm.py, with 7 same-line conflicts, yet a stale S2P_PASS stays valid.

**Fix.**
- Record the md5 of the 4 core .py files in each PASS JSON.
- ARMED compares them with the current files.
- Re-run the combined suite after the merge.

### 14. Switch: the abort and timeout paths leave things running or stranded
**Where:** `switch_to_pose.sh:125-131`, `156-158`.

**What is wrong.**
- **Stage-failure abort.** It cancels `$J $WD $HC $EL`, but not the roll supervisor `$RS_J`, not STOP_ROLL, and not `end_of_run.sh` (which waits forever).
- **"GPUS FREE" wait.** It is a hard 20 min, but mot_pause.sh may take longer (up to 3 backup attempts).
  - The switch then dies while the pause continues.
  - The hosts sit idle, and a re-run of `--go` calls mot_pause.sh again (flock REFUSED or nogo).

**Fix.**
- Abort path: add `scancel $RS_J`, `$B_ROLL_STOP`, and stop end_of_run.
- Keep waiting while mot_pause.sh is alive, alerting every 10 min.
- Add an `--after-pause` entry that skips step 1 when `mot/guard/stop/GPUS_FREE` exists.

### 15. The superseded launcher is still executable
**Where:** `launch_pose5.sh --go --stop-mot [--free-disk]`.

**What is wrong.** It contradicts the current decisions:
- it stops MoT by touching MoT's STOP_AFTER_SAVE directly, not through mot_pause.sh;
- `SPARE=slinky-18`;
- it resumes from `CKPT/dcp-17850` inside CKPT;
- it has a `--free-disk` deletion path;
- its `npl5_launchday.sbatch` uses `-N16 -x slinky-9` with no `-w`, so it can land on slinky-10/18.

**Fix.** Make it exit at once with "superseded by switch_to_pose.sh".

### 16. Older saves in CKPT get deleted without the HF re-check main required
**Where:** `kit/stage_host.sh:208-210` refuses only NEWER saves.

**What is wrong.** With KEEP=1, the first pose save prunes every older COMPLETE `dcp-N` except 16861 (PROTECT_STEPS; checked from_step 16861). Main approved deleting only dcp-17798, after a live HF re-check.

**Fix.** List `CKPT/dcp-*`. Anything other than {16861, 17798, 17850} needs a PASS verify record, or the stage aborts and lists it.

### 17. The codependence speech->pose probe is blind to s2p (eval only)
**Where:** `train_ppm.py:487-498` (towers keep the REAL `depth`; only `depth_t` varies); `model_ppm.py:562-563` (s_emb reads `b["depth"][1]`).

**What is wrong.** In speech_shuf and speech_sil, the pose tower still gets the real speech through s_emb. So the probe that should show the user's fix measures only the trunk path.

**Fix.** User or main decides. Add NEW codependence keys in which s_emb reads `depth_t[1]`, and keep the old keys for comparison with 27653.

---

## NITs
- **N1. s2p variant not confirmed.** Board t1027-24 still says "YOU: full vs cb0 (asked 06:35 PT)". The env uses cb0, which is main's default. Confirm that the user chose it.
  - The env comment [POSE5-2] says "B's 32 speech codebooks" while the value is `PPM_S2P=cb0`.
  - `CFG_DIFF_append5_s2p_vs_27653.txt` is stale (s2p full, no keep_ckpt, no g0cond). Mark it superseded.
- **N2. cfg_diff misses resume-filled keys.** `test811/cfg_diff.py` diffs only the import-time CFG. These keys are never diffed: `fresh.pose_append_at`, `fresh.pose_digests`, `s2p_from_step`, `g0cond_from_step`, `s811.from_step`. Switch step 6 should also grep "POSE-S2P ADDED at step 17850", "G0COND ADDED at step 17850", and the `NOT in the checkpoint` list (exactly `inp.g0.weight` and `tower_p.s_emb.weight`).
- **N3. G0 features.**
  - Aspect-ratio coverage is evidenced for plan 1 and npl5 (150 files each, 0 invalid).
  - I checked one samdata record (`slot9/posestore/rows/v3-0248.words.arrow`): it has orig_h, orig_w, frame0_root, frame0_width and frame0_ok.
  - npl2 and npl3 have no evidence. A missing key silently gives f = 0 for the whole clip. Log the valid fraction per plan.
  - The features are unclipped: |f| reaches 42, and the 99.9th percentile is ~16. That follows the user's spec.
- **N4. `verified()` keys on step only.** In `hold_chain.sh:33-39` it uses `done/dcp-M-final`. No markers above 17850 exist today (checked). Also require the ppm-v1 lineage.
- **N5. A failed node-local copy is left in CKPT.** If a node-local dcp-17850 fails its sha check, it stays in CKPT unpinned, and keep-1 prunes it. That is a dcp-17850 deletion main never approved. Log it to main.
- **N6. Fetcher small items.**
  - The fetcher imports `fresh_pose` from code-811, not code-811-s2p (identical today).
  - `present()` is size-only.
  - `.roll` partials are never cleaned up.
  - There is no realpath check before `os.remove`.
  - The spare-node test scripts `rm -rf /scratch/amartyavarma/npl-append` (`safety_test.sh:41` unguarded).
- **N7. Shared-infra edits.**
  - Additive: `snap_code.sh` (one new case branch), `ppm_doc.py` (`roll_ahead`), `ppm.tpl.html` (one header span).
  - `hfb_watch.sbatch` changed its host pin, slinky-15 to slinky-10, plus the resubmit pin. That is NOT additive, but main ordered it, and the diff is exactly that.

---

## Checked and found correct
- **Baseline code.** `ppm/code-811` model/train/data/safety/s811 are byte-equal to hfb `codesnap/27653`. code-811-s2p differs from that snapshot only in model_ppm.py, train_ppm.py, the sbatch and the watchdog paths.
- **s2p maths.**
  - With the flag off there is no module, no key and no cfg key, and the op order `(e + ce_ + cnd)`, then `[None]`, is unchanged.
  - s_emb reads `b["depth"][1][T]` unshifted. That is the Mimi of frame T-CD, the same frame as the label `pcb0[T] = pdepth[T,0]` (pose_delay = cd = 2 in 27653's cfg), so there is no extra delay. The tower's own codes stay `shift1`.
  - This is identical to `ppm/pose-s2p/code/model_ppm.py`. Training has no CFG dropout to port.
  - `code_feedback` is `F.embedding_bag`. The cb0 slice uses global ids 0..2047. -1 gives 0. It uses bf16 like `code_emb`.
  - It runs on pose rows only, with zero init, group tower_p, AdamW `wd=0`, and per-parameter step counters.
  - Resume: only `tower_p.s_emb.weight` may be missing, it gets fresh AdamW state, digest checked; it is strict afterwards, and the other mode is refused.
  - CPU evidence: all 12 CMP files PASS (flag off 0/92 grads differ vs the original; zero-init bit-identical; non-zero affects pose rows only), and resume A-L PASS for full and cb0.
  - The real-width (b-ii) run is still RUNNING.
- **G0COND (code-811-g0).**
  - Flag off: no module or key, and `CFG['data']` is unchanged, since `pcfg` is a copy.
  - The SLOT position is `pose_s3q` gpos (asserted `KIND_SLOT`).
  - f is the user's 7 terms plus log(H/W), with r and w image-normalised (`fresh_pose.py:131-141`). Invalid gives 0, and the projection is bias-free.
  - It adds to the TRUNK input by `index_add` at unique positions, in group 'audio'.
  - Real-width zero-init: identical loss on all 5 row types, 0/92 grads differ. Resume A-I PASS. G0_PASS written at 07:31 PT.
- **Config.**
  - Env lines 1-308 are verbatim append3.
  - The alias rank order puts slinky-15 at index 7 and slinky-7 at index 8. That equals dcp-17850's `hosts`, which `load()` asserts.
  - The LR is constant (warmup 100), and the data-rule knobs are in the verbatim region.
- **Stream continuity (code).** Tail=1 concatenation plus the prefix and digest asserts give 17850..18021 = 27653's rows, then each rank's npl3 tail, then npl5. All 48 off-host plan-1/npl2/npl3 digests equal dcp-17850's.
- **Slot keying.** slinky-15 uses key slinky-9; slinky-7 uses key slinky-7. The slot-9 hook's digests equal dcp-17850's `cfg.fresh`.
- **Missing npl5 file.** Nothing catches the error (FreshRows, `data_ppm`, the DataLoader). The job dies through `--kill-on-bad-exit=1`. No row is skipped or substituted, and the segment_id check catches wrong content.
- **Ops safety.**
  - The stage is fail-closed (16 OK lines, ppm811 `afterok`). DependencyParameters is null, so a failed stage leaves ppm811 PENDING, which the watchdog ignores.
  - Nothing at switch time targets slinky-9. Companions use `-w slinky-4`.
  - The 4 node-local copies are verify-only. slinky-15 is hard-linked from the sha-checked v5 copy. The Weka copy is 11 dirs, files r--r--r--, VERIFIED.json present.
  - The dcp-17798 drop re-checks HF live, and hfb released its 17798 hold on 10-23 04:43, so the drop frees real space.
  - Every rm in stage_host and hold_chain uses constant roots or regex-checked step names. Fetcher deletions are confined to `npl-append/npl5/jobs`.
  - dcp-16861 is protected by PROTECT_STEPS.
  - The switch matches mot_pause.sh's strings, and without `--free-on-backup-fail` MoT stays resumable.
  - 27653's log has 0 lines that would trip the switch's error grep.

## Could not audit (round 2)
- The G0 merge into code-811-s2p: the export `side` dict with both s2p and g0cond, and the `PPM()` and `build()` signatures.
- The combined tests, S2P_PASS, and A's real-width results.
- `CFG_DIFF_switch.txt`.
- The final npl5 plans, dry-run RESULT and GUARD, the per-host windows, PPM_MAX_STEPS, and SWITCH_HOOKS.env.
- How full the live job cgroup's page cache gets, which decides how fast #2 fires. Check `memory.stat` on any long-running job.
- The live host state: CKPT contents, free space, and slinky-15's existing `/scratch` dirs. These are visible only after MoT pauses, and a surprise there aborts the stage with MoT already paused. That residual risk is by design.
