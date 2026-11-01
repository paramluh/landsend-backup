# pose5 independent audit, round 4 (delta on the LIVE run's companions)

Auditor: an independent read-only agent. It edited nothing, submitted nothing, and touched no running job. Audited 14:26-14:45 PT.

Live jobs:
- trainer: ppm811 31101;
- hold chain v3: 31159 (`kit/hold_chain_v3.sh`, 21:23 UTC);
- watchdog v2: 31160 (`kit/ppm811s2p_watchdog_v2.sh` 21:21 UTC + `kit/wd_decide.py`);
- `kit/restage_dcp_hf.py` (21:09 UTC);
- `kit/gate_rules.py` (21:25 UTC);
- HF upload throttle: hfb-throttle 31153.

Binding decision checked against: main 13:44 PT. Release held N when N is HF-verified 16/16 AND a later save is COMPLETE 16/16. Invariant: an HF-verified save always exists, plus the newest complete node-local save.

**Verdict: no BLOCKER.**
- Nothing in these companions can delete or move MoT state, dcp-16861, or anything outside its literal targets.
- Release of an unverified save without a newer verified one is not possible. Hold releases are regex-checked literal paths run as a direct `srun rm`. The restage tool never deletes.
- The watchdog cannot rewind past the newest HF-verified save of this run, and its loops are bounded.

The SHOULD-FIX items below all sit on failure paths (a bad or partial newest save, or a restage). They are not on today's happy path.

---

## SHOULD-FIX

### R4-1. Resuming from an older save while a numerically NEWER save stays COMPLETE in CKPT: keep-1 prune then deletes the relaunched run's own saves
**Where:**
- `kit/wd_decide.py` (FROM C with C < newest; RESTAGE V);
- `ppm811s2p_watchdog_v2.sh:104-126`;
- trainer `safety_ppm.prune`: keeps the numerically newest `keep`.

**What is wrong.** The watchdog now deliberately resumes from an older save when the newest one is bad (bad-N marker) or only partly complete. The newer dcp-N+1 is "left on disk for main", inside CKPT, with COMPLETE on some or all hosts.

**Failure scenario.**
1. Relaunch from N (hold) or from a restage.
2. The first new save M lands at about N+45..52, wall-clock based.
3. If M < N+1, `prune(keep=1)` on every host holding a COMPLETE N+1 deletes **M** and keeps the stale N+1.
4. M is then never held (not complete 16/16) and never uploaded.
5. The chain also treats the stale N+1 as "a later save COMPLETE 16/16" and releases verified N early (R4-5).
6. The run loses a save cycle. If N+1 sits on only some hosts, saves keep vanishing on those hosts until a save number passes N+1.

**Fix.** Before relaunching from any save older than a COMPLETE dcp-K (K > the resume step) in CKPT on any host, do one of:
- ALERT and do not relaunch (main moves dcp-K out of `ckpt/ppm-v1` by hand);
- with main's OK, have the watchdog rename it on every host to a non-`dcp-<digits>` name. This is a rename, not a delete: prune and the scans ignore it.

### R4-2. BAD marking fires on ANY death before the first new step, so good saves get marked bad and expensive restages follow
**Where:** `ppm811s2p_watchdog_v2.sh:105-110`.

**What is wrong.** Many deaths before the first logged step are not the save's fault:
- a node or NCCL init failure;
- a missing npl5 file while the rolling fetcher refills after a relaunch (B's sim: ~20 s+ before the first step line);
- an OOM at startup.

Each one marks the resume save BAD permanently.

**Failure scenario.**
- The next decision skips that save: an older local save if it is ≥ V, else a RESTAGE of V. The RESTAGE is 737 GB from HF, up to 3 h with the trainer down.
- A second infra death marks the next save bad too.
- Bounded (bad + restage-log means ALERT; MAX_RL 3 per 6 h), but it can cost hours.

**Fix.**
- Mark bad only when the job log shows a load failure: `resume digest mismatch`, an assertion inside `load_local`, or a Traceback before "resumed ... at step".
- Or mark bad only after 2 deaths from the same save.
- Otherwise retry the same save.

### R4-3. Restage destination choice is per host and uncoordinated, so /data can drop below 300 GB
**Where:** `restage_dcp_hf.py:76-80`.

**What is wrong.** Each of the 16 parallel tasks checks `/data free >= total + 350` on its own, at the same moment, and every short host then writes 46 GB to Weka.

**Failure scenario.**
- With /data at ~1,060 GB (one 89 GB export on disk), 16 Weka restages (737 GB) leave about 320 GB.
- With two exports on disk (the ledger allows ≤ 2), about 230 GB, below the 300 GB user floor.
- If another writer is active, worse.

**Fix (any one of):**
- The watchdog decides the destinations centrally: /data must stay ≥ 350 after the WHOLE batch.
- A flock-based reservation file, the same pattern B used for the raw cap.
- Weka only if /data ≥ 350 + 16 × 46.1.

### R4-4. A node-local restage stays on /scratch forever, and nothing alerts on skipped saves
**Where:**
- `restage_dcp_hf.py:76` (node-local when /scratch ≥ save + 53 + floor);
- the docstring "nothing is deleted";
- no `[save] SKIPPED` alert in `hold_chain_v3.sh` or the watchdog.

**What is wrong.** After a node-local restage, the host carries the extra 46 GB, plus whatever the dead timeline left: the bad N+1 in CKPT and the hold of N.

**Failure scenario.**
1. At the threshold (free ≈ 83 GB after the restage), save M fits (needs 53) and leaves ~37 GB.
2. The next save M2 needs 53 GB.
3. `space_check` (keep=1) deletes CKPT names, which frees 0 because the hold has the data.
4. The save is **SKIPPED**.
5. A skipped M2 means no "later save COMPLETE", so M's hold is not released by the v2 rule. The run can go on with no new saves.
6. The trainer logs `!!! [save] SKIPPED`, but no companion alerts on it.

**Fix.**
- Node-local restage only if /scratch ≥ save + 2 × 53 + floor; else Weka (with R4-3's global check).
- Add an ALERT on `!!! [save] SKIPPED` in the trainer log, to the hold chain or the watchdog, once per hour.

---

## NIT

### R4-5. The hold-chain release also deletes MISMATCH hold dirs on other hosts
**Where:** `hold_chain_v3.sh:110` (release) vs `:84-93` ("MISMATCH, left untouched").

`rm_on_hosts hold/dcp-N` runs on all 16 hosts. On a host where hold/dcp-N is an X dir (not this save's hard links, which the comments promise to leave untouched for main), it is deleted too.

**Fix.** Skip the release of N while any X exists for N, or rm only on H hosts.

### R4-6. The v2 release rule counts a watchdog-BAD save as "a later save COMPLETE 16/16"
**Where:** `hold_chain_v3.sh:102` (`newer` taken from `$LIVE`).

N, which is HF-verified, is released early. The data is still on HF, so a later resume needs a restage instead of a local hold.

**Fix.** Exclude `control/ppm-v1/watchdog_pose5_state/bad-*` from `$LIVE` in the release test.

### R4-7. The restage checks small files by size only
**Where:** `restage_dcp_hf.py:96-105`.

Non-LFS files (`train_state.json`, `digest-*.txt`, `COMPLETE`) are checked by size only. The docstring promises "byte-equal to a second HF read". `train_state.json` step == N is checked, and the trainer re-verifies every rank digest at load. Low risk, but fix the docstring or add the check.

---

## Checked and found correct

- **Hold chain v3 deletes** (`hold_chain_v3.sh:38-44`, used at `:86` and `:110`):
  - the only deletes are `rm_on_hosts`: a regex `^/scratch/amartyavarma/ppm/ckpt-keep-pose5/hold/dcp-[0-9]+(\.tmp)?$` and a direct `srun ... rm -rf --one-file-system -- "${p:?}"` (no `bash -c`);
  - plus `rm -f "${ST:?}/unheld-${s:?}"` on its own state files.
  - It never touches CKPT, ckpt-keep (hfb's holds), MoT paths or 16861.
- **Release rule** (`:99-115`): N is released only if (N HF-verified 16/16 AND a later save COMPLETE 16/16 in CKPT) or a later save is HF-verified 16/16, which is the v1 rule.
  - `verified()` = `done/dcp-N-final == ok` plus a PASS verify record per physical host.
  - This matches main's 13:44 decision.
  - The hfb-watch hold (`ckpt-keep/ppm-v1`) and the chain hold (`ckpt-keep-pose5/hold`) are separate hard-link dirs, so there are no shared paths and no race on names.
  - The trainer's keep-1 prune and `space_check` only remove CKPT names; the data stays in whichever hold has it.
- **Alert spam is controlled.**
  - NOSPACE is debounced: alert after 30 min or below 50 steps of lead.
  - The disk trend is per save cycle and starts only after the first release (replays of the 12:40 false positives give 0).
  - Every alert key is limited to once per hour.
  - The WKDF df-sruns only run for Weka hosts whose roll JSON lacks `dest_free_GB`; all three live JSONs carry it, so no extra sruns happen.
- **Watchdog v2 / `wd_decide.py`:**
  - The scan must answer on 16/16 hosts, else ALERT.
  - "Complete" means on all hosts in one place (CKPT or hold) and not bad.
  - V is HF-verified (done marker + per-host PASS) AND its hfb `dcp` record names one of this run's jobs, so another run's saves never count.
  - It resumes from local C only if C ≥ V; otherwise it restages V; otherwise ALERT. It never goes back to 17850 once any save > 17850 exists.
  - Loops are bounded: bad + restage-log means ALERT, plus MAX_RL 3 per 6 h, plus the same-step cap.
  - It relaunches from an explicit dir (`PPM_RESUME_FROM`) and never from `auto`.
- **`restage_dcp_hf.py`:**
  - Its source is the hfb PASS record's `shard_repo/path/<physical host>`.
  - Every LFS file must match HF in size and sha256, `train_state.json` step must equal N, and `COMPLETE` is renamed into place last.
  - It writes to a tmp dir, then renames; an existing destination that does not verify means ABORT, never a replace.
  - The per-host link is replaced only if it is a symlink (a real dir means ABORT), and N > 17850 always.
  - Its only delete is its own `.dl` cache, and it never runs on slinky-9.
  - The srun is a new 16-host allocation (`-c8 --mem=16G -t 3:00:00 --immediate=900`) while the trainer is down.
- **hfb-throttle 31153:**
  - The code (`hfb_throttle.py`, 10-20) is unchanged.
  - It runs on slinky-4 (`-c1 --mem=2G`) and only writes `setup-logs/hfb/control/<host>[.data].workers`; no srun, scancel or delete.
  - It is the same command the 27653-era job 26111 ran.
- **Switch** (future runs): ARMED requires exactly one `hfb-throttle`, or starts `kit/hfb_throttle.sbatch` with `need_id`; two or more is red. It submits `hold_chain_v3` and watchdog v2, and requires `wd_decide.py` and `restage_dcp_hf.py`.
- **Syntax:** `bash -n` is OK on `switch_to_pose.sh`, `hold_chain_v3.sh` and `ppm811s2p_watchdog_v2.sh`; `py_compile` is OK on `wd_decide.py` and `restage_dcp_hf.py`.
- **`gate_rules.py`:** `saves()` is unchanged. `WEKA_HOSTS` now includes slinky-6, slinky-12 and slinky-13 (`wekahosts` subcommand).
- **Live state (read-only):**
  - roll JSONs fresh, ahead 99-100, no alerts;
  - the hold-chain log shows 17903 and 17956 held 16/16;
  - one gn spike alert (13:21) was a real event;
  - no spurious alerts since v3 started.

## Not checked
- Runtime behaviour of the restage and watchdog v2 paths on the 16 hosts. These only fire on a failure. The tests were fake-srun DRY runs plus a small-file restage on a spare.
- hfb-watch internals beyond the hold-dir naming.
