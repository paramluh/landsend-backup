# motguard review of go/mot_stop.sh + go/mot_resume.sh (10-26)

From motguard (tmux motguard). I read both scripts and spot-checked the key lines. **Verdict: OK to use on main's relayed GO, with the one change in (4a).**
The mechanism is the same one motguard used for the 10-23, 10-24 and 10-25 stop/resume cycles:
- the trainer writes STOP_AFTER_SAVE, then the `[control] … saved` and `STOPPED at step S` lines;
- the submission matches swap_exec.sh;
- `JOB`, `watchdog/CURRENT_JOB` and `LAST_LAUNCH.env` are rewritten after submit.

The stop never scancels. On timeout it hands over to motguard (stacks first). On SKIPPED it retracts cleanly.

## (1) Markers
Let the scripts set and remove `watchdog/WATCHDOG_OFF` and `guard/PAUSED` as written:
- `mot_stop.sh` sets both before STOP_AFTER_SAVE;
- `mot_resume.sh` removes PAUSED at submit and WATCHDOG_OFF after verification.

That matches what motguard did itself on 10-24 and 10-25. motguard will not touch them in parallel.

## (2) Final HF state backup of dcp-S
Not needed. Two reasons:
- HF `InternalCan/ppm-mot-14b-state` always holds the newest 2-hourly dcp, so it is at most about 2 h behind S.
- dcp-S stays node-local on all 16 hosts and is protected (`/scratch/amartyavarma/ppm-mot/*` in PROTECTED_SCRATCH.txt).

A dcp-S upload during the pause would need a 16-host holder allocation (2 CPUs and 12 G per host for about 15 min) competing with the FT. Skip it unless the user asks. The first hourly save after the resume is backed up by the normal 2-hourly cycle.

## (3) What acts on the hosts during the pause
**Nothing.**
- **motguard tick in PAUSED mode:** reads only squeue, df /data and status files. There is no srun, and the 10-min host check runs only while a ppm-mot job is RUNNING.
- **ppm-mot-watchdog 28257:** inert while WATCHDOG_OFF is set.
- **mot-hf-watch 28265 (slinky-4):** idle, because there are no new exports.
- **Export /data drop:** does not run in paused mode.
- **mot-state-bk (slinky-18):** idle without a RUNNING ppm-mot job, but see (4a) for one race.

## (4) Corrections
- **a) Add `touch /data/home/amartyavarma/ppm/mot/guard/state_backup/OFF` in mot_stop.sh phase 2,** after the "no cycle in flight" wait and before writing STOP_AFTER_SAVE.
  - Remove it in mot_resume.sh together with WATCHDOG_OFF, after verification. Only remove it if mot_stop created it: record `SBOFF_PRE` in go state.
  - Why: the stop save is a fresh `[save]` line while the job is still RUNNING for about 10 s. If the 2-hourly cadence is due at that moment, the driver can start a cycle on dcp-S: scan or hold via srun --overlap into a job that is exiting. It then ends in SKIP, or in a hold that waits until the T+100 min deadline and is released through tiny allocations on hosts the FT occupies.
  - It is harmless for data, since the hold is hardlinks of the kept dcp-S at 0 B. It is messy, and OFF avoids it. OFF also aborts nothing, because the wait just before guarantees no cycle is in flight.
- **b) mot_resume pre-check:** also refuse if any host still has `/scratch/amartyavarma/ppm-mot/hfhold/*` (a leftover hold from the race above). Your scan already reports SHOLD; make it a `fail`, not a `warn`, at resume.
- **c) FT side:** never write into, or delete anything under, `/scratch/amartyavarma/ppm-mot/**`. Also leave ≥ 64 GiB free on every one of the 16 hosts at FT end, or the resume pre-flight aborts (mot_resume already checks this).
- **d) After the resume, expect motguard tick lines** "job PENDING / no step yet" for up to about 10 min. These are informational. motguard verifies digests == saved and step continuity on its own as well.

motguard will log the stop and the resume in ppm/mot/GUARD_LOG.md and re-arm its normal 5-min tick when guard/PAUSED disappears.
