# MoT planned pause / resume: mechanism, companions, disk (monosam, 10-26 05:25 PT)

STAGED ONLY. Nothing here has stopped, signalled or relaunched MoT. Run only when main relays the user's explicit go.

- Stop: `go/mot_stop.sh` (`--dry-run` first).
- Resume: `go/mot_resume.sh` (`--dry-run` first).
- Probe used by both: `go/mot_hostcheck.sh`. It is read-only and runs as one task per host.
- Shared helpers: `go/mot_common.sh`.

## 1. Stop mechanism (what MoT actually honours)
- **Control file.** `touch /data/home/amartyavarma/ppm/mot/control/STOP_AFTER_SAVE`.
  - The control dir is `mot.env` `PPM_CONTROL_DIR`, which equals the `mot.sbatch:30` default.
  - The trainer logs the same dir at start: `[safety] checkpoints ... control /data/home/amartyavarma/ppm/mot/control/SAVE_NOW | STOP_AFTER_SAVE` (log line 163 of 30289).
- **Polling.** Rank 0 checks the file at every step boundary. The flag is all-reduced with MAX (`code/safety_ppm.py:321-328`, `train_ilp.py:1519`).
  - `want_save` includes `c_stop` (`train_ilp.py:1529`).
  - The save is at `:1533`, the prune at `:1535`.
  - The file is consumed at `:1546`: it is renamed to `STOP_AFTER_SAVE.done-<step>` (`safety_ppm.py:330-335`).
  - Then `break`, and the log says `STOPPED at step S (clean exit)` (`:1548`, `:1568`). The exit code is 0.
- **USR1 is equivalent.** `scancel -b -s USR1 <job>` makes the batch shell's trap touch the same file (`mot.sbatch:40`; `--signal=B:USR1@300` at `:4`). SIGUSR1 to a rank does the same (`safety_ppm.py:35-39`).
  - The scripts use the file. The documented stop is `mot.sbatch:8` and OPS_PLAYBOOK "Stop".
  - Precedents: `guard/hostswap/swap_exec.sh stop`, and the 10-24 stops at 3270 and 5829.
  - Never use a bare scancel.
- **Save skipped.** If `/scratch` is short, the save is SKIPPED. The log says `!!! [control] STOP_AFTER_SAVE at step N: the save was SKIPPED ... NOT stopping` (`train_ilp.py:1541-1544`), and the run keeps training.
  - Space rule (`safety_ppm.py:89-128`, `PPM_SAVE_PRUNE_FIRST=0`): need = 33.673 GiB × 1.002 + 30 GiB floor = **63.74 GiB free per host**.
  - Today's minimum is 73 GiB on slinky-13, so the stop save fits.
  - `mot_stop.sh` refuses up front if any host is short. If a skip still happens, it retracts the request, restores the markers and exits 3.
- **No export at the stop.** `PPM_EXPORT_EVERY=0` (`train_ilp.py:1537`), so the stop writes no export. The newest bf16 export is w-9828 (05:15 PT, HF upload submitted 05:17 PT).

## 2. How a save is marked complete
- **Per node.** `safety_ppm.save_local` (`:131-162`): every rank writes `dcp-S.tmp/rank<R>.pt` and `digest-<R>.txt` → barrier → local rank 0 writes `train_state.json` and `COMPLETE` (content = S), then renames `.tmp` to `dcp-S` → barrier → rank 0 logs `[save] .../dcp-S node-local per-rank state (16 s)`.
  - That log line comes after the post-COMPLETE barrier, so it proves COMPLETE on all 16 hosts. `mot_stop.sh` waits for it.
  - After the job is gone, a tiny read-only allocation re-checks every host: 20 files, COMPLETE = S, newest = S, no `.tmp`.
- **Layout per host.** `/scratch/amartyavarma/ppm-mot/ckpt/ppm-mot/dcp-S/` holds 20 files: 8 `rank*.pt`, 8 `digest-R`, `rep.pt`, `digest-rep.txt`, `train_state.json` and `COMPLETE`.
  - Size: 33.673 GiB per host (36.17 GB with the small files; 578.7 GB across the 16 hosts).
- **Keep-1 rule.** `PPM_KEEP_CKPT=1`. `prune()` deletes the older save only after the new one is COMPLETE (`safety_ppm.py:297-311`, called at `train_ilp.py:1535`).
  - So after the stop, each host keeps exactly `dcp-S`. Free space is about what it is now.
  - `/scratch/amartyavarma/ppm-mot` is in `ppm/PROTECTED_SCRATCH.txt`. Never delete it.
- **Cadence.** Saves are hourly (`PPM_SAVE_EVERY_MIN=60`, last 04:26 PT dcp-9693), evals every 30 min (~45-70 s) and exports every 6 h.

## 3. Resume mechanism
- **Which save it picks.** `mot.env` has `PPM_RESUME_FROM=auto`. At `train_ilp.py:1160-1163`, `SF.newest_common` takes the MIN over nodes of each node's newest COMPLETE step and asserts that step is present on every node (`safety_ppm.py:285-295`).
  - `load()` asserts per-rank digests == saved (`safety_ppm.py:253-267`).
  - It asserts world 128 and the same rank→host map (`train_ilp.py:843`). This needs the same host file and `PPM_HOST_ALIAS=slinky-7:slinky-10,slinky-15:slinky-9`.
  - It asserts the RESUME_KEYS cfg == saved (`:856-857`).
  - `MOT_INIT` is ignored on a resume.
- **Command.** This is the same submission as 30289 (swap_exec.sh / watchdog `relaunch()`). The overrides match 30289's logged `== submission overrides:` line. `--mem=560G` equals the `mot.sbatch` header.
  ```
  PPM_ENV=/data/home/amartyavarma/ppm/mot/mot.env MOT_INIT=base PPM_BACKBONE=qwen3-14b PPM_LAYERS=40 ILP_MIN_SCRATCH_GB=64 PPM_RESUME_FROM=auto \
    sbatch --mem=560G -N 16 -w /data/home/amartyavarma/ppm/mot/launch/nodes16_s15.txt /data/home/amartyavarma/ppm/mot/mot.sbatch
  ```
- **What `mot_resume.sh` asserts first.**
  - The recipe sha (mot.env + mot.sbatch + host file + code/**) is unchanged since the stop.
  - No recipe file is newer than 30289's start (newest is 10-23 16:33 UTC).
  - No ppm-mot job is in the queue, and no pending STOP_AFTER_SAVE or SAVE_NOW.
  - All 16 hosts are idle. MoT takes all 64 CPUs, 560G and 8 GPUs per host, so even a 1-CPU side job blocks it.
  - dcp-S is intact on 16/16 hosts and is the newest COMPLETE there.
  - `/scratch` free is ≥ 64 GiB on every host. Below that, the `mot.sbatch` pre-flight ABORTs (`mot.sbatch:34-38`).
- **Log lines it then verifies, in order** (about 2.5 min to the first step, from 10-25):
  1. `== submission overrides: ILP_MIN_SCRATCH_GB=64 PPM_BACKBONE=qwen3-14b PPM_RESUME_FROM=auto MOT_INIT=base PPM_LAYERS=40`
  2. The `== host alias ... rank order slinky-0,...,slinky-8,slinky-15,slinky-7,slinky-11,...,slinky-19` line, equal to 30289's.
  3. `[resume] auto: newest step COMPLETE on every node under /scratch/amartyavarma/ppm-mot/ckpt/ppm-mot: S`
  4. `[resume] .../dcp-S: step S, weights + 3 optimizers + sched restored, per-rank digests == saved on every rank`
  5. `[plan] steps S..17366: ... combined <x>`. This digest depends on the start step, so the script checks the range. An independent proof is `tools/plan_check.py phase2 --starts S` (mot agent).
  6. The resolved cfg equals 30289's except `run_tag`. If a stderr line is interleaved into the cfg line (as in 29514), the script warns and relies on the trainer's own asserts.
  7. `EVAL step S` word_ce_norm within 0.03 of the pre-stop EVALs (now 1.9506 / 1.9505 / 1.9486).
  8. The first step line is `step S+1/17366`, about 27 s, then about 21 s. The mean word_ce_norm of the first 4 steps is within 1.5 sd + 0.02 of the last 50 pre-stop steps (1.948, sd 0.015).
- **Self-test.** `--verify-only=NEW:STOP:STEP` replays this check on the two past resumes. Both PASS:
  - 30289 from 29514 at 5829.
  - 29514 from 28155 at 3270.
  - Each has one WARN: 29514's cfg line was interleaved.

## 4. Companions: what each does when MoT stops
| who | on a planned STOP_AFTER_SAVE stop | marker / action |
|---|---|---|
| **ppm-mot-watchdog 28257** (slinky-18, `watchdog/mot_watchdog.sh`) | A deliberate stop (`STOPPED at step`, not its own) is **not relaunched** (`:193-194`). It relaunches only on an unclean death (`:195-203`), a hang (`:205-239`) or the 3-day time limit (`:190-192`, elapsed ≥ limit − 15 min, which does not apply). It reads `LAST_LAUNCH.env` only at its start. It adopts any newer ppm-mot job (`:178-179`). | `mot_stop.sh` also touches `watchdog/WATCHDOG_OFF` (no actions at all, `:185`), so a crash during the save cannot trigger a relaunch onto the monoplex hosts. `mot_resume.sh` removes it after the resume verifies. |
| **motguard** (tmux, 5-min `guard/tick.sh` cron) | Without a marker it would raise `ALERT: job ... state 'gone'` and companion alerts. | `guard/PAUSED` puts the tick in paused mode (`tick.sh:11-17`): no liveness, step or export checks; it only alerts if a ppm-mot job appears, plus /data and state backup. `mot_stop.sh` writes it before the request. `mot_resume.sh` removes it at submit, as motguard did on 10-24 and 10-25. |
| **mot-state-bk 29802** (slinky-18, `guard/state_backup/sb_driver.py`) | 2-hourly full-state HF backup (InternalCan/ppm-mot-14b-state, keeps the newest 2 VERIFIED: now dcp-9355 + dcp-9693, 04:42 PT). It starts a cycle only on a fresh `[save]` line of a RUNNING job, so it idles while MoT is stopped and resumes by itself afterwards. | A cycle in flight (hardlink hold `hfhold/dcp-N` + an upload step inside the job, ~15 min after the :26 saves at 06:26, 08:26, ...) must not be cut: the stop's prune would make the hold cost ~34 GiB per host. `mot_stop.sh` waits for `state.json` `"cycle": null`. It then sets `guard/state_backup/OFF` (no new cycle) and re-checks that none started (motguard review 4a). That stops the driver from starting a cycle on the stop save's fresh `[save]` line while the job exits. `mot_resume.sh` lifts OFF after the resume verifies. |
| **mot-hf-watch 28265** (slinky-4) | Watches `ckpt-weka/ppm-mot/w-N` (6-h exports) and uploads to InternalCan/ppm-mot-14b. It never touches the MoT hosts. A stop makes no export. | none |
| **motlive-gpu 30570** (slinky-4 GPUs 4-5, MoT live demo) | Serves from HF exports in RAM. It is independent of the training job. | none |
| shared `watchdog` 30537 (`setup-logs/watchdog/watch.sh`) | Watches the job named `ppm` only. | none |
| INTL watchdog (next-arch) | `next-arch/watchdog/WATCHDOG_OFF` has been set since 10-23, so it never relaunches ppm-next. | none |
| ppmguard | Does not act on MoT (GUARD_PROMPT, OPS_PLAYBOOK). | none |

The three pause markers are `guard/PAUSED`, `watchdog/WATCHDOG_OFF` and `guard/state_backup/OFF`, plus `go/MOT_PAUSE_IN_PROGRESS`.
- `mot_resume.sh` removes only the ones `mot_stop.sh` created (recorded in `go/MOT_STOP_STATE`).
  - It removes `guard/PAUSED` at submit.
  - It removes `WATCHDOG_OFF` and `state_backup/OFF` only after the verification passes.
- A leftover `hfhold/*` on any host is a resume FAIL (motguard review 4b).
- If the resume verification fails, it keeps `WATCHDOG_OFF`, so the watchdog cannot loop, and hands over to motguard.
- **FT-side rule** (motguard review 4c): the monoplex must never write into, or delete anything under, `/scratch/amartyavarma/ppm-mot/**`. At its end it must leave ≥ 64 GiB free on all 16 hosts.
- **After the resume** (motguard review 4d): motguard's tick may show "job PENDING / no step yet" for up to ~10 min. That is informational; motguard re-verifies the digests and step continuity itself.

## 5. Must an HF backup finish before the hosts' /scratch is reused?
- **No.** The monoplex does not delete anything of MoT's: `dcp-S` stays node-local and protected.
- The only /scratch-touching backup artifact is the state-backup hardlink hold, and the stop waits until no cycle is in flight.
- Fallback if a host's disk is lost during the pause: the HF state backup holds dcp-9693, or a newer 2-hourly dcp if one runs before the stop.
- Optional, not required: a final HF state backup of `dcp-S`, as motguard did at the 10-24 pauses (CPU holder 16 × `-c2 --mem=12G` + `sb_driver.py --force-step S`, ~16 min, `SB_HOLDER_JOB`/`SB_LOG_JOB` env). Its holder needs 2 CPUs per host, which conflicts with a monoplex that takes all 64 CPUs. Decision is motguard's or main's.

## 6. /scratch free per MoT host (fresh, 10-26 05:20 PT)
- **Method.** One tiny read-only `srun --overlap --jobid 30289 -N16 -c1 --mem=256M --cpu-bind=none`. Job 30289 is not heterogeneous.
- **Disks.** Each host has 1760 GiB; `dcp-9693` is COMPLETE (20 files) on all 16 hosts, and no hfhold.

| host | free GiB (GB) | host | free GiB (GB) |
|---|---|---|---|
| slinky-0 | 101 (108) | slinky-11 | 100 (107) |
| slinky-1 | 122 (131) | slinky-12 | 91 (98) |
| slinky-2 | 118 (127) | slinky-13 | **73 (78)** |
| slinky-3 | 109 (117) | slinky-14 | 221 (237) |
| slinky-5 | 172 (185) | slinky-15 | 630 (676) |
| slinky-6 | 91 (98) | slinky-16 | 97 (104) |
| slinky-7 | 143 (154) | slinky-17 | 113 (121) |
| slinky-8 | 103 (111) | slinky-19 | 95 (102) |

- **After the stop.** Free space stays about the same: the stop save replaces the kept one.
- **Monoplex need (~175 GB per node).** Only **slinky-5, slinky-14 and slinky-15** have that much. **The other 13 are short**, by 21-97 GB. Room has to be found before 12 hosts can be used. Never delete `ppm-mot`, `ppm-v1` dcp-16861/17798/17850, or anything on the protected list.
- **Before the resume.** Every host must be back to **≥ 64 GiB free**. That is the `mot.sbatch` pre-flight, and the first new save needs 63.74; motguard alerts below 66.
  - Upper bound on monoplex leftovers per host = free now − 64 GiB. On slinky-13 that is only 9 GiB.

## 7. Timing
- **Stop.** Next step boundary (≤ 21 s, s/step 20.9), plus 45-70 s if the 30-min EVAL falls there, plus ~15 s if the 6-h export does. Then save 15-18 s, prune ~4 s and exit ~10 s. Total about **45-120 s**.
  - 10-24 measured: request 23:57:43 → save 23:58:09 → exit 0 at 23:58:23.
  - Next EVALs are about :12 and :42; next saves about :26; next export about 11:15 PT; next state-backup cycles about 06:26-06:42 and 08:26-08:42 PT.
- **Resume.** Submit → RUNNING (hosts must be idle) → first step in about 2.5 min (10-25: 05:12:30 start, 05:13:54 digests ==, 05:14:36 EVAL, 05:15:04 first step).
- **ETA.** After the resume, MoT needs about 7,500 more steps × 21.6 s ≈ 45 h, plus the pause.

## 8. motguard review (MOTGUARD_REVIEW.md, 10-26 05:27 PT)
Verdict: OK to use on main's relayed GO. Its corrections are now applied.

1. **Markers.** The scripts set and remove `WATCHDOG_OFF` and `guard/PAUSED` as written. motguard will not touch them in parallel.
2. **Final HF state backup of dcp-S: not needed.**
   - HF keeps the newest 2-hourly dcp, so it is at most about 2 h behind S.
   - dcp-S stays node-local and protected.
   - Skip it unless the user asks.
3. **Nothing acts on the hosts during the pause.**
   - The paused tick runs no srun.
   - The watchdog is inert while `WATCHDOG_OFF` is set.
   - hf-watch is idle.
   - state-bk is idle without a RUNNING ppm-mot job.
4. **Corrections (all applied):**
   - (a) `state_backup/OFF` is set during the pause and removed after the verified resume, each only if `mot_stop.sh` created it (`SBOFF_PRE` in `MOT_STOP_STATE`).
   - (b) A leftover `hfhold` is a resume FAIL.
   - (c) The FT-side rule in section 4.
   - (d) The tick note in section 4.
