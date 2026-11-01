
## RESUME (23:37 PT 10-23): the login pod restarted and killed your previous tmux session. The FT itself is unaffected (it runs in Slurm).
- Main session now: amartyavarma-c6.
- BASE was CORRECTED to pure PPM dcp-16861 (the user said "17850 is 8,1,1, not PPM"). See decision 1 above.
- **Current state:** FT job 28249 TRAINING on slinky-18 x8 + slinky-4 GPUs 0-5, ~30 s/step, ETA ~02:00 PT.
  - Step-0 eval normalised word CE 2.0088 (ref 26232 at step 0: 2.0125).
  - Watchdog aloesamsep-wd 28263 (slinky-4; moved off slinky-10, which the user is reprovisioning).
  - Final gather 28250 is chained afterok.
  - HF uploader jobs aloesamsep-lowprio 28254/28255. The uploader nice-19/idle-IO enforcer and the trainer CPU-affinity fix were applied live; check they still hold (the enforcer may have died with the pod; re-apply if so).
  - Run dir and logs: /data/home/amartyavarma/ppm/aloe-ft-32-samsep (logs/aloesamsep-28249.log, CFG_DIFF.txt).
- **Your job now:** babysit to the end. Watch for stalls or slow steps and fix them without touching the maths. Make sure the intermediate exports and the final export reach private HF and are verified.
- Then report to main in ≤3 lines: final eval normalised word CE vs reference 26232 and 26467 at the same step, the export location (local + HF), anything odd.
- Never restart the FT without telling main first. Never delete checkpoints/training state without main's go.
- pagex owns the live page.
