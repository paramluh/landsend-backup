## STOPPED 10-23 23:41 PT (main's order: user wants a Voxtral ASR adapter matched to 16861 first; FT to be rerun with that adapter, same recipe)
- Job 28249 stopped via STOP_AFTER_SAVE at step 90/249 (training exit 0). Resume state NOT saved (no /scratch room); the rerun is fresh anyway.
- Last eval step 83: eval_speech/word_ce_norm 1.8257 (26232 @84: 1.8342; 26467 @84: 1.8270).
- export-stop90 (bf16 weights): node-local + /data copies DELETED 23:57 PT (see below);
  private HF InternalCan/ppm-aloe-samsep-16861-final/export-stop90/ VERIFY PASS both hosts.
- Intermediates export-41, export-83 on private HF InternalCan/ppm-aloe-samsep-16861 (VERIFY PASS; node-local copies deleted per approved rule).
- Gather 28250 and watchdog 28263 cancelled; uploads finished in CPU-only jobs 28268/28269. All stores, run.env, CFG_DIFF.txt, code kept intact.
- 10-23 23:57 PT (main amartyavarma-48 go): deleted node-local export-stop90 (slinky-4 31G, slinky-18 43G) and /data export-249-stop90 (73G); only copy of the step-90 weights = private HF InternalCan/ppm-aloe-samsep-16861-final/export-stop90/ (VERIFY PASS). Logged in setup-logs/hfb/DELETED-1021-hfb-hold.log.
