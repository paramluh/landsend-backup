# si20k-vox-ppm-nocorrupt: the si20k-vox PPM FT WITHOUT audio corruption (10-19)

User-approved diagnostic: why does speech-in trail text-in on si20k (on Aloe, with no added corruption, they matched)? Recipe IDENTICAL to
si20k-vox-ppm j24920 (init local-12829 = dcp-12829, ASR adapter w-1972, Voxtral A from raw audio in the loop, gauss:6 mic gain, B text + speech,
decoder K 1 on light 0.1, B clip 0.8 / persona 0.2, 12,625 / 192 strict rows, 198 steps, warmup 8, decay to 0, seeds, the same podcast rows)
EXCEPT SI20K_A_CORRUPT=none: qqa-corrupt-v2 OFF on the si20k user audio and on the podcast A audio (both fed clean).
Bookkeeping as si20k-vox-ppm11488: no-A eval at every eval, fp32 theta export, adapter load check, cfg diff vs j24920's final cfg (allowed: run
dirs, data.a_corrupt / podcast.data.a_corrupt, store root, ledger dir, runtime calibration). Page label "PPM · dcp-12829 + ASR w-1972 · no corruption".
Launch: tools/chain_noc.sbatch on the 8 PPM hosts after si20k-vox-ppm11488's final export is HF-verified (idle fallbacks); drain DRAIN_FOR_FT.si20kvoxnocorrupt.

## Log
- 10-19 04:23 PT: prepared; chain job 25282 waiting for si20k-vox-ppm11488's final + HF verify.
- 10-19 04:32:08 PT: launched job 25283 (chain: FALLBACK: 11488 ended without a final export).
- 10-19 04:33:51 PT: launched job 25293 (after the 11488 scratch backfill; the first chained launch 25283 aborted at preflight: /scratch full).
- 10-19 04:38:40 PT: launched job 25312 (min /scratch free 40 GB after the 11488 rescue; earlier attempts 25283 / 25293 aborted at preflight: /scratch full).
- 10-19 04:43 PT: training (launched 04:38 when every PPM host had >= 40 GB /scratch). Step 0 si20k eval normalised word CE with A 2.2134, without A 2.2161 (j24920: 2.213 / 2.216); ~11.5 s/step, data wait < 1 s (no corruption).
- 10-19 05:17 PT: STOPPED by the user at step 153 (STOP_AFTER_SAVE): weights-only export export-198-stop153 (78.2 GB, COMPLETE) on /data; drain file removed, the 8 PPM hosts released.
