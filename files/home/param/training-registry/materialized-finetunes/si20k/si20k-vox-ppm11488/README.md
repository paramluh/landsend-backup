# si20k-vox-ppm11488: PPM speech-in / speech-out FT from dcp-11488 (10-19)

User (10-19): "the PPM speech-in, speech-out fine-tune again, but initialised from PPM dcp-11488, the backbone the ASR adapter w-1972 was
trained on, instead of 12829". All other runs untouched.

- Recipe IDENTICAL to si20k-vox-ppm (j24920): Voxtral A from raw audio in the loop with qqa-corrupt-v2 and the gauss:6 mic gain; B text +
  speech (decoder K 1 on light 0.1); B clip p 0.8 / persona p 0.2; the 12,625 / 192 strict rows (plan 01cacb274286ddb6); 198 steps; warmup 8,
  LR peaks, decay to 0; seeds; 1:1 podcast co-training on the SAME unseen rows (positions >= 25,658, unseen by 11488 too); adapter = ASR w-1972.
- The ONLY change: QQA_INIT = /data/.../ckpt-infer/ppm-v1/local-11488 (COMPLETE, step 11488, ppm-local-bf16-shards-v1 world 128, converted
  from dcp-11488; kept: the m1vox live demos reference it).
- Bookkeeping: si20k eval without the user input (adapter output x 0) at every eval + gap; vgain theta saved in fp32 in every export; ASR
  adapter load check; cfg diffed against j24920's own final cfg (only run dirs, init, store root, ledger dir, runtime calibration may differ).
- Code: code/train_si20k_vox11488.py from tools/make_m11488_trainer.py (diff tests/trainer_11488_vs_vox.diff); sbatch si20k_vox_ppm11488.sbatch;
  CPU layout 7/task + tools/repin_loop_11488.sh. Exports -> /data -> HF InternalCan/si20k-vox-ppm11488. Drain DRAIN_FOR_FT.si20kvox11488.
- Page: si20k_ft.html "PPM · dcp-11488 + ASR w-1972 (matched)".

## Log
- 10-19 03:33 PT: submitted job 25245 on the 8 PPM hosts (waits for slinky-16 / 17 to be released).
- 10-19 03:42 PT: training. ASR adapter load verified; s_init 0.00335. Step-0 si20k eval normalised word CE with A 2.221, without A 2.224 (j24920 from 12829: 2.213 / 2.216).
- 10-19 03:54 PT: step time 16-19 s -> ~13 s after a live re-pin to ONE corruption / collator pool shared across both sockets (socket 0's pool was saturated, socket 1's idle; mains keep 2 whole cores each; tools/repin_smt_11488.py, original kept as .per-socket-pool). Still corruption-CPU-bound (~24 CPU-s per rank-step); ~11 s would need --cpus-per-task 8 (restart).
- 10-19 04:33 PT: CRASH at the end: all 198 steps trained (step-198 si20k eval normalised word CE with A 1.4648, without A 1.9524), then the final node-local export failed (/scratch full on slinky-0/8/9: the watcher's /data gate (330 + 80 GB) had held export-66..165 node-local while /data sat at ~403 GB). The final weights are lost. Fixes: gate 305 + GB in every watcher; tools/backfill_11488.sbatch copies export-66..165 to /data + HF and deletes the partial export-198 dirs.
- 10-19 04:36 PT: CRASH at the final export: training finished (step 198 eval with A 1.4648, without A 1.9524) but /scratch filled while export_local wrote the final node-local copy (the watcher's /data gate 330 + 80 GB had kept export-66..165 on the nodes) -> no step-198 weights (no full training state is ever saved, so no re-export / 33-step rerun possible); export-165 = the last weights (/data -> HF). export-132 / 99 / 66 rescued from /scratch one at a time (tools/rescue_exports.sbatch). Fixed in every si20k trainer copy: final consolidated /data export first, export_local pre-checks /scratch on every node and skips instead of crashing; watcher /data gate 305 + size.
- 10-19 04:43 PT: export-165 HF-verified (InternalCan/si20k-vox-ppm11488/export-165) = the last weights; export-132 / 99 / 66 deleted from /scratch unexported (coordinator: /data has no room, result recorded).
- 10-19 05:17 PT: drain file DRAIN_FOR_FT.si20kvox11488 removed (PPM hosts released).
