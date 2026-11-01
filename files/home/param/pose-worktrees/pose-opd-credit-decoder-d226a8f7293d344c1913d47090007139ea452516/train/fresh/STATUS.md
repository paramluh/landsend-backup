# 8,1,1 fresh-data run: status
**STEPPING since 23:30 PT 10-21: job 27100 (watchdog 27101), restarted from the END OF PPM dcp-16861 (user order 23:20 PT): 67 s/step, 16862 -> 17352 (end of the fresh pose, 491 steps), ETA ~08:55 PT 10-22**

| item | state | time (PT) |
|---|---|---|
| fresh pose (new-pose-llm-data: 85,076 h, 528,857 rows, payload-hash-verified releases, segment_id dedup, minus PPM TikTok v2+v3 videos (283 segments) and eval-hash videos); rows built on the fly per host from a fixed PLAN | done | 21:15 |
| speech flip (epoch >= 2 rows) + text flip (Gemma + chat, from the resume step), separate knobs; row checks pass | done | 21:05 |
| first fresh run from dcp-17012 (job 27030, 17013..17120) | stopped by user order (STOP_AFTER_SAVE at 17120) | 23:21 |
| old branch on every host: dcp-17118 deleted (superseded by 17120); dcp-17120 moved to /scratch/amartyavarma/ppm/ckpt-old811f/ppm-v1 (not exported) | done | 23:24 |
| slinky-10 dcp-16861 shards -> slinky-7 (20 files, 46.1 GB, 20/20 sha256 match) | done | 23:26 |
| launch from dcp-16861 (PPM_RESUME_FROM explicit; env prod_811_fresh16861.env: fresh pose + text flip from 16861, max_steps 17352) | done | 23:26 |
| resume digests at dcp-16861 (every rank == saved); 8,1,1 switch from PPM's GA as job 26391; link off as before | done | 23:28 |
| start eval vs PPM's eval at 16861: 184/184 metrics exact | done | 23:29 |
| first steps: grad isolation / AdamW counters / 64+72 decoder updates at 100% frames all as expected; 67.3 s/step | done | 23:32 |
| evals auto-appended to fresh/EVALS_16861.md | running | 23:33 |
| Mimi: fresh audio = lufs23-local-v1 = the CORRECT encode (user); TikTok-encode pose eval audio cb0 will rise (old encode) | info | |
| chat eval (unflipped) rose 0.59 -> 0.82 in 27 steps on the 17012 branch (text flip effect); flagged to main | flagged | 21:47 |
| live page stage3_ppm811.html: chain fixed to PPM -> 27100 only (abandoned 26391 / 27030 legs excluded; ppm_doc.py predecessor must start before the resume step; .bak-pre-branchrestart-1021) | done, page at 16865 | 23:35 |
| save cadence shifted off the old branch's HF step names (HF done markers are keyed by step: dcp-16914/16967/17012 hold the OLD branch): SAVE_NOW at 16865 (done) + at ~16890 (job 27103) -> hourly saves ~16942, ~16994, ~17046 | in progress | 23:36 |
| HF backups of the new branch: the hfb watcher holds a save only if every host keeps >= 80 GB free; slinky-7 sits at ~20-63 GB (old-branch dcp-17120 kept, unexported, + stale qwen-s3-ckpt 183 GB / ppm-qqa-scale 165 GB / qwen-s4-ckpt 103 GB there) -> no HF holds until space is freed | needs main's call | 23:36 |
| 10-22 APPEND prep (user: more distinct pose data coming, same format): trainer PPM_FRESH_POSE_ROOT2 / _TAIL (backups .bak-pre-append-1022; unset = 27114 byte-identical), ingest pipeline fresh/append/ingest_more.sbatch <SRC>, env prod_811_fresh_append.env (PPM_MAX_STEPS filled by the ingest); all in fresh/append/APPEND_READY.md | ready, not launched | 10:35 |
