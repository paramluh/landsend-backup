# npl5 (delivery 5) staged ingest: summary (10-27 16:42 local / see TIMES.tsv for PT)

- source hf://buckets/InternalCan/pose-llm-data/new-pose-llm-data-delivery5: 552,329 parquet + 552,329 release.json, 1223.7 GB, 552,329 releases of 552,308 jobs; 0 releases already used (plan 1 + npl2 + npl3 = 622,178)
- placement (src_place.py, BALANCE=tail, PRIOR_TAGS=npl2,npl3, unconstrained free): 1223.66 GB, est 1,338,027 rows (~1306 steps); per plan key GB 0:76.14, 1:76.18, 2:76.25, 3:76.28, 5:76.36, 6:76.34, 7:76.43, 8:76.5, 9:76.49, 11:76.54, 12:76.6, 13:76.62, 14:76.62, 16:76.73, 17:76.73, 19:76.85
- scans done 16/16: 56,945,936 segments, 224,662 h
- orig_h/orig_w sidecars (user change #2; scan/origwh/<slot>.parquet: file,row,segment_id,clip_id,orig_h,orig_w) 16/16: 56,945,936 segments, null orig_h 0
- global dedup (plan 1 + npl2 + npl3 scans, PPM TikTok, eval videos, SAM): scanned 56,945,936, kept 56,674,353; drop_eval_video 56,935, drop_sam_filter 10,868, drop_prior_segment 21,443, drop_dup_segment 182,325, drop_prior_video_overlap 3, drop_ppm_video 9
- plans: 1,391,232 rows, 223,599 h over 16 slots; src_root /scratch/amartyavarma/npl-append/npl5/jobs; per key rows 0:86820, 1:86839, 2:86902, 3:86922, 5:86909, 6:86904, 7:86935, 8:86984, 9:86943, 11:87031, 12:86989, 13:86972, 14:86931, 16:87009, 17:86973, 19:87169
- dry run: ends [17352, 17641, 18022, 19382] -> PPM_MAX_STEPS=19382 (+1532 steps after dcp-17850); rows/plan [528857, 271367, 395285, 1391232], h/plan [85076, 43661, 63606, 223599]; at 17850 every rank is in plan [3], npl5 starts at steps [18022, 18034]; unused rows 5,237; flags ['speech enters epoch 2 (flipped repeats) from 17872']
- GUARD-s2p.json: ok, append_at [17352, 17641, 17850], refusals {'npl3 dropped': 'refused', 'reordered': 'refused', 'tail 0': 'refused', 'other key': 'refused'}, trainer /data/home/amartyavarma/ppm/pose5/code-811-s2p/train_ppm.py
- GUARD.json: ok, append_at [17352, 17641, 17850], refusals {'npl3 dropped': 'refused', 'reordered': 'refused', 'tail 0': 'refused', 'other key': 'refused'}, trainer /data/home/amartyavarma/ppm/code-811/train_ppm.py
- fresh_pose check (raw parquet == built rows / micros): 16/16 slots OK
- launch-day parquet share per physical host GB: 0:74.83, 1:74.87, 2:74.93, 3:74.97, 5:75.04, 6:75.03, 7:75.11, 8:75.18, 11:75.22, 12:75.28, 13:75.3, 14:75.3, 15:75.18, 16:75.41, 17:75.41, 19:75.52

## TIMES.tsv
```
fetch	slinky-0	546s	A@slinky-10	10-27 05:51:04 PT
fetch	slinky-1	560s	B@slinky-18	10-27 05:51:18 PT
verify	slinky-0	35s	A@slinky-10	10-27 05:51:39 PT
verify	slinky-1	107s	B@slinky-18	10-27 05:53:05 PT
fetch	slinky-2	606s	A@slinky-10	10-27 06:01:45 PT
fetch	slinky-1	4s	B@slinky-18	10-27 06:02:23 PT
verify	slinky-1	88s	B@slinky-18	10-27 06:03:51 PT
verify	slinky-2	154s	A@slinky-10	10-27 06:04:19 PT
origwh	slinky-1	273s	B@slinky-18	10-27 06:08:24 PT
origwh	slinky-0	448s	watch@slinky-10	10-27 06:09:00 PT
fetch	slinky-3	57s	B@slinky-18	10-27 06:09:21 PT
verify	slinky-3	281s	B@slinky-18	10-27 06:14:02 PT
origwh	slinky-2	442s	watch@slinky-10	10-27 06:16:22 PT
scan	slinky-0	1536s	A@slinky-10	10-27 06:17:15 PT
fetch	slinky-5	612s	A@slinky-10	10-27 06:27:32 PT
verify	slinky-5	123s	A@slinky-10	10-27 06:29:35 PT
origwh	slinky-5	64s	watch@slinky-10	10-27 06:30:47 PT
scan	slinky-2	1345s	A@slinky-10	10-27 06:39:44 PT
origwh	slinky-3	1624s	B@slinky-18	10-27 06:41:06 PT
fetch	slinky-6	603s	A@slinky-10	10-27 06:49:51 PT
verify	slinky-6	96s	A@slinky-10	10-27 06:51:27 PT
scan	slinky-1	2632s	B@slinky-18	10-27 06:52:16 PT
origwh	slinky-6	62s	watch@slinky-10	10-27 06:52:30 PT
scan	slinky-5	1311s	A@slinky-10	10-27 07:01:39 PT
fetch	slinky-7	678s	B@slinky-18	10-27 07:05:45 PT
verify	slinky-7	352s	B@slinky-18	10-27 07:11:38 PT
fetch	slinky-8	610s	A@slinky-10	10-27 07:11:54 PT
verify	slinky-8	111s	A@slinky-10	10-27 07:13:45 PT
origwh	slinky-8	64s	watch@slinky-10	10-27 07:14:56 PT
origwh	slinky-7	239s	B@slinky-18	10-27 07:15:37 PT
scan	slinky-6	1320s	A@slinky-10	10-27 07:23:44 PT
fetch	slinky-9	600s	A@slinky-10	10-27 07:33:49 PT
verify	slinky-9	111s	A@slinky-10	10-27 07:35:40 PT
origwh	slinky-9	63s	watch@slinky-10	10-27 07:37:01 PT
scan	slinky-3	2598s	B@slinky-18	10-27 07:37:45 PT
fetch	slinky-11	153s	F@slinky-18	10-27 07:44:24 PT
verify	slinky-11	74s	F@slinky-18	10-27 07:45:38 PT
scan	slinky-8	1331s	A@slinky-10	10-27 07:46:00 PT
origwh	slinky-11	48s	F@slinky-18	10-27 07:46:26 PT
fetch	slinky-12	568s	G@slinky-10	10-27 07:53:05 PT
verify	slinky-12	78s	G@slinky-10	10-27 07:54:23 PT
origwh	slinky-12	35s	G@slinky-10	10-27 07:54:58 PT
fetch	slinky-13	617s	A@slinky-10	10-27 07:56:22 PT
verify	slinky-13	106s	A@slinky-10	10-27 07:58:08 PT
origwh	slinky-13	66s	watch@slinky-10	10-27 07:59:29 PT
scan	slinky-11	932s	F@slinky-18	10-27 08:01:58 PT
fetch	slinky-16	659s	G@slinky-10	10-27 08:05:57 PT
scan	slinky-9	1349s	A@slinky-10	10-27 08:08:34 PT
verify	slinky-16	176s	G@slinky-10	10-27 08:08:53 PT
origwh	slinky-16	62s	G@slinky-10	10-27 08:09:55 PT
fetch	slinky-14	562s	F@slinky-18	10-27 08:13:58 PT
verify	slinky-14	86s	F@slinky-18	10-27 08:15:24 PT
origwh	slinky-14	49s	F@slinky-18	10-27 08:16:13 PT
fetch	slinky-17	606s	A@slinky-10	10-27 08:18:45 PT
scan	slinky-7	2386s	B@slinky-18	10-27 08:19:40 PT
verify	slinky-17	107s	A@slinky-10	10-27 08:20:32 PT
origwh	slinky-17	63s	watch@slinky-10	10-27 08:21:54 PT
scan	slinky-13	1339s	A@slinky-10	10-27 08:30:58 PT
scan	slinky-14	895s	F@slinky-18	10-27 08:31:08 PT
fetch	slinky-19	631s	F@slinky-18	10-27 08:32:48 PT
verify	slinky-19	83s	F@slinky-18	10-27 08:34:11 PT
origwh	slinky-19	48s	F@slinky-18	10-27 08:34:59 PT
scan	slinky-12	2736s	G@slinky-10	10-27 08:40:34 PT
scan	slinky-19	790s	F@slinky-18	10-27 08:48:09 PT
scan	slinky-17	1170s	A@slinky-10	10-27 08:50:33 PT
scan	slinky-16	2270s	G@slinky-10	10-27 09:18:27 PT
plan	-	583s	slinky-10	10-27 09:28:14 PT
dry_ends	-	0s	slinky-10	10-27 09:28:14 PT
dry_need	-	14s	slinky-10	10-27 09:28:28 PT
dry_fetch	-	22s	slinky-10	10-27 09:28:50 PT
dry_verify	-	0s	slinky-10	10-27 09:28:50 PT
dry_check	-	141s	slinky-10	10-27 09:31:11 PT
dry_micros	-	72s	slinky-10	10-27 09:32:23 PT
dry_guard_811	-	4s	slinky-10	10-27 09:32:27 PT
dry_guard_s2p	-	4s	slinky-10	10-27 09:32:31 PT
dry_unittest_map3	-	1s	slinky-10	10-27 09:32:32 PT
windows	-	20s	slinky-10	10-27 09:32:53 PT
rehearsal	slot9@slinky-4	hook 35s rc 0; prep 5s rc 1	-	10-27 09:33:33 PT
rehearsal	slot9@slinky-4	hook 33s rc 0; prep 34s rc 0	-	10-27 09:35:15 PT
```

## Tests (10-27, all on spares; nothing touched the 16 hosts)
- Smoke test of the off-host plan build + dry run on 16 tiny slot scans: PASS (test/).
- Provisional slinky-0 npl5 plan (06:35, from its own scan): digest ef8b3d8f43bd0b3ff57d2cdd66fadbbc == the final plan's (determinism).
- Plans 1/npl2/npl3 rebuilt off-host for all 16 keys (rebuild13/): 48/48 digests == dcp-17850 pose_digests.
- Rolling-fetcher sim v2 (sim/run-31010-slinky-0, final npl5_roll.py): keep=1 + chained hold + pin; relaunch from the held save 18110 with 25 s of
  init and no step line; LOWMEM window 18060-18080; 520 steps / 336 npl5 steps; 0 missing; a second daemon refused by the host lock.
- Overlap-step safety test v2 (safety/run-1793114529, final code, dummy homogeneous job 31034): the main step survived all of: an overlap-step
  OOM (confined), a daemon SIGKILL (STEP_DIED + re-attach in 15 s), a STOP (supervisor scancelled only its own step), and an srun client SIGKILL.
- Launch-day rehearsal (launch/rehearsal-31047, slinky-4 acting as slinky-15 = slot 9): leg1_slot9_hook.sh OK 33 s, npl5_prep.sh OK 34 s.
- 16-slot end-to-end (e2e/, job 31048): every slot's real 4 plans at the canonical dirs; digests == the dry run's (dcp-17850 prefix + npl5);
  npl5 sha == PLANS.json; 24 trainer micros per slot (8 ranks x 17850 / first npl5 row / last step): 16/16 OK.
- Note (audit R2 NIT): rehearsal 31047 ran with NPL5_MIN_FREE_GB=8 (test only) because the spare slinky-4 had 26 GB free /scratch; the
  production default floor is 66 GB node-local / 350 GB when the dest is on /data. The first try (31041) correctly stopped with NOSPACE.
- Round 2 (10-27 ~10:15 PT): npl5_roll.py v2.1:
  - the heartbeat recomputes live / covered / ahead / alerts / markers every 30 s, also mid-pass;
  - 429 count; SLOWPASS marker; headroom uses file - shmem;
  - the floor is lowered per (job log, resume) event;
  - in-memory presence; every delete goes through the realpath check;
  - slinky-13 Weka-served window (S13_WEKA_FOR_C.txt), spare-tested in wekatest/.
- Weka-served window (s13; s6 approved too), spare tests on slinky-4 with the REAL plans:
  - wekatest/run-31074: sim rc 0, 0 missing, 294 npl5 steps; relaunch from the held save; 3,417 drops, all inside the Weka dir; 0 files touched outside.
  - run-31077, final v2.1 code: rc 0, 0 missing. Relaunch from the held save 18058. live never goes back except at the relaunch. The status carries scratch_free_GB + dest_free_GB.
  - check6-31078: cross-host refuse rc 4 (s6 pointed at s13's dir); s6 window through its own symlink OK; fresh_pose check through the symlink OK.
  - All test dirs removed; npl5-weka/ is absent until C's stage block creates npl5-weka/<host> at launch.
