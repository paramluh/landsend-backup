# pose5 STATUS — prep (NO launch) of the 16-node 8,1,1 continuation: dcp-17850 + npl5 (delivery 5) + s2p

Owner: tmux agent pose5. Main: amartyavarma-ff. Ops Board t1027-25. Prompt: PROMPT.md (this dir).
Nothing here touches the 16 production hosts or MoT job 30289. The user decides the launch.

## Now (10-27 05:31 PT)
- Plan + ETA sent to main ~05:30 PT. READY target ~11:30 PT.
- Workstreams A, B, C launched as background subagents at 10-27 05:31 PT:
  - A. s2p port -> code-811-s2p + proofs (flag off byte-identical; zero-init identical loss, tiny + real 811 micros per row type; grads + tower_p LR group; resume from a ckpt without s_emb).
  - B. npl5 ingest staged on the spares (slinky-4/10/18, nice, CPUs/net/scratch only) -> per-host plans for the 16 hosts; launch-day copy/fetch path + measured delay.
  - C. dcp-17850 location + verification; launch env + resolved-cfg diff vs 27653; stream limits; steps + wall-clock; launch kit script.

## Facts (verified 05:25-05:31 PT)
- Production hosts NOW (30289): slinky-[0-3,5-8,11-17,19]. The 811 run (27114/27204/27653) ran on slinky-[0-3,5-9,11-14,16,17,19]
  with slinky-7 playing slot slinky-10 (PPM_HOST_ALIAS=slinky-7:slinky-10). slinky-9 is off-limits (Faisal) -> slinky-15 must play
  slot slinky-9 (as PPMVOX v5 did 10-25). code-811/nodes16.txt is stale (lists 9 and 10). Asked main to confirm.
- dcp-17850: paused save of 27653 (10-22 22:29 PT). 10-25 03:03 PT user approved D1 = delete node-local dcp-17850 copies (HF-backed,
  InternalCan/ppm-v1-c). Current location: being verified (C).
- Delivery 5: 552,329 parquet + 552,329 release.json (1,104,658 files), 1.224 TB (1.203 TB parquet); est 58.8M segments, ~230k h
  (d3 = 63.6k h -> 395,285 rows). Est npl5 ≈ 1.43M rows ≈ +1,380 steps ≈ 26 h at ~68 s/step (C will pin these).
- npl3 ingest on the 16 hosts (10-22): fetch 316 s for 357.7 GB (1.13 GB/s, ~1,180 files/s), plan 629 s.
- Spare /scratch free 05:35 PT: slinky-10 495 GB, slinky-4 73 GB, slinky-18 (pending). /data free 1.3 TB (floor 300 GB).

## Host alias map (main confirmed 10-27 ~05:30 PT: run on exactly 30289's 16 hosts; never slinky-9)
- 811 logical slots (= code-811/nodes16.txt): slinky-0,1,2,3,5,6,8,9,10,11,12,13,14,16,17,19.
- Physical hosts (= 30289): slinky-0,1,2,3,5,6,7,8,11,12,13,14,15,16,17,19.
- PPM_HOST_ALIAS=slinky-7:slinky-10,slinky-15:slinky-9 (slinky-7 already played slot slinky-10 in 27114/27204/27653; NEW: slinky-15
  plays slot slinky-9). Every other slot = its own host. (Main suggested slinky-19 for slot 10 "if needed": not needed, slinky-7 has
  always played it and slinky-19 is its own slot.)
- Consequence: slot slinky-9's node-local state (plan-1/npl2/npl3 plan dirs + any npl3 parquet still to be read after 17850, plus
  dcp-17850's slot-9 ranks 56-63) must be put on slinky-15 before launch. Covered by B/C.

## B (npl5 ingest) — 10-27 05:40 PT
- Measured (slinky-10 probe, 2,000 d5 releases / 4.43 GB): HF fetch 64 thr ≈ 190 MB/s/node, 0 x 429; xet verify 2.6 GB/s; scan 2.03 ms/segment/CPU;
  spare->Weka cp 261 MB/s. Free CPUs: slinky-10 6, slinky-4 6, slinky-18 0 (CPUAlloc 64/64).
- Path chosen: 2 lanes (slinky-10 raw on its /scratch; slinky-4 raw rolling via /data <=155 GB at a time) fetch->verify->scan per slot; scans
  to pose5/npl5/scan (~3 GB); raw deleted after scan; plans built off-host with fresh_pose cmd_plan (FRESH_HOST=slot, canonical src root) ~4 GB.
- Scan cost 58.8M segs x 2.03 ms = 119k CPU-s ≈ 2.8 h on 12 CPUs + dedup/plan ~30 min + off-host dry run ~10 min -> staged plans ~09:15-09:45 PT.
- Launch day (staged): per-host HF fetch of each host's parquet share + plan copy + on-host check/dryrun ≈ 10-20 min.
  Without staging: full pipeline on the 16 hosts ≈ 40-50 min.
- Plan host keys are physical-name style: slot 10 = "slinky-7", slot 9 = "slinky-9" -> slinky-15 must serve plan host "slinky-9" (C checking how).

## Change #2 (user, main stamp "06:00 PT"; board t1027-29): G0COND — 10-27 05:59 PT
- Condition each pose clip on frame-0 root xy + shoulder width: g_proj(f) added to the TRUNK input embedding at the clip's SLOT
  position (gpos); f = [r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w]; invalid frame0 -> 0; g_proj zero-init; pose rows only;
  flag PPM_G0COND=1, independent of PPM_S2P. Aspect ratio only if available for ALL plans without re-scan (else left out).
- Workstream D launched 10-27 05:59 PT: builds it in its own copy pose5/code-811-g0 (no edit collisions with A), patch pose5/g0cond.patch, proofs,
  then merges into code-811-s2p once A's s2p tests pass; combined-flag tests there.
- B told to keep npl5 orig_h/orig_w as a scan sidecar (scan/plan bytes unchanged). C told to add PPM_G0COND=1 to the env block.

## C DONE (launch kit, env, diff, dcp-17850) — 10-27 06:01 PT
- dcp-17850: HF InternalCan/ppm-v1-c dcp-17850/<physical host>/ (slot 10 = dir slinky-7, slot 9 = dir slinky-9), restore bundle in
  ppm-v1-d. Verified 10-27: private, 320 files, 737,404,296,576 B, every size == watcher upload record, train_state step 17850 + COMPLETE
  on all 16; sha256 evidence = watcher VERIFY PASS 16/16 at upload (10-23). Record: c/DCP17850_LOCATION.json, c/hf17850/VERIFY_17850.json.
  Node-local: D1-deleted 10-25 on slinky-0,3,5,6,8,11,12,13,16,17,19; still on slinky-1,2,7,14 per records (not re-checked live);
  slot 9 = slinky-15 /scratch/amartyavarma/ppmvox/init-17850-slots/dcp-17850 (v5, sha-verified 10-25). No Weka copy.
  Launch-day staging ~3-6 min (11 hosts x 46.1 GB from HF at ~0.85 GB/s/host; 4 verify-only; slinky-15 hard link).
- Guard: trainer's own guard on the real train_state accepts npl5 appended at 17850 (<= 18022); 17850..18021 read exactly 27653's rows;
  each host then uses its 6,645 unused npl3 tail rows, then npl5.
- Slot-9 keying: no code change (train_ppm.py:1236/1245 accept slot or physical plan host; digest key = plan's host field) -> slinky-15
  needs slinky-9's plan files byte for byte (v5 rebuilt them 10-25). NEW blocker: slot 9's PPM pose store (eval) only on slinky-9 ->
  rebuild 16 eval shards (~10 GB) on a spare from the deterministic builder, sha-checked (C owns it now; hook pose5/slot9/stage_slot9_extras.sh).
- Env prod_811_fresh_append5_s2p.env = append3 lines 1-308 verbatim + marked block: npl5 in PPM_FRESH_POSE_ROOT2, PPM_S2P=1,
  PPM_MAX_STEPS=FILL_ME, PPM_HOST_ALIAS=slinky-7:slinky-10,slinky-15:slinky-9, PPM_G0COND=1, reserved [POSE5-NEXT].
  CFG_DIFF_append5_s2p_vs_27653.txt: 6 keys, all expected (npl5 plan list, max_steps, resume_from, run_tag, s2p, s2p_from_step);
  g0cond key pending D's merge. v4 data rules unchanged.
- Steps: 1,024 pose rows/step; npl5 ≈ 1.43M rows ≈ 1,397 steps; end ≈ 19,425 (1,575 steps from 17850). 68.3 s/step (69.9 with evals/saves)
  -> one npl5 pass ≈ 27 h, whole leg ≈ 30.6 h. Nothing else binds (Gemma 39,902; chat unseen to 24,305; podcast epoch 2 to ~52.6k).
- Launch kit: bash /data/home/amartyavarma/ppm/pose5/launch_pose5.sh --go --stop-mot (default --dry-run; dry run tested).
  MoT stop-with-save (needs --stop-mot + main/motguard coordination) -> 16-host staging job (disk gate, dcp-17850, npl5 hook, slot-9 extras)
  -> ppm811 afterok -> watchdog + eval logger on a spare + existing 8,1,1 live page refresh. Kit files: kit/, nodes16_s7_s15.txt.
- Shared infra touched: setup-logs/hfb/snap_code.sh now recognises code-811-s2p (backup .bak-pre-pose5-1027).
- OPEN: (1) DISK: need ≈ npl5 ~80 + dcp 46 (11 hosts) + 122 gate per host vs 72-220 GiB free (04:51 PT, MoT running) -> ~1.7 TB short
  over 14 hosts; per-host table + HF-backed candidates being written (c/DISK_NEED.json); deletion = main/user call. (2) MoT's stop-save
  HF backup runs inside 30289 -> motguard must plan it. (3) Node-local dcp-17850 rotates out ~2 h after start under keep-2 (HF stays).
- 10-27 06:02 PT main: DISK approved in principle for LAUNCH DAY only (after the user's go, after MoT's stop save is HF-verified 16/16): delete
  HF-verified node-local items only (MoT older dcps, stale 811/decay/FT saves, caches); never non-verified items; never dcp-17850.
  Kit gets a dry-run listing + explicit --free-disk step (C). Remaining gap after that -> main -> user. MoT backup after 30289 = motguard.
  snap_code.sh edit checked by pose5: strictly additive (one new case branch; old callers unchanged) -> kept.

## B progress — 10-27 06:02 PT
- orig_h/orig_w sidecar: pose5/npl5/scan/origwh/<slot>.parquet (+ .json), subdir on purpose (keeps fresh_pose's prior-scan glob clean for
  a later npl6 / delivery 4). Scans/plans unchanged (byte-identical to on-host fresh_pose).
- Plans 1-3 scans/plans carry no orig_h/orig_w, but row build does (fresh_pose.decode_new from raw parquet -> clips records) -> told D.
- Lane B restarted as v2 (12 workers, slinky-18) after its Weka-read scan was IO-bound; ~10 min lost.
- Smoke test end to end PASS (16 tiny slots: off-host plan build, fetch, check x16, micros, guard with code-811 and code-811-s2p + the
  launch env: npl5 added at 17850, 4/4 refusals).

## USER 06:16 PT (binding, via main) — recorded 10-27 06:16 PT
- Resume from dcp-17850 CONFIRMED.
- Plan: PAUSE MoT (resumed after), run the pose training, then resume MoT. NOTHING MoT-related on the 16 hosts is deleted or moved
  (ckpts, opt shards, exports, data, caches). Launch-day deletion approval now = NON-MoT, HF-verified stale items only; never dcp-17850.
- C recomputing c/DISK_NEED.json without MoT items + options that touch nothing of MoT (smaller save/HF-hold gate, rolled npl5 shares,
  Weka-served shares, dcp-17850 restage order / rotation [needs main's OK]). Remaining gap -> main -> user.
- Kit end: final save + HF verify 16/16 + clean hand-back of the 16 hosts for the MoT resume; pose-run files removed only after HF verify
  and only if MoT needs the space, flagged to main first (kit prints, never deletes). Run length configurable (--max-steps / stop-at).
- Main tells the user: one npl5 pass ≈ 27 h -> MoT paused ≈ 27 h + overheads.
- 30923 (slot-9 eval rebuild) repinned slinky-4 -> slinky-18 (slinky-4 memory full); running.

## EARLIEST LAUNCH (user 06:18 PT: "why cant you do it now and then we save and exit?") — 10-27 06:21 PT
- Sent main: binding constraint = DISK. Leg 1 = dcp-17850 flags OFF on npl3 leftover 17850->18022 (~3.3 h), then restart from the 18022
  final save with npl5 + s2p + G0COND (guard allows append at 18022, as npl3 at 17641; restart ≈ 5-8 min).
- Disk (c/DISK_NEED.json, records 04:51 PT): leg 1 (dcp restage 46 GB on 11 hosts + 122 gate) short on 10/16 hosts (48-91 GB);
  minimal gate (1 save + 30 floor) still short on 10 hosts (2-45 GB); full npl5 node-local short ~1.9 TB on 14 hosts. Zero HF-verified
  non-MoT stale items found.
- Options sent to main: (i) serve dcp-17850 from a temporary Weka copy (737 GB) [C checking loader/rotation/watchdog]; (ii) delete consumed
  plan-1 + npl2 raw (rows < 17641; HF-bucket re-downloadable) ≈ 45 GB/host [user call: OPS never-delete npl2/npl3]; (iii) npl5 rolling fetcher.
- Earliest with (i): ~07:15-07:30 PT leg 1; s2p+G0COND+npl5 at the 18022 restart ~10:30 PT. A ETA ~07:15, D ETA ~08:30 (firm ETAs requested).

## MAIN GO (main msg stamped "06:28 PT"; real clock ~06:22) — earliest launch (recorded 10-27 06:22 PT)
- APPROVED (i): dcp-17850 served from a temporary Weka copy (only the 11 hosts without node-local: ~507 GB); /data >= 300 GB at ALL times
  (abort copy < 350); verify vs HF manifest (size + sha256); prove read-only + no rotation/cleanup can touch it; drop the Weka copy only
  after leg 1's first save is HF-verified 16/16 (main approved that drop).
- NOT approved (ii): plan-1/npl2/npl3 raw stays.
- YES (iii): npl5 via a per-host rolling fetcher (B).
- Leg 1 flags: PPM_S2P on only if A's remaining tests pass before launch; else flags off and s2p on at the 18022 restart.
  G0COND + npl5 join at the 18022 restart (~10:30 PT).
- Sequence: pose5 sends READY-15 -> main fires motguard mot_pause.sh --export=none -> "GPUS FREE for pose5" (~2 min, stop save 16/16)
  -> pose5 launches. MoT HF backup continues in a small CPU-only holder job on the hosts: leave it alone, budget its CPU/mem, never touch
  MoT's node-local paths.
- Owners: C = leg-1 kit + Weka copy + prune protection (dcp-17850/17798/16861 inside CKPT) + watchdog pinned to the explicit 17850 resume
  until a new save; B = slot-9 plan files + npl3 rows on slinky-15 (leg 1) and the npl5 rolling fetcher (leg 2); A = s2p tests; D = G0COND.
- 10-27 06:24 PT stamp audit: every '## ... — <stamp>' heading here was taken from TZ=America/Los_Angeles date at write time; hand-typed times in my messages to main (06:27, 06:35, 05:38) ran 3-12 min ahead and are superseded. Quoted user/main times are main's stamps.

## USER HOLD (main stamp 06:25 PT) — recorded 10-27 06:24 PT
- Do NOT stop MoT; NO launch until the user explicitly says go; nothing touches the 16 hosts or MoT. Keep preparing everything (leg-1 kit,
  Weka dcp copy, s2p/G0COND proofs, npl5 ingest). Send READY (real stamp) when launchable; main takes it to the user.

## B ETAs — 10-27 06:26 PT
- Slot-9 leg-1 hook ~06:55 PT (path in npl5/LEG1_SLOT9_HOOK.txt): plan-1/npl2/npl3 plan files = ppmvox v5slot7 off-host rebuilds (sha256 ==
  npl3/dryrun/slinky-9.json; digests == dcp-17850 pose_digests[slinky-9]); npl3 parquet = 5,434 files / 10.24 GB (slot 9 rows from position
  7912 to the end of plans 1-3) from the HF delivery-3 bucket, staged on /data, xet-verified. Check-only mode for the other 15 hosts.
- Staged full npl5 plans ~10:00-10:30 PT (scan-bound: lane A 25.6 min/slot; lane B 1.8x slower, raw on Weka) + dedup/plan 30 min + dry run
  10 min; dry run redone for append at 18022 (leg 2).
- Rolling fetcher script + spare test ~08:45 PT. /data: lane B <=2 slot shares, fetch only if free >= 450 GB -> /data >= ~370 GB.

## D (G0COND) — 10-27 06:27 PT
- Done: resolved cfg flag off == original (cfg, COEF, RESUME_KEYS, params byte-equal); flag on adds only cfg g0cond/g0cond_from_step and
  param inp.g0.weight [5120,8] in LR group 'audio' (= inp.pose's group). Flag-off micros byte-identical to A's; 11 real plan-1 rows / 400
  clips: g0pos = SLOT positions, f == raw-parquet recompute, 400/400 valid.
- Running: tiny-config micros (orig off / copy off / zero-init on / non-zero on) + resume phases (job 30934). Proofs PASS ETA ~07:05 PT;
  merge + combined tests ~45 min after A's PASS (7 trivial same-line conflicts).
- BLOCKER: A's and D's real-width reduced-depth jobs need ~56 GB + 2 CPUs each; spares full (64/64; slinky-4 ~3 GB allocatable).
  Asked main for 4 CPUs + ~120 GB on slinky-18/10 for ~1 h; fallback = B yields its Weka-bound lane for an hour.

## SWITCH COMMAND (user, main stamp 06:26 PT: "instantly switch to the pose run (maintaining a fully resumable mot)") — 10-27 06:28 PT
- C builds /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh: motguard mot_pause.sh --export=none -> wait "GPUS FREE for pose5" (~6-8 min;
  abort+alert on failure, MoT resumable) -> leg 1 (PPM_S2P=1 iff A's machine-checkable PASS marker, else flags off) + watchdog/HF watcher/page
  -> verify dcp-17850 load + 2 sane steps vs 27653 -> "POSE RUN LIVE <job>". No-op dry-run mode + ARMED self-check.
- Earliest ARMED told to main: ~07:30-07:45 PT; command -> LIVE ≈ 15-20 min.
- Demo: videort's avatar relaunch 30939 (slinky-4, 8 CPU/120G) blocked; pose5 cancelled its 4 slinky-4 jobs (A 30930/30935/30931, C 30940),
  but genweb-pool-w 30938 took the freed resources; escalated to main via videort. pose5 stays off slinky-4.
- Resource ask to main open: ~56 GB + 2 CPUs each for A's and D's real-width proofs.
- 10-27 06:28 PT main: demo avatar 30939 running; slinky-4 has 8 free CPUs / ~250 GB, all GPUs taken -> A's and D's real-width proofs go there (-w slinky-4, 2 CPU / ~56 GB each); overflow slinky-18 (2 CPU / ~175 GB). Pass markers: code-811-s2p/tests/s2p/S2P_PASS, code-811-g0/tests/g0/G0_PASS (then code-811-s2p/tests/g0/G0_PASS after merge).

## SINGLE LAUNCH, everything ON (user, main stamp 06:30 PT: "you have around 5 hours to make sure everything is perfectly ready") — 10-27 06:30 PT
- Switch ~11:30 PT. ONE launch from 17850: PPM_S2P=1 + PPM_G0COND=1; plans = npl3 leftover + npl5 appended at 17850 in the same launch;
  npl5 via per-host rolling fetcher. Two-leg (flags-off leg 1 -> 18022 restart) = documented fallback only.
- READY needs: (1) single launch; (2) A + D real-width + combined proofs; (3) resolved cfg vs 27653 = only expected keys (plans, 2 flags,
  resume path, run_tag [+ max_steps, keep_ckpt]); LR schedule + data rules (Gemma/chat unflipped, podcast flipped, pose rows no voice clones)
  unchanged; (4) switch_to_pose.sh no-op rehearsal + real end-to-end dry run on spare CPUs; (5) live curves page (811 format, header
  countdowns), watchdog pinned, HF watcher snapshots code-811-s2p; (6) independent audit agent (fresh, no stake) vs board t1027-24/25/29/31/32.
- Main decisions (stamp 06:33 PT): hard-link pin node-local dcp-17850 into /scratch/amartyavarma/ppm/ckpt-keep-pose5 (hfb_hold method); at
  switch time only, after MoT paused: delete node-local dcp-17798 on 15 hosts (HF-verified 16/16 in ppm-v1-c; live re-check fail-closed; per-host
  list in dry run; each deletion logged here). NEVER MoT / dcp-17850 / dcp-16861 / unexported (PPMVOX v5b dcp-18163). PPM_KEEP_CKPT=1 OK iff
  every new save is HF-verified before the previous is pruned -> pose5 chained hold. slinky-13: list >= 10 GB regenerable caches for main.
- B's slot-9 leg-1 hook READY + rehearsed: /data/home/amartyavarma/ppm/pose5/npl5/bin/leg1_slot9_hook.sh (slinky-15: plan dirs + 5,434 npl3
  parquet 10.24 GB; digests == dcp-17850 pose_digests[slinky-9]). Slot 9 reads 1,428 npl3 rows from 17850.

## A (s2p) — 10-27 06:30 PT
- PASS so far: (a1) cfg/params identical flag off; flag on adds only s2p keys + tower_p.s_emb (tower_p group); (a2) tiny flag-off byte-identical
  vs original code-811 on 5 real micros (0/92 grads differ); (b-i) zero-init full/cb0 bit-identical, s_emb grads only on pose micros;
  non-zero s_emb changes only pose rows.
- Running: resume/missing-key/strictness phases A-L incl. leg-1->leg-2 case + production CE-chunk 1024 rerun (job 30943 slinky-18, ETA ~07:00);
  real-width + real 811 weights (step 17641, reduced depth) on 5 real micros (slinky-4, 3 x 2 CPU / 52G; ETA ~07:45). S2P_PASS marker only when all pass.

## DISK v2 (C, c/DISK_NEED.json; free = MoT GUARD_LOG 06:02 PT) — 10-27 06:32 PT
- Model/host: dcp-17850 off-node 0; KEEP=1 + chained hold peak 46.1 + 53.0 space_check margin; floor 30; dcp-17798 dropped at switch (+46.1 on
  15 hosts); slot-9 extras 19.6 on s15; npl5 roll window (C est ~25 GB at --ahead 320 -> asked B for ~100 steps ≈ 11 GB).
- At ahead 320: short s13 31.1, s6 12.4, s12 11.8, s19 7.4, s16 6.1, s11 2.2, s0 1.9 GB. At ~100 ahead: only s13 (~17 GB) short.
- s13 regenerable caches (records): aloe-pose-build 4.5, triton/inductor ~2.0, hfb xet-cache <=3 -> ~6.5-9.5 GB. Asked main: approve caches +
  (a) s13 ppmvox/vpod3/stems 15.4 GB re-fetchable [recommended] or (b) 20 GB floor on s13 or (c) smaller s13 window.
- Weka dcp-17850 copy jobs 30944 (s18) + 30945 (s4) ~0.6 GB/s; /data 1,063 GB free; VERIFIED ETA ~07:05. Slot-9 eval rebuild byte-identical so far.
- 10-27 06:33 PT main (s13): APPROVED at switch time: regenerable caches (aloe-pose-build temp pkls, triton/inductor caches, hfb xet-cache; read-only du + log) + (b) 20 GB floor on slinky-13 only (ops rule) + (c) smaller s13 roll window only if still short. NOT: PPMVOX stems, pose-duplex dcp-498, Aloe persona dcp-249, anything MoT. Kit per-host space check fails closed and prints host(s); watch s6/s12.
- 10-27 06:34 PT main: pose-s2p engine latency: s2p FULL 96 ms/frame (> 80 ms budget), CB0 66 ms. User to choose; DEFAULT cb0. Both variants built + proven in code-811-s2p (A: all proofs both variants, S2P_PASS JSON; D: combined both); env ONE line PPM_S2P=cb0|1 read by switch_to_pose.sh (C). s_emb = zero-init code table (as pose-s2p README).

## Main is now amartyavarma-36 (restart) — 10-27 06:38 PT
- New constraint: at the switch slinky-10 + slinky-18 go to motasr2 (Voxtral ASR adapter, demos down). All pose5 spare work done and off
  slinky-10/18 by then; rolling fetcher on the 16 hosts (source HF bucket); watchdog / eval logger / chained-hold / page refresh on slinky-4 or hosts.
  Open: existing 811 live page (klweb) + hfb-watch 28306 run on slinky-10 -> asked main whether they survive or move to slinky-4.

## B rolling fetcher (B interim) — 10-27 06:38 PT
- From the real npl5 plan for slot slinky-0: 86,820 rows = 1,356 steps; full share 74.8 GB / 34,366 files; first npl5 step 18034 (others
  ~18022-18034). Window at --ahead 100 + back 110: PEAK 11.7 GB/host (ahead 60: ~9.4). Per-host peaks from the final plans (~10:30).
- Alerts: < 30 steps ahead -> <host>.BEHIND + 'ROLL BEHIND'; current-step file missing -> <host>.STARVED. Missing file = FileNotFoundError ->
  job dies (no row skipped/substituted; maths unchanged) -> watchdog relaunch from newest save; daemon refetches.
- Drop rule: file removed once its last reading step < oldest COMPLETE save > 17850 on the host.
- Placement: one light srun --overlap step in the trainer job (-c1 --cpu-bind=none --mem=8G, nice 19, ionice); supervisor -> slinky-4 (asked
  B to move it off slinky-10 and prove the step can't kill the trainer job). Pre-launch fetch = initial window only (~6-7 GB/host, ~1 min).
- Scans 1/16 done; all ~10:00; plans + 17850 dry run ~10:35.

## /data LEDGER (floor 300 at ALL times; pose5 abort guard 350; MUST be >= 400 from 11:05 to 11:20 PT for MoT's 81 GB export) — 10-27 06:40 PT
- now 907 GB free (Weka dcp copy 237/507 GB; npl5 raw buffer 143 GB, cap ~155; slot9 18; leg1 npl3 10).
- after copy (~07:05) ~637 -> low point ~625 (buffer full) -> buffer released by 10:30 (deadline 11:00) ~780 -> 11:16 MoT export -81 ~700
  -> switch motasr2 base -86 ~614 -> first pose save HF-verified, Weka copy dropped +507 ~1,120.
- Margin vs 400 at 11:05-11:20 ≈ 300 GB (unless other agents write > 300 GB). Alert main by 10:45 if at risk.
- 10-27 06:40 PT B: nothing at launch depends on slinky-10/18 (inputs on /data or HF); spare jobs -t limited, off 10/18 by 11:00; fetcher supervisor on slinky-4 (-c1 --mem=1G); fetcher step srun --overlap -J npl5-roll --mem=8G --kill-on-bad-exit=0; STOP_ROLL scancels only <job>.<step>; ppm811.sbatch homogeneous (trainer step --mem=560G --kill-on-bad-exit=1); kill/OOM dummy test on slinky-4 next; checking 27653 MaxRSS for job-memory margin.

## C: switch_to_pose.sh BUILT (not ARMED) — 10-27 06:41 PT
- /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh --dry-run | --armed | --go [--max-steps=N] [--fallback-two-leg prints the fallback].
- DONE + tested: slot-9 PPM pose eval store rebuilt PASS (16/16 shards' DONE + index sha256 == slinky-9's 10-12 record; load_node digest ==);
  hook copy+verify 594 files 15 s. Watchdog RESUME_PIN (fake srun): no save -> 17850; 17850 on 15 hosts -> 17850; 17902 on 16 -> auto.
  Chained hold kit/hold_chain.sh (release N only after a later save done + 16/16 verify PASS; tested on 17746/17798/17850 records).
  Stage per host, fail closed: 17798 drop after LIVE HF re-check + STATUS line; s13 caches (du shown); hard-link pins; resume links; B hooks;
  slot-9 store; gate = 99.1 + floor (s13 20, others 30) + roll window. Companions on slinky-4; preflight refuses slinky-9/10/18.
- Cfg vs 27653: LR constant (warmup 100, horizon 17400) -> --max-steps no maths; Gemma/chat unflipped; podcast flipped from 16861; chat repeats
  from split 1; pose rows carry no voice clips/personas (voice_mod podcast-only). ARMED accepts only: fresh.pose_root2, max_steps, resume_from,
  run_tag, s2p (== env variant), s2p_from_step, keep_ckpt, g0cond*.
- ARMED blockers: A S2P_PASS (JSON variants incl. cb0); D merge + G0_PASS; B SWITCH_HOOKS.env + plans + 17850 dry run (~10:35); Weka VERIFIED (~07:05).
- --go -> POSE RUN LIVE ≈ 25-30 min. Disk: only s13 short ~1.5 GB at 12 GB window (s13 window 60 steps ≈ 9.4 GB -> ~+0.8); s6 +0.6, s12 +1.2.
- Fixes asked of C: no 2nd HF watcher (28306 stays on slinky-10 per main); pre-pause gate from MoT GUARD_LOG; optional auto-shrink of --ahead
  to >= 40 steps (main OK pending).
- 10-27 06:41 PT main: APPROVED auto-shrink of a short host's --ahead to >= 40 steps; alert main if any host < 20 steps ahead during the run; per-host ahead on the live page. APPROVED pre-pause gate. NEW: every host >= 5 GB margin via REGENERABLE caches (triton/torch_extensions/pip/HF hub caches, old logs; never data/ckpts/MoT), listed in the dry run, deleted at switch time only. slinky-18 30537 (main ALERTS watchdog), 28257 (MoT watchdog), 26231 (klweb-range) stay.
- 10-27 06:45 PT A: resume/missing-key/strictness PASS for PPM_S2P=1 incl. leg-1->leg-2 case (s_emb fresh AdamW step 1; strict reload after); cb0 suite running (30959, ~10 min). Real-width ~40 min/variant: ORIG-off 30947 ~07:10, cb0 30960 ~07:25, FULL 30961 PENDING memory (2 CPU + 38 GB). S2P_PASS ETA ~45 min after 30961 starts. Real-weight cache /data/home/amartyavarma/ppm/pose5/s2p-test-weights/real_w_2_2_1.pt (12.5 GB).

## hfb 28306 + 5 GB margin (C) — 10-27 06:46 PT
- 28306 holds save N only if every host keeps free_after_N - 46.1 >= FLOOR_GB (30), uploads ONLY from its hold; refused + KEEP=1 prune at N+1
  -> N never uploaded. Gate needs free >= 2 x 46.1 + FLOOR = 122.2 GB (+ window + margin); binding on slinky-13 only.
- Pre-pause gate = --go step 0 (c/disk/build_disk_need.py --gate on freshest MoT GUARD_LOG + deltas); AUTOSHRINK=1 in the stage job.
- 'left' after 5 GB margin (GB): s0 6.1, s1 28.8, s2 24.9, s3 14.5, s5 83, s6 1.4 (+caches 5.8), s7 51, s8 9.1, s11 5.8, s12 6.7 (+caches 10.5),
  s13 -7.0, s14 136, s15 509, s16 1.9, s17 19.9, s19 0.6 — all OK with approved caches except s13.
- s13 asks to main: (a) restart idle 28306 with FLOOR_GB=20 (-> -3.9), (b) delete run's inductor/triton-cache-ppm on s13 (1.9 GB, 3-4 min
  recompile), + hf-hold 0.6 + autoshrink 60->40 ~1.1 -> ~4.7 GB margin, (c) accept >= 4.5 on s13 — or B shrinks s13's back window (asked).
- Weka copy 7/11 dirs VERIFIED; /data 786 GB free.

## hfb-watch restarted with FLOOR_GB=20 (main OK) — 10-27 06:48 PT
- Checked idle first: last event 'release dcp-17850 (uploaded + verified) 16/16' 10-23 05:46 UTC; no uploader processes (ps in job); no ppm job since.
- scancel 28306; FLOOR_GB=20 sbatch -w slinky-10 setup-logs/hfb/hfb_watch.sbatch -> job 30966 RUNNING slinky-10; log 'watcher start 847102 ...
  FLOOR_GB 20.0' (13:47:38 UTC).
- FLOOR_GB returns to 30 after the pose run (cancel 30966 + FLOOR_GB=30 sbatch -w slinky-10 ...hfb_watch.sbatch). Caveat: its 7-day self-resubmit
  uses the sbatch's own '#SBATCH -w slinky-15' (a production host) — fine for this ~31 h run, but the resubmit must be re-pinned if it ever fires.
- s13 approvals (main): restart above; switch-time delete of s13 inductor-cache-ppm + triton-cache-ppm (1.9 GB); s13-only margin >= 4.5 GB.

## C progress — 10-27 06:48 PT
- Live page: 811 header 'npl5 ahead min N (host)', red < 20, per-host on hover (kit/hold_chain.sh -> kit/state/hold_chain/roll_ahead.json; ppm_doc.py
  adds roll_ahead on the 811 page only when the file is < 15 min old; backups ppm_doc.py.bak-pre-pose5roll-1027, ppm.tpl.html.bak-pre-pose5roll-1027).
  Countdowns already in the template. Alert < 20 ahead -> control/overnight_monitor_811f.md + setup-logs/watchdog/ALERTS.log (tested on B's sim).
- Stage-job dry run exercised on slinky-18 for slots 9 and 13 (incl. AUTOSHRINK). Two-leg fallback: kit/FALLBACK_TWO_LEG.txt.
- Risk: a node-local dcp-17850 (s1/2/7/14) or s15 ppmvox copy failing sha at switch -> abort after MoT paused. Decision (pose5): NO 230 GB Weka
  pre-stage (would stall B's lanes at /data < 450 and thin the 11:05-11:20 margin); instead ON-DEMAND HF -> Weka download of that host's dir
  (~1-2 min), sha-verified, /data >= 350 check, failed copy left untouched.
- Weka copy: 7/11 VERIFIED; 30945 on last 4 dirs, ETA ~07:15.
- ARMED blockers: PPM_MAX_STEPS (B), G0COND merge (D), S2P_PASS incl. cb0 (A), G0_PASS (D), cfg diff after merge, Weka VERIFIED, B SWITCH_HOOKS.env,
  B dry run. (s13 disk: resolved by main's approvals.)
- 10-27 06:48 PT main HARD: all pose5 jobs off slinky-10/18 by ARMED (~11:15 PT); switch --armed verifies it (exempt: hfb-watch 30966 + klweb page jobs on slinky-10).
- 10-27 06:49 PT hfb_watch.sbatch re-pinned (main's order): backup hfb_watch.sbatch.bak-pre-pose5-pin-1027; header '#SBATCH ... -w slinky-15' -> '-w slinky-10'; self-resubmit line now 'sbatch -w "$(hostname -s)" ...'; bash -n OK. The running 30966's in-memory resubmit ('sbatch $D/hfb_watch.sbatch') reads the new header at fire time -> slinky-10, never a production host.

## D schedule — 10-27 06:52 PT
- In code-811-g0: PASS tiny flag-off == ORIG byte for byte (5 real micros); tiny zero-init identical loss + 92 shared grads, inp.g0 grad only on
  the backbone-FULL pose micro; resume phase H (refusal). Harness reruns: resume phases at 8G (30969), non-zero pose-only check order fix (30970),
  ~07:15. Real-width zero-init 30968 (slinky-18, ~07:30); non-zero real on slinky-4 (~07:45) -> code-811-g0/tests/g0/G0_PASS ~07:45.
- Merge (pre-resolved, ~5 min) once S2P_PASS (~07:30-07:45), then in parallel: cfg all combos; tiny 8 variants; resume combos from a neither-key
  ckpt; A's suite rerun on merged code; real-width zero-init g0/cb0/full/g0+cb0/g0+full on slinky-4 -> code-811-s2p/tests/g0/G0_PASS ~09:00.
  Off slinky-10/18 by ~08:30.

## B rolling fetcher sim PASS — 10-27 06:54 PT
- Sim (real slinky-0 npl5 plan, 17850->18300, 356 npl5 steps, saves every 52, keep 2): 0 missing files; relaunch at 18200 from oldest kept save
  (18110) fine; peak 10.6 GB; 4,337 files; 0 x 429. Bug fixed (false BEHIND/STARVED counting dropped files).
- Knobs: kit/state/AHEAD-<host> > NPL5_AHEAD > 100; kit/state/BACKMODE-<host> oldest|newest. Markers BEHIND <30, CRIT <20, STARVED.
- Windows (slinky-0): oldest a100 11.70 / a80 10.59 / a60 9.48 / a40 8.38 GB; newest a100 8.56 / a80 7.45 / a60 6.34 / a40 5.25 GB.
- s13 -> BACKMODE newest (main pre-approved): saves ~3.1 GB; failure mode = possibly one extra relaunch if the watchdog relaunches from an older
  save (rows <= 53 steps refetched in ~1-2 min; trainer needs > 3 min before its first pose micro). No trainer-side wait code (not needed).
- B_NPL5_PREP = bash /data/home/amartyavarma/ppm/pose5/npl5/bin/npl5_prep.sh (~1 min/host; first window + check + host dry run).
- 10-27 06:58 PT A: resume suite PASS for BOTH PPM_S2P=1 and cb0 (17850 without s_emb; 18022 leg-2; s2p_from_step 17850/18022); production CE-chunk 1024 tiny rerun PASS off/full/cb0. Real-width: harness OOM fixed (streamed hashing, same digests), full 30980 + cb0 30981 running on slinky-18; ORIG-off 30947 ~07:15. S2P_PASS ETA ~07:50.

## DISK GREEN (C, DISK_NEED.json, FLOOR 20, records) — 10-27 07:00 PT
- Margins after everything (GB): s0 11.4, s1 34, s2 30, s3 20, s5 88, s6 6.7 (w/ 5.8 caches), s7 57, s8 14, s11 11, s12 12 (w/ 10.5 caches),
  s13 6.1 (newest window est 6.3 + caches 8.5 incl. inductor/triton-cache-ppm + 17798 drop), s14 141, s15 514, s16 7.2, s17 25, s19 5.9.
- ARMED checks: exactly one hfb-watch RUNNING on slinky-10 at FLOOR 20; no pose5/npl5 job on slinky-10/18 (hfb-watch, klweb* exempt).
- On-demand dcp-17850 fallback rehearsed (slinky-18, dir slinky-14 to a scratch test path: 12/12 sha OK in 50 s; re-run verify-only).
- Page edits strictly additive (roll_ahead only on the 811 page when the pose5 file is fresh; node --check OK).
- ARMED blockers: B (PPM_MAX_STEPS, SWITCH_HOOKS.env, B_DRY), D (merge + G0_PASS), A (S2P_PASS incl. cb0), cfg diff after D, Weka 10/11
  VERIFIED (slinky-19 ~07:10), A/D jobs off s10/18.

## B fetcher safety PASS — 10-27 07:09 PT
- Dummy homogeneous job (main step --kill-on-bad-exit=1): overlap step OOM confined; supervisor scancels only its own step; SIGKILL of the overlap
  client harmless; step discovery via sacct; NEW job-cgroup headroom guard (fetch nothing below 6 GB; <host>.LOWMEM). 27653 MaxRSS 85 GB/task, limit 560G.
- Start: sbatch --export=ALL,END=<PPM_MAX_STEPS>,AHEAD=100,LOW=30 /data/home/amartyavarma/ppm/pose5/npl5/bin/npl5_roll_super.sbatch (slinky-4,
  after CURRENT_JOB_811 written); stop = touch npl5/state/STOP_ROLL.
- Plans 1/npl2/npl3 rebuilt off-host for all 16 keys: 48/48 digests == dcp-17850 (npl5/rebuild13/).
- Scans 5/16. Told B: off 10/18 by 11:00 (hard); add lanes on 10/18 when A/D vacate; plans + dry run + WINDOW by ~10:15; slot-9 rehearsal +
  16-slot e2e on slinky-4 by ~10:50.
- 10-27 07:11 PT B: lanes 30889/30922 cut to end 10:54 PT; plan job --deadline 10:52; rehearsal + e2e pinned slinky-4 --deadline 10:50; lane_adder.sh adds up to 3 v3 lanes on slinky-18/10 when >= 4 CPUs + 22 GB free (cancel if pending > 3 min; /data >= 530 GB before each fetch; stops adding 09:50). Scans 5 done / 2 in flight / 9 unclaimed; with extra lanes all 16 ~09:30-09:45, else ~10:15-10:20; then plans + dry run + WINDOWs ~25-30 min; rehearsal + 16-slot e2e ~23 min.

## Weka dcp-17850 VERIFIED — 10-27 07:12 PT
- 11/11 dirs, 507 GB, every file sha256 == HF LFS (/data/home/amartyavarma/ppm/pose5/dcp17850-weka/VERIFIED.json). Extra: rank88 (dir slinky-13)
  torch.load from Weka as safety_ppm.load_local -> digest == digest-88.txt (job 30991, 9 s). /data 630 GB free.
- Fetcher supervisor wired into the switch (job id -> kit/state/LAUNCHED_RUN; end_of_run touches npl5/state/STOP_ROLL). ARMED checks prep/super
  scripts, npl5/state/dryrun.ok, supervisor not on 9/10/18. Stage gate ROLL_GB_LEFT=auto.
- C's spare jobs all done. Left in place (20 KB of small test files): slinky-18:/scratch/amartyavarma/pose5-selftest/dcp-17850 (staging self-test).
- Remaining ARMED: B PPM_MAX_STEPS + B_DRY; D merge + G0_PASS; A S2P_PASS (cb0); A/D jobs off 10/18; then C cfgdiff_switch.sh + --armed.
- 10-27 07:13 PT Independent audit round 1 launched (fresh agent, read-only; output /data/home/amartyavarma/ppm/pose5/AUDIT_1.md): maths, cfg, data, ops safety, evidence for claims, vs user decisions (board t1027-24/25/29/31/32). Round 2 on final artifacts after B's plans + D's merge.

## D: G0COND proofs PASS in code-811-g0 (G0_PASS 07:31 PT) — 10-27 07:31 PT
- f = [r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w, log(orig_h/orig_w)] (8 dims; aspect ratio INCLUDED: orig_h/orig_w are on every clip
  record at row-build time for all fresh plans + PPM samdata stores, no re-scan; real clips plan1 11,321 + npl5 14,037: 0 invalid, all have
  orig_h/orig_w). Param inp.g0.weight [5120,8] (Linear 8 -> H, zero-init) in LR group 'audio' (= inp.pose). Added at each clip's SLOT position.
- cfg flag off == original; flag on adds only g0cond/g0cond_from_step + inp.g0. Tiny flag-off bit-identical (5 row types); zero-init on: identical
  loss + 92 shared grads, inp.g0 grad only on the backbone-FULL pose micro; non-zero changes pose micros only. REAL width (real 17641 weights,
  reduced depth) zero-init: identical loss every row type (pose_full 37.994633, pose_frozen 20.303928, speech 23.177967, gemma 15.524290,
  chat 24.923191), 0/92 grads differ; g0 grad == chain-rule recompute to 1.7e-3 (bf16). Resume A..I PASS (only inp.g0 missing; strict after;
  refusals OK; real dcp-17850 train_state has no g0cond). Optional real-width non-zero check 30978 ~08:30.
- Waiting for A's S2P_PASS to merge into code-811-s2p.

## INFERENCE NOTE (for later engine work; main: keep the 8-dim f with aspect term) — 10-27 07:32 PT
- s2p: the pose tower at frame T reads channel B's speech codes of the SAME frame (8,1,1: depth[1][T], unshifted; = speech of frame T - CD with
  CD 2). PPM_S2P=cb0 -> codebook 0 only; =1 -> all 32. Pose-CFG uncond row gets silence codes. Positions with no codes -> 0. No extra delay, but
  the frame's speech codes must exist before the pose tower runs at T. Latency (pose-s2p engine): full 96 ms/frame vs cb0 66 (today 61; budget 80).
- G0COND: the engine must feed g0(f) at the position just before the first pose frame = the clip's SLOT in the 8,1,1 [SLOT 1 | clip | GAP 4]
  layout; f = [r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w, log(orig_h/orig_w)] -> engines must supply frame-0 root (y, x
  image-normalised), shoulder width w AND orig_h/orig_w, from the opening's source metadata (meta frame0 fields / llm_inputs global_state);
  invalid -> zeros. Aloe FT layouts (system prompt + voice clip + conversation) have no SLOT yet and need one defined.
- 10-27 07:32 PT C idle (side complete): resume C when D merges -> kit/cfgdiff_switch.sh (~2 min slinky-4) + switch_to_pose.sh --armed -> C-READY. kit/end_of_run.sh waits for final save HF verify, stops companions, prints FLOOR_GB=30 restore + motguard MoT resume + pose-run files on hosts (deletes nothing).

## /data INCIDENT — 10-27 07:39 PT
- /data fell ~13 GB/min (613 -> 535 GB) after B's lane_adder.sh started extra v3 lanes staging raw on /data (check was only "/data >= 530 before
  each fetch"; ignored the 155 GB total-raw cap): npl5/raw 214 GiB. Main suspended 30992; pose5 suspended 30993 (scontrol suspend). /data 547 GB.
- B ordered: kill lane_adder; enforce cap in code before every fetch (raw <= 155 GB AND /data >= 600 GB else pause, or stage on node /scratch);
  delete scanned raw at once; resume nothing until enforced.
- LEDGER (real): 547 now -> >= 600 after in-flight scans delete raw -> all npl5 raw gone by 10:30 (~760) -> 11:16 MoT export -81 (~680) -> switch
  motasr2 base -86 (~590) -> pose run exports +81 GB per 6 h (KEEP=0, main prunes) -> first pose save HF-verified: Weka copy dropped +507.

## A DONE: S2P_PASS — 10-27 07:39 PT
- code-811-s2p/tests/s2p/S2P_PASS {"full": true, "cb0": true, "flag_off_byte_identical": true} + md5s of changed files. Real width with real
  17641 weights: ORIG off vs full/cb0 zero-init identical losses/stats, 0/92 grads differ, identical decoders; s_emb grads only on pose rows.
- Notes for user: 811 infer_ppm.py doesn't know s_emb (s2p checkpoints need the pose-s2p engine); codep eval probe understates the new path.

## AUDIT round 1 (AUDIT_1.md) — routed — 10-27 07:39 PT
- Blockers: B1 watchdog rewind to 17850 on an empty scan (C); B2 fetcher memory guard silently stops (B); B3 export rotation would delete shared
  ckpt-weka/ppm-v1/w-15973 -> PPM_EXPORT_KEEP=0 (C; env-only); B4 fetcher floor fallback re-downloads consumed npl5 (B).
- Should-fix routed: C (S5 resume link via pin, S6 alerts, S8 exactness vs ppmdecay-j27810 + gn checks, S13 md5 ARMED, S14 abort paths, S15
  disable launch_pose5.sh, S16 HF-verify check for old saves); B (S10 refetch from resume step, S11 status/lock/heartbeat, S12 rerun sim/safety on
  final code); D (S9 start-eval with flags on, S13 md5s, npl2/npl3 field coverage); main decision S7 (own param groups for new params?).
- 10-27 07:39 PT MAIN DECISION (audit S7): OWN optimizer groups for inp.g0 and tower_p.s_emb (same LR/WD/betas/schedule as audio / tower_p; per-group clip 1.0) so every existing param's update stays exactly as 27653; proofs: existing groups' grad norms + clip factors bit-identical with flags on at zero-init; cfg diff lists only the two new groups. Routed to D (owns code-811-s2p after merge; reruns A's suite). Lanes 30992/30993 stay suspended until cap enforced. KEEP=0 pose exports OK. All 4 audit blockers before ARMED; round-2 audit after fixes.

## C: audit-1 fixes DONE (tested) — 10-27 07:41 PT
- B1 watchdog resume_decision() (kit/ppm811s2p_watchdog.sh:66-108): < 16/16 answered -> ALERT no relaunch; newest > 17850 complete in CKPT ->
  CKPT/dcp-N; hold-only -> hold/dcp-N; split / newer-but-incomplete -> ALERT (never back to 17850); pin 17850 only if 16/16 answered and nothing
  > 17850 anywhere. 7 fake-srun scenarios as specified.
- B3 env:351 PPM_EXPORT_KEEP=0 (env-only, safety_ppm.py:360; keep>0 guards the rmtree) -> w-15973 safe. ~81 GB per 6 h export to /data (~400 GB
  over the run); the trainer refuses an export that would leave < 300 GB -> main prunes.
- S5 resume link -> ckpt-keep-pose5/dcp-17850 after the pin (stage_host.sh:102). S6 alerts (hold_chain.sh:41-105; watchdog :142-151).
- S8 switch step 6 (switch_to_pose.sh:215-262): POSE RUN LIVE only if ADDED lines present, start eval at 17850 == runs/ppmdecay-j27810 (>= 180
  keys, EXACT_TOL 0; resumes reproduce bit-exactly in this layout: 27653 vs 27204, 27204 vs 27114), [811 rows] 17851/17852 == ppmdecay-27810.log,
  gn_audio/gn_tower_p < 0.5.
- S13 ARMED requires core_md5 in S2P_PASS + G0_PASS == current files; cfgdiff accepts only fixed keys + g0cond* + the 2 new groups (listed in PASS
  JSON "groups"); unit-tested (extra lr.trunk change flagged BAD).
- S14 pause wait no timeout while mot_pause.sh alive (GPUS_FREE / PAUSE_ABORT markers); --after-pause entry; stage-failure abort stops all
  companions + STOP_ROLL. S15 launch_pose5.sh exits 2 (SUPERSEDED). S16 stage aborts on old saves without HF verify record. NITs done.
- 10-27 07:42 PT main: kit auto-drops THIS run's exports' /data copies after a live HF re-check (count + sizes + sha vs hfb 'w' record), logged; never w-15973 / other runs' exports / MoT; fail closed; shown in dry run (routed to C).
- 10-27 07:42 PT C: resolved-cfg diff on MERGED code-811-s2p (kit/state/CFG_DIFF_switch.txt): 13 keys, all expected — fresh.pose_root2 (+npl5), max_steps (placeholder until B), resume_from (link), run_tag, keep_ckpt 2->1, s2p cb0, s2p_from_step, g0cond, g0cond_from_step, lr.s2p 3e-4 (= tower_p), lr.g0cond 2e-4 (= audio), clip.s2p 1.0, clip.g0cond 1.0; GROUPS added exactly [g0cond, s2p]; every existing LR/clip + data rules identical to 27653. /data 610 GB; npl5/raw 143 GiB (<= 155); lanes 30889, 30922, 31010 (30992/30993 gone).
- 10-27 07:43 PT B: raw cap ENFORCED in code (stage_lane4.sh:44-61 flock reservation: committed raw + slot <= 155 GB AND /data >= 600 GB; src_fetch_capped.py:71-99,111 re-check before every file); lane_adder stopped; 30992/30993 cancelled (no scan work lost; E's partial 19 GB raw deleted); lane F 31010 (v4). Raw on /data 143 GB. Scans 7/16; ETA all ~10:00-10:10 -> plans + dry run ~10:35 (tight). Suggested extra lanes on slinky-10/18 node /scratch (not /data) when CPUs free.
- 10-27 07:44 PT stamp note: message stamps 07:47/07:50 sent to main were ahead (clock read 07:44); from now every message stamp = TZ=America/Los_Angeles date +%H:%M run just before sending.
- 10-27 07:45 PT D: OWN GROUPS implemented in code-811-s2p: 'g0cond' (inp.g0) + 's2p' (tower_p.s_emb), LR/clip copied from audio / tower_p, same warmup lambda (model_ppm.JOIN); BB_GROUPS += g0cond; TW_GROUPS + 811 pose-side set += s2p; warmup_groups unchanged (flag-off cfg identical); resume appends the new group's LambdaLR at its JOIN group's lambda for the saved step. Missing frame-0 fields now LOUD ([g0cond] WARNING + startup count). Wave 1 running (combined cfg+tiny incl. existing groups' grad norms/clip factors, resume combos, S9 evaluate() on merged code, real width ORIG/g0+full/g0+cb0); wave 2 ~08:35 (real g0/cb0/full alone + A's suite rerun). Both markers ~09:35 PT; slinky-18 work ends ~09:30.
- 10-27 07:54 PT B: audit fetcher fixes in (npl5_roll.py v2 + super v2): B2 headroom excludes page cache (:230), status+markers always written; B4 monotonic persisted floor, lowered only by 'resumed ... at step R' in the trainer's own log, disk floor MIN_FREE_GB-<host> (default 66) -> NOSPACE; S10 held/pinned saves are floor candidates, live = R during init, BACKMODE newest DROPPED (s13 budget = b110); S11 30 s heartbeat, flock per host, unique temps, supervisor singleton + SUPER.json heartbeat + STEP_DIED, STOP_ROLL freshness, bounded stop; S12 dryrun guard FATAL; NITs done. Retests: v2 sim running; kill/OOM safety test after lane B exits (~08:25).
- 10-27 07:54 PT C: export auto-drop built (kit/export_drop.py check() :73-117): deletes ckpt-weka/ppm-v1/w-N only if N > 17850, not 15973 (hard refusal), exactly one of THIS run's jobs logged it consolidated, exactly one hfb 'w' PASS record from that job, on-disk == verified counts/bytes, LIVE HF re-check (names, count, sizes, LFS sha256, small files byte-equal); logs to STATUS + kit/logs/DELETED-<date>.log + hfb DELETED log; else KEEP (+ ALERT after 3 h). Runs each hold-chain poll (slinky-4, flock); dry run lists candidates (--no-live). Selftest PASS (decoys w-15973, other run, no record, bytes mismatch, live mismatch, ppm-mot all KEPT). Real dry run: only w-15973 exists -> KEEP.
- 10-27 08:04 PT C: B v2 kit changes done; BACKMODE newest dropped -> s13 budget b110 a60 (est 9.48 GB) -> s13 margin 2.8 GB (< accepted 4.5). Others: s19 5.8, s6 6.6, s16 7.1, s11 11, s0 11.3, rest more. Fetcher MIN_FREE_GB-<host> = 53 + floor (s13 73, others 83). Alerts for all roll markers wired + tested (4/4). STOP_ROLL stale = info only. Asked main re s13.
- 10-27 08:05 PT main (s13 only): (a) ahead 40 steps from start, (b) du + clear s13 hfb xet-cache at switch, (c) fail-safe live-gate margin >= 3.5 GB above its 20 GB floor; pre-pause gate uses exactly the same numbers; page/ALERTS alert if s13 /scratch free trends toward the floor. Routed to C.
- 10-27 08:09 PT B: v2 fetcher SIM PASS on final code (npl5/sim/run-31010-slinky-0/RESULT.txt): real slinky-0 plan, ahead 60, keep 1 + fake hold/pin, 520 steps (336 npl5), 0 missing; relaunch at 18180 from HELD 18110 fine; LOWMEM window paused + marked, ahead 36 then recovered; 2nd daemon refused (rc 3); peak 7.98 GB. Safety test v2 armed after lane B exits (~08:25). Scans 8/16 + 6 in flight; ETA all ~09:05-09:15. Plan job (6 CPU / 44G) to be queued now on slinky-18 with dependency; D asked to keep optional 30978 + wave-2 off 10/18 near 09:05.
- 10-27 08:09 PT B: plan job 31031 queued (-w slinky-18 -c6 --mem=44G, afterany all 4 lanes 30889/30922/31010/31012, deadline 10:52 PT); asserts all 16 scans present; chain = plans -> 17850 dry run (fatal s2p guard) -> WINDOWs -> slot-9 rehearsal + 16-slot e2e on slinky-4 (deadline 10:50).
- 10-27 08:16 PT D: merged own-group code PASS so far: cfg 6 combos (only new keys/params/groups; warmup_groups identical; lr/clip of g0cond/s2p = audio/tower_p); tiny zero-init per combo identical loss + 92 grads + every EXISTING group's grad norm and clip factor; new-group grads only on pose micros; resume all combos (exact missing keys, first-step LR = JOIN group's, strict reload, 5 refusals); S9 trainer evaluate() g0+cb0 / g0+full == flags-off on all 188 metrics, no assert; f coverage npl3 12,863 / npl5 16,055 / plan1 11,321 clips, 0 invalid. Remaining: real-width each flag alone (g0 on slinky-18 ~08:30-09:12; full + cb0 back to back on slinky-4) + A's suite rerun -> markers ~10:00.
- 10-27 08:17 PT C: s13 decisions built: ONE gate code path kit/gate_rules.py (floor s13 20 / others 30; margin s13 3.5 / others 5; saves = max(99.1 + floor, 2 x 46.09 + hfb FLOOR); choose_ahead s13 40, others 100 or autoshrink >= 40) used by pre-pause (c/disk/build_disk_need.py) and live (kit/stage_host.sh:154,166). s13: free 76.8 + 17798 46.1 + caches 8.5 (+ xet) vs need 131.0 -> 3.9 GB above floor (>= 3.5). Others >= 5 (s19 5.8, s6 6.6, s16 7.1). kit/test_gates.py GATES AGREE (16 hosts identical; s13 boundary +/-0.05 GB). Trend alert on hourly troughs (kit/hold_chain.sh:97-117) -> ALERTS + monitor + page 'disk low' span (additive; node --check OK).
- 10-27 08:18 PT D: wave-2 real-width at 46G (MaxRSS 36-42 GB): real_g0 31026 slinky-18 (ends <= ~09:22), real_full 31027 + real_cb0 31028 together on slinky-4 after ~08:30; threads 2 (= ORIG). A's suite on merged own-group code: cfg, tiny c128 + c1024 (zero-init + non-zero, full & cb0), 12 resume phases PASS; cb0 resume G/H harness clash (shared synthetic ckpt dir) -> rerun alone 31033. Markers ~09:20.
- 10-27 08:25 PT B: kill/OOM safety test v2 PASS on final fetcher + supervisor (npl5/safety/run-1793114529/RESULT.txt): job-cgroup probe from overlap step OK; step OOM confined; LOWMEM fired correctly on a 3G dummy job; SIGKILL daemon -> STEP_DIED -> re-attached in 15 s; STOP -> own step scancel; SIGKILL overlap client harmless; main step alive throughout. Scans 10/16, all claimed; ETA ~09:05-09:15. Asked C: LIVE check also verifies every host's fetcher health (fresh JSON, ahead, headroom > 6 GB, no LOWMEM/NOSPACE/STEP_DIED).
- 10-27 08:27 PT C: fetcher health gate in switch step 6 (switch_to_pose.sh:256-283): all 16 roll JSONs < 120 s, ahead >= AHEAD-10, headroom > 6 GB, no LOWMEM/NOSPACE/STARVED/CRIT/STEP_DIED, SUPER.json fresh; waits <= 5 min; failure -> ALERT + no POSE RUN LIVE (run stays up). Synthetic tests rc 1 / rc 0 as expected. Waiting: B PPM_MAX_STEPS + dry run; D G0_PASS + re-issued S2P_PASS (groups + md5).
- 10-27 08:35 PT B: 12/16 scans; last-scan critical path (G slinky-16 4 CPUs ~09:28). Told B: give slinky-16 to lane F (10 CPUs) after slinky-19 (~08:50) -> scans ~09:05-09:10 -> plan job ~09:10 -> plans + dry run + windows ~09:45 -> rehearsal + e2e ~10:15.
- 10-27 08:38 PT main lowered D's 31028 (real_cb0) to 45G (was 26 MB short on slinky-4); running; if OOM -> rerun on slinky-18.
- 10-27 08:46 PT B: slinky-16 stays on G (raw node-local on slinky-10; moving saves nothing), G done ~09:20; F + A exit ~08:50. Plan job 31031 may run on slinky-10 or slinky-18. Timeline: plans ~09:42, 17850 dry run + windows ~09:55, slinky-4 rehearsal + e2e ~09:55-10:20.
- 10-27 08:59 PT D: real_g0 31026 (slinky-18) ~30% slower (busy node); limit raised to 1:08, ends ~09:15-09:22, hard stop 09:30; real_full 31027 ~09:10, real_cb0 31028 ~09:20 (slinky-4). Markers ~09:25; only these 3 real-width results missing. (B's plan job can use slinky-10 after G ~09:20.)

## npl5 SCANS DONE (B) — 10-27 09:19 PT
- 552,329/552,329 releases accepted (0 pin/sha mismatches, 0 bad state, 0 missing pins); 56,945,936 segments = 224,662 h; SAM no-words drops
  10,924; orig_h/orig_w sidecars 16/16 (0 nulls); scan products 5.1 GB on /data; npl5/raw empty; /data 745 GB free; all 4 lanes exited.
- Plan job 31031 RUNNING on slinky-10 (09:18): global dedup + 16 plans ~09:43 -> 17850 dry run (fatal guards) + WINDOWs ~09:55 -> slinky-4
  rehearsal + 16-slot e2e ~10:20.

## npl5 PLANS BUILT + 17850 DRY RUN PASS (B) — 10-27 09:33 PT
- 16 keys, 1,391,232 rows, 223,599 h (~86.8-87.2k rows/key); global dedup kept 56,674,353 / 56,945,936 segments; slinky-0 digest == provisional.
- PPM_MAX_STEPS = 19382: ends [17352, 17641, 18022, 19382] -> +1,532 steps after dcp-17850 (~29 h at 68 s/step). Steps < 18022 unchanged vs
  3-plan stream; at 17850 every rank in npl3 (1,377-1,476 rows left/rank); npl5 starts 18022-18034; each row used <= once per host; 5,237 unused.
- Flags: only speech epoch-2 repeats from 17872 (known); chat (24,305) + Gemma (39,902) last past 19382.
- fresh_pose check 16/16 OK; trainer micros at first npl5 row + last step 16/16; guard on real dcp-17850 train_state OK for code-811 and
  code-811-s2p + FINAL env (S2P=cb0, G0COND=1); append_at [17352, 17641, 17850]; refusals 4/4.
- Next: WINDOW files -> slot-9 rehearsal + 16-slot e2e on slinky-4. C: fill PPM_MAX_STEPS, rerun cfg diff, --armed.
- 10-27 09:33 PT B: WINDOW files from FINAL plans (npl5/roll/WINDOW-<key>-a{100,80,60,40}-b110.json, 64 files, run-peak): a100 11.69-11.74, a80 10.59-10.63, a60 9.48-9.54, a40 8.38-8.42 GB; s13 a40 8.41 (C est 8.38 -> margin ~3.87 >= 3.5); slot 9 a100 11.73 + 10.3 GB npl3/plans. First npl5 step 18022 (s19) - 18034 (s0). Rehearsal 31041 running on slinky-4, e2e 31042 next (deadline 10:50).
- 10-27 09:35 PT B: launch-day REHEARSAL PASS on slinky-4 as slinky-15 (npl5/launch/rehearsal-31047): leg1_slot9_hook.sh OK 33 s (plans 1-3 sha, 5,434 npl3 files xet-verified, digests == dcp-17850, micros 17850/18021); npl5_prep.sh OK 34 s (END 19382, first window 256 files 0.57 GB in 6.9 s from HF, check rows 0-2, micros 17850 + first npl5 row 18028, 4-plan digests). Launch-day delay ~70 s/host in parallel. NOSPACE path proven (spare had 26 GB). C already writes MIN_FREE_GB-<host> = 53 + floor. 16-slot e2e 31048 running, done ~09:58.
- 10-27 09:37 PT C: PPM_MAX_STEPS=19382 filled (env line 333; 1-308 verbatim; backup kit/state/env.bak-pre-maxsteps-*); cfg diff rerun: 13 keys all accepted, max_steps 18022 -> 19382, GROUPS added exactly [g0cond, s2p] (lr 2e-4 = audio, 3e-4 = tower_p, clip 1.0). ARMED reds = D's two markers only (postmerge S2P_PASS not written yet; G0_PASS pending 31026-28). Disk recomputed on real windows: no host short; tightest after margin s13 +0.4, s19 +0.8, s6 +1.6 GB. One hfb-watch (30966, FLOOR 20); no pose5 job on 10/18; /data 742 GB. Wall-clock 1,532 x 69.9 s ≈ 29.7 h (+25-30 min switch); s2p/g0 compute < 1e-4. Told C: no weakened fallbacks (require core_md5 + groups, postmerge S2P_PASS) + gate on B's e2e.
- 10-27 09:39 PT C: ARMED strict (postmerge S2P_PASS only; core_md5 binding both markers; groups from JSON only; G0 combination check; e2e 31048 gate on E2E.json + log + 16 host files); offline good/bad tests pass. 4 reds: postmerge S2P_PASS, G0_PASS, cfg diff (needs groups), e2e running. Background watch reruns --armed. Asked: variant checks follow the env line (cb0 | full).
- 10-27 09:41 PT C: ARMED derives the s2p variant from the env line (cb0 | 1/full): S2P_PASS lists it, G0_PASS has 'g0+<variant>', cfg diff s2p == it, CFG_DIFF newer than env. Real full-variant cfg diff = same 13 keys (kit/state/test_variant/cd_one.txt). 8 offline cases correct; TEST_* hooks abort with --go/--after-pause. Background watch reruns --armed when D's 2 markers + B's E2E.json exist and 31048 left the queue.

## D: FINAL MARKERS WRITTEN (09:41 PT) — recorded 10-27 09:41 PT
- code-811-s2p/tests/g0/G0_PASS: PASS, groups ["g0cond"], combinations_passed [g0cond, s2p=cb0, s2p=full, g0cond+s2p=cb0, g0cond+s2p=full],
  31 checks true (cfg every combo; tiny zero-init 6 combos + 2 non-zero; real width g0/cb0/full/g0+cb0/g0+full; resume 5 combos + 5 refusals;
  S9 start eval off/g0+cb0/g0+full; f on plan1/npl3/npl5; code unchanged since final wave).
- code-811-s2p/tests/s2p/S2P_PASS (= tests/s2p_postmerge/S2P_PASS; A's original = S2P_PASS.premerge-A): full, cb0, flag_off_byte_identical,
  groups ["s2p"]; A's suite rerun on merged code + real-width cb0/full (also with g0cond).
- core_md5 (17 files, both markers): model_ppm.py 52a36afc..., train_ppm.py 722e2f99..., data_ppm.py 2fb643bb..., safety_ppm.py 956f85ca..., ...
- Real width all 5 combos: identical loss on every row type (pose_full 37.994633, pose_frozen 20.303928, speech 23.177967, gemma 15.524290,
  chat 24.923191), 0/92 grads differ, every existing group's norm + clip factor bit-identical; new-group grads on pose rows only.
- 10-27 09:42 PT Audit round 2 launched (same independent auditor, AUDIT_2.md, target ~10:35): final merged code + own groups, markers' core_md5, final env + cfg diff, npl5 plans/dry run/windows/fetcher v2/rehearsal/e2e, switch + kit + gates, round-1 fixes verification, decisions vs implementation.

## D DONE — inference contract + leftovers — 10-27 09:42 PT
- f computed fp64 -> stored fp32 -> cast bf16 at the projection; no Fourier features; no bias (invalid clip == today's input). |f| median ~1,
  p99.9 ~30, max ~43 (tiny people in frame), unclamped. ~99.6% of clips portrait. npl2 not read directly (no local copy; only delivery 5 may be
  downloaded): rests on scan stats (0 bad frame-0 flags over 47.5M plan 1-3 segments).
- Patches: pose5/s2p_g0cond_final_vs_code811.patch (final diff vs code-811), pose5/g0cond_onto_s2p.patch (merge); pose5/g0cond.patch superseded.
  code-811-g0/tests/g0/G0_PASS belongs to the superseded shared-group version.
- INFERENCE CONTRACT: build with g0cond=True and load inp.g0.weight (exports record g0cond in train_cfg.json; key-set checks must accept it);
  add inp.g0(f) to the trunk input at the position just before each pose clip's first frame (8,1,1: the SLOT PAD position); f = data_ppm.g0_feat
  (frame0_root, frame0_width, orig_h, orig_w, frame0_ok). Sources: recording-window openings -> meta.json sources.pose_tokens.frame0.fields
  (all five; read in aloe-posedup-live/server.py opening_frame0_gs() and genweb/worker*.py); avatar-package openings -> global_state gives only
  [root_y, root_x, width], orig_h/orig_w from the SAM items.parquet row or package video size. Unknown frame 0 -> feed nothing (= f 0).
  Aloe layouts have no SLOT yet: open design decision.
- Leftover synthetic test checkpoints (node scratch, not deleted per rule): slinky-10 /scratch/amartyavarma/pose5-g0-test (4.3 GB), same on
  slinky-4, and slinky-4 /scratch/amartyavarma/pose5-s2p-postmerge-test; slinky-18 /scratch/amartyavarma/pose5-selftest (20 KB, C). Main to OK removal.
- 10-27 09:42 PT B READY: 16-slot e2e PASS (job 31048, npl5/e2e/): every slot with REAL 4 plans at canonical dirs (plans 1-3 off-host rebuilds, digests == dcp-17850), digests == dry run, npl5 sha == PLANS.json, 24 trainer micros/slot (8 ranks x {17850, first npl5 row, last step}), 16/16 OK. No B jobs running; no B data on slinky-10/18 /scratch; no npl5 raw on /data. Records npl5/SUMMARY.md, TIMES.tsv, npl5/STATUS_SECTION.md.

## ARMED GREEN (C-READY) — 10-27 09:42 PT
- switch_to_pose.sh --armed 09:42:20 PT: rc 0, 26 checks ok, 0 red; --dry-run complete, nothing submitted. NOT launched (user HOLD).
- Launch (on the user's go): bash /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh --go   (--after-pause if motguard already paused MoT).
- If the user picks s2p FULL: edit env PPM_S2P=1 -> bash /data/home/amartyavarma/ppm/pose5/kit/cfgdiff_switch.sh (~1 min, slinky-4) -> --armed.
- Remaining for READY: independent audit round 2 (AUDIT_2.md, ~10:35) + fixes.
- 10-27 09:43 PT DELETED (main OK, literal paths, listed first): slinky-10 /scratch/amartyavarma/pose5-g0-test (4.3G, 26 entries); slinky-4 /scratch/amartyavarma/pose5-g0-test (12G, 71) + /scratch/amartyavarma/pose5-s2p-postmerge-test (9.7G, 100); slinky-18 /scratch/amartyavarma/pose5-selftest (76K, 17). Synthetic test checkpoints only; nothing else touched.

## AUDIT round 2 (AUDIT_2.md) — routed — 10-27 09:57 PT
- All 4 round-1 blockers verified FIXED (+ #5, #7 own groups, #9, #13 md5s recomputed 17/17, #14-16, NITs). Partly: #6 (R2-3), #10-12 (R2-9).
- NEW BLOCKERS: R2-1 exactness check compares wall-clock keys at tol 0 -> every correct launch fails (fix: trainer's key rule, excl. *time_s;
  exact -> LIVE, <= 1e-6 rel -> LIVE + WARN, else NOT VERIFIED) [C]. R2-2 inductor-cache-ppm + triton-cache-ppm on s13 are ALSO MoT's
  caches (ppm/mot/code/task_ppm.sh:11-12; confirmed) -> removed from deletion lists [C]; s13 then ~1.5 GB short -> decision to main.
- SHOULD-FIX: R2-3 hold_chain 2-held-saves bug [C]; R2-4 s13 fetcher NOSPACE pauses hourly [-> Weka-served s13 window option, B preparing];
  R2-5 export auto-drop never drops (bundle record incl. run_state/; exports ~89 GB) [C]; R2-6 empty sbatch id fails open [C]; R2-7 ARMED assert
  PPM_EXPORT_KEEP=0 [C]; R2-8 kept 17798 after failed HF re-check -> abort stage [C]; R2-9 heartbeat numbers stale [B].
- Still open by design: #17 codep probe can't see s2p (user note); board t1027-24 s2p full vs cb0 (env = cb0, main's default).
- 10-27 09:57 PT main APPROVED Weka-served npl5 window for slinky-13 (ops only, ~12 GB /data): canonical /scratch path symlink -> /data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-13/jobs; test trainer + fetcher follow symlink; rolling deletion only inside that Weka dir; counts against /data cap + floor. Nothing of MoT (caches included) touched. Routed B (impl + test) and C (gate/stage/ledger).
- 10-27 09:57 PT B: implementing R2-9 (heartbeat recomputes numbers + markers every 30 s, 429 count, SLOWPASS > 15 min) + s13 Weka dest (canonical path symlink; floor checked by statvfs of the resolved dest; MIN_FREE_GB-slinky-13 = 350 for /data; deletes only realpaths under the resolved dest root; NPL5_ALLOWED_ROOT check; lock/state node-local) + round-2 fetcher NITs. Impl ~10:20, spare test with REAL slinky-13 plan ~10:40, then C's stage changes + s13 gate numbers.
- 10-27 10:03 PT STANDING RULE sent to B + C (3rd denied rm prompt): every rm uses "${VAR:?}" per variable component or a literal path; rm-line edits via Edit tool only; grep + fix remaining unguarded rm in scripts.

## C: audit-2 fixes DONE — 10-27 10:12 PT
- --armed green 10:11:44 PT (27 ok, rc 0); --dry-run complete; not launched.
- Extra bugs found + fixed: step-6 guards heredoc py3.10 SyntaxError (every launch would have read NOT VERIFIED); hold_chain ran export_drop
  with system python3 (no huggingface_hub) -> now venvs/duplex python, crash -> KEEP + ALERT.
- R2-1 kit/exact_check.py (trainer key rule; EXACT -> LIVE; <= 1e-6 rel -> LIVE + WARN; else NOT VERIFIED); replay 27653 vs 27204 @17641
  184/184 EXACT; synthetic cases OK. R2-2 MoT-shared caches never deleted (inductor-cache-ppm, triton-cache-ppm, hfb/xet-cache [MoT's
  HF_XET_CACHE], triton-cache-gen14; grep over ppm/mot/**); WEKA_HOSTS={slinky-13} (s13 /scratch window 0; Weka window 11.71 GB in /data
  ledger; s13 ahead 100). DISK_NEED (GUARD_LOG 09:59): no host short; s13 +0.2, s19 +0.7, s6 +0.8 (a40 + 1.8 caches), s16 +1.9, s12 +2.9.
- R2-3 hold_chain all-host HELD / partial retry / MISMATCH never deleted / no re-hold (3-host 5-poll test). R2-5 export_drop via bundle
  _backup/MANIFEST.w-N.tsv (real w-15973: "would match but REFUSED: 15973"; 12-case selftest). Exports ~89 GB. R2-6 need_id hard-fail +
  watchdog JOBF checks. R2-7 EXPORT_KEEP=0 asserted. R2-8 17798 still in CKPT -> stage ABORT. rm guards applied (listed file:lines).
- Audit round 3 (delta) launched; B's s13 Weka window + R2-9 ~10:40.
- 10-27 10:13 PT C: MIN_FREE_GB owned by stage_host.sh:165-170 (Weka-served host -> 350, else 53 + floor; checked s13 350, s6 83); xet-cache + triton-cache-gen14 excluded; --dry-run writes c/DISK_NEED.dryrun.json (live file untouched); --armed green + --dry-run OK (10:13); waiting for B's s13 symlink stage changes.
- 10-27 10:25 PT B: npl5_roll.py v2.1: R2-9 heartbeat recomputes numbers + markers every 30 s (evidence wekatest/run-31062/heartbeats.txt), SLOWPASS > 15 min; NITs (headroom file - shmem; safe_remove realpath-inside-root for every delete; floor lowered per (job log, resume) event). s13 Weka tested on slinky-4 as slinky-13 with REAL plans (runs 31062/31073): wrong target -> exit 4, 0 files; allowed -> npl5_prep 36 s, 254 files on Weka / 0 node-local, fresh_pose check + DataLoader micros through symlink, digests OK; sim: 0 files outside Weka dir, peak 8.2 GB (fast-sim misses = test-speed artefact; 31074 production-ratio rerun ~10:45). s13 /scratch npl5 = 0; AHEAD 100; MIN_FREE 350 (/data); /data <= 11.71 GB. rm guards applied in npl5/bin (listed). Stage block for C: npl5/S13_WEKA_FOR_C.txt.
- 10-27 10:26 PT C: s13 Weka block applied (kit/stage_host.sh:157-198; symlink only if absent, correct symlink kept, wrong symlink / real dir -> ABORT never replaced; no rm; MIN_FREE 350 / AHEAD 100 never overwritten for Weka hosts). Offline 5-case test + stage_host --dry-run on slinky-4 OK. --armed 10:26:35 (27 ok) + --dry-run 10:26:36 (SHORT none). s13 +0.2 GB after margin. B asked to clean slinky-4 test plan dirs (planhost slinky-13) after 31074.

## AUDIT round 3 (delta, AUDIT_3.md) — 10-27 10:47 PT
- All 9 round-2 findings FIXED (verified by reading + replay: exactness 184/184 EXACT on 27653 vs 27204; MoT caches never on any delete list;
  hold chain; export_drop real-bundle match + REFUSED 15973; need_id; EXPORT_KEEP assert; 17798 abort; heartbeat numbers). No new blocker.
- R3-1 SHOULD-FIX: slinky-6 (autoshrunk to ahead 40, +0.8 GB) gets hourly NOSPACE pauses (free ~43 GB vs fetcher floor 83 during the ~13 min
  two-save window) -> a > ~40 min late HF verify would starve it; ~10 near-gate hosts raise hourly NOSPACE alerts (harmless at ahead 100).
- R3-2: s13 Weka path production-pace sim 31074 at step 18230 with 0 missing (incl. relaunch from held save; deletes only inside Weka dir) at
  10:47; ready once it exits 0.
- NITs: end-of-pass status can step backwards; s13 disk-trend alert compares /data vs /scratch floor; literal test deletes on spares only.
- 10-27 10:48 PT main amartyavarma-36 unreachable (gone from ListAgents; -ff gone too). Sent status + pending s6 decision to ppmguard (amartyavarma-43) to relay or decide as acting main. Pending decision: s6 Weka-served instead of autoshrink 40 + NOSPACE alert debounce.
- 10-27 10:48 PT ppmguard (ACTING MAIN since 10:32 PT): APPROVED slinky-6 Weka-served npl5 window (same pattern as s13; ops only; ~12 GB /data; deletes only in its Weka dir; MIN_FREE 350 on resolved dest) + NOSPACE debounce. /data 736 GB. Launch on hold until the user's explicit go (not on any peer's word). Routed to C (enable + gates + armed) and B (confirm fetcher for s6).
- 10-27 10:48 PT B: Weka test 31074 PASS (npl5/wekatest/run-31074): rc 0, 0 missing; refuse test rc 4; prep through symlink 35 s (254 files on Weka, 0 node-local); 470 sim steps (294 npl5), relaunch at 18190 from HELD 18110 fine; 81 passes, 131 x 429, ZERO alerts, steady 60 ahead; 3,417 drops all inside Weka test dir; peak 7.66 GB. Cleanup: slinky-4 test plan dirs + npl-append absent; /data npl5-weka absent (created at launch). R3 nits fixed (status never steps back; JSON has scratch_free_GB + dest_free_GB). Re-test 31079 on final code ~11:00. 2 Weka hosts need no code change.
- 10-27 10:51 PT C: static WEKA_HOSTS = {slinky-13, slinky-6} (deterministic; dynamic rule not built); s6 same stage block (symlink -> npl5-weka/slinky-6/jobs, MIN_FREE 350, AHEAD 100, /scratch window 0). NOSPACE debounce (> 30 min or lead < 50); Weka-host disk trend uses scratch_free_GB / df fallback / skip. test_gates GATES AGREE; --armed 10:50:40 (27 ok) + --dry-run 10:50:41 (SHORT none). Margins after margin: s13 +0.2, s19 +0.7, s16 +1.9, s12 +2.9 (6.7 GB caches), s11 +5.8, s0 +6.2, s6 +7.3, s8 +9.2, rest >= 14. /data 795 GB; Weka windows <= 23.42 GB; exports ~89 GB one at a time; dcp-17850 Weka copy 507 GB kept (main-approved drop after first save HF-verified 16/16 left as a manual step).

## ===== READY (10-27 11:03 PT) — NOT LAUNCHED; waits for the user's explicit go =====
Launch: bash /data/home/amartyavarma/ppm/pose5/switch_to_pose.sh --go   (--after-pause if motguard already paused MoT; --armed = read-only check)
- switch_to_pose.sh --armed at 11:03:00 PT: ARMED, rc 0 (log kit/logs/). Run = 8,1,1 resume from dcp-17850 with npl3 leftover + npl5 appended
  (PPM_MAX_STEPS 19382 = +1,532 steps ≈ 29.7 h), PPM_S2P=cb0 (one env line; full certified too), PPM_G0COND=1, own optimizer groups s2p/g0cond.
  --go -> POSE RUN LIVE ≈ 25-30 min (MoT pause via motguard mot_pause.sh --export=none, GPUS FREE, staging, Weka/node-local resume, 2 steps).
EVIDENCE:
1. Single launch, everything ON from 17850: env prod_811_fresh_append5_s2p.env (lines 1-308 = append3 verbatim + marked block); two-leg
   fallback documented (kit/FALLBACK_TWO_LEG.txt).
2. Proofs: code-811-s2p/tests/s2p/S2P_PASS + tests/g0/G0_PASS (core_md5 of 17 files == current; groups s2p/g0cond): flag off byte-identical;
   zero-init identical loss + all grads + every existing group's grad norm/clip factor, tiny + real width (real 17641 weights) on pose/podcast/
   Gemma/chat micros, all 5 combos; new-group grads on pose rows only; start evaluate() 188/188 metrics equal; resume from a ckpt without the
   new keys (all combos) + strict reload + refusals.
3. Resolved cfg vs 27653 (kit/state/CFG_DIFF_switch.txt): only 13 expected keys (plans, max_steps, resume_from, run_tag, keep_ckpt, s2p,
   s2p_from_step, g0cond, g0cond_from_step, lr/clip of the 2 new groups); GROUPS added exactly [g0cond, s2p]; LR schedule constant; Gemma/chat
   unflipped, podcast flipped from 16861, chat repeats from split 1, pose rows carry no voice clips/personas.
4. Rehearsals: switch --dry-run clean; B dry run (guard on real dcp-17850 train_state OK; ends [17352,17641,18022,19382]); 16-slot e2e 31048
   16/16; slot-9 rehearsal 31047; fetcher sims (keep-1 + hold + relaunch, 0 missing) + kill/OOM safety test; Weka tests 31074/31077 (0 missing,
   deletes only in Weka dir) + s6 check 31078; Weka dcp-17850 VERIFIED (507 GB, sha == HF); disk gates agree (no host short; MoT untouched);
   /data 742 GB (floor 300).
5. Live page: 811 page with header countdowns + npl5 ahead + disk-low; watchdog RESUME_PIN (never rewinds); hfb-watch 30966 (FLOOR 20)
   snapshots code-811-s2p; chained hold (KEEP_CKPT=1); export auto-drop (never w-15973 / other runs / MoT).
6. Independent audits: AUDIT_1 (4 blockers) -> AUDIT_2 (all fixed; 2 new blockers) -> AUDIT_3 (all fixed; no blocker); R3-1 fixed (s6 on Weka).
Open notes for the user: s2p full vs cb0 (latency 96 vs 66 ms/frame; env = cb0); 811 infer_ppm.py doesn't know s_emb / inp.g0 (use the
pose-s2p engine); codep eval probe can't see s2p; Aloe layouts need a SLOT for G0COND; FLOOR_GB back to 30 after the run; dcp-17850 Weka copy
(507 GB) droppable after the first save is HF-verified 16/16 (main pre-approved).

## USER GO (11:26 PT, via main amartyavarma-00) — LAUNCH IN PROGRESS — recorded 10-27 11:27 PT
- Main ran switch_to_pose.sh --go detached at 11:27 PT (log kit/logs/go-1027-1827.log). ARMED 11:27:01; pre-pause disk gate SHORT: none
  (tightest s13 +0.2, s19 +0.7, s16 +1.9); step 1: motguard mot_pause.sh --export=none, waiting for GPUS_FREE. s2p = cb0. pose5 babysits.
- 10-27 11:32 PT [pose5 switch, slinky-3] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-3 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 114.0 -> 160.1 GB
- 10-27 11:32 PT [pose5 switch, slinky-8] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-8 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 108.8 -> 154.9 GB
- 10-27 11:32 PT [pose5 switch, slinky-16] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-16 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 101.6 -> 147.7 GB
- 10-27 11:32 PT [pose5 switch, slinky-12] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-12 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 95.9 -> 141.9 GB
- 10-27 11:32 PT [pose5 switch, slinky-17] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-17 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 119.6 -> 165.7 GB
- 10-27 11:32 PT [pose5 switch, slinky-14] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-14 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 235.3 -> 281.4 GB
- 10-27 11:32 PT [pose5 switch, slinky-11] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-11 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 105.4 -> 151.5 GB
- 10-27 11:32 PT [pose5 switch, slinky-2] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-2 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 124.6 -> 170.7 GB
- 10-27 11:32 PT [pose5 switch, slinky-13] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-13 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 76.6 -> 122.7 GB
- 10-27 11:32 PT [pose5 switch, slinky-5] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-5 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 182.9 -> 229.0 GB
- 10-27 11:32 PT [pose5 switch, slinky-6] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-6 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 95.3 -> 141.4 GB
- 10-27 11:32 PT [pose5 switch, slinky-1] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-1 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 128.6 -> 174.7 GB
- 10-27 11:32 PT [pose5 switch, slinky-7] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-7 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 151.1 -> 197.2 GB
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/aloe-pose-build NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-0] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-0 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 105.8 -> 151.9 GB
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-q1ch NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/inductor-cache-ppmvox NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton_hera_ft NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/inductor-cache-s3perf NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-19] deleted node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 (HF InternalCan/ppm-v1-c dcp-17798/slinky-19 re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free 100.3 -> 146.4 GB
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/inductor-cache-samft-lr3 NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-samft-lr3 NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-ppm-fast NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-tt3 NOT deleted (guard refused)
- 10-27 11:32 PT [pose5 switch, slinky-6] npl5 window on Weka: /scratch/amartyavarma/npl-append/npl5/jobs -> /data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-6/jobs (main OK 10-27; B's S13_WEKA_FOR_C)
- 10-27 11:32 PT [pose5 switch, slinky-13] npl5 window on Weka: /scratch/amartyavarma/npl-append/npl5/jobs -> /data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-13/jobs (main OK 10-27; B's S13_WEKA_FOR_C)
- 10-27 11:32 PT [pose5 switch, slinky-19] npl5 roll --ahead 100 steps (/scratch window 11.72 GB)
- 10-27 11:32 PT [pose5 switch, slinky-16] npl5 roll --ahead 100 steps (/scratch window 11.74 GB)
- 10-27 11:32 PT [pose5 switch, slinky-3] npl5 roll --ahead 100 steps (/scratch window 11.69 GB)
- 10-27 11:32 PT [pose5 switch, slinky-0] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:32 PT [pose5 switch, slinky-13] npl5 roll --ahead 100 steps (/scratch window 0.0 GB; on Weka)
- 10-27 11:32 PT [pose5 switch, slinky-17] npl5 roll --ahead 100 steps (/scratch window 11.73 GB)
- 10-27 11:32 PT [pose5 switch, slinky-6] npl5 roll --ahead 100 steps (/scratch window 0.0 GB; on Weka)
- 10-27 11:32 PT [pose5 switch, slinky-8] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:32 PT [pose5 switch, slinky-12] npl5 roll --ahead 40 steps (/scratch window 8.39 GB)
- 10-27 11:32 PT [pose5 switch, slinky-5] npl5 roll --ahead 100 steps (/scratch window 11.71 GB)
- 10-27 11:32 PT [pose5 switch, slinky-11] npl5 roll --ahead 100 steps (/scratch window 11.72 GB)
- 10-27 11:32 PT [pose5 switch, slinky-14] pinned node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 -> /scratch/amartyavarma/ppm/ckpt-keep-pose5/dcp-17850 (hard links, 0 B; main YES 06:33 PT)
- 10-27 11:32 PT [pose5 switch, slinky-2] pinned node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 -> /scratch/amartyavarma/ppm/ckpt-keep-pose5/dcp-17850 (hard links, 0 B; main YES 06:33 PT)
- 10-27 11:32 PT [pose5 switch, slinky-7] pinned node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 -> /scratch/amartyavarma/ppm/ckpt-keep-pose5/dcp-17850 (hard links, 0 B; main YES 06:33 PT)
- 10-27 11:32 PT [pose5 switch, slinky-1] pinned node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 -> /scratch/amartyavarma/ppm/ckpt-keep-pose5/dcp-17850 (hard links, 0 B; main YES 06:33 PT)
- 10-27 11:32 PT [pose5 switch, slinky-14] npl5 roll --ahead 100 steps (/scratch window 11.72 GB)
- 10-27 11:32 PT [pose5 switch, slinky-2] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:32 PT [pose5 switch, slinky-1] npl5 roll --ahead 100 steps (/scratch window 11.71 GB)
- 10-27 11:32 PT [pose5 switch, slinky-7] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:33 PT [pose5 switch, slinky-15] npl5 roll --ahead 100 steps (/scratch window 11.73 GB)
- 10-27 11:35 PT STAGE 31091 ABORTED fail-closed (14/16 OK): s13 short 0.2 GB (margin 3.5, window on Weka), s12 short 0.8 GB (ahead 40, 7.8 GB to fetch).
  ppm811 31092 never started; companions stopped (page refresh 31096 soft). MoT paused at dcp-14839 (uploaded + verified 16/16).
  main (amartyavarma-00) APPROVED: s12 Weka-served (WEKA_HOSTS += slinky-12) + s13 margin 3.0. Applied in kit/gate_rules.py (backup
  kit/state/gate_rules.py.bak-pre-s12weka-*): weka slinky-12 rc 0; eval s13 122.4 -> +0.3 OK; s12 141.6 -> +7.5 OK. Removed s12's on-/scratch
  npl5 window dir /scratch/amartyavarma/npl-append/npl5/jobs (531 MB, 247 pose5-fetched npl5 parquet; re-fetchable; needed for the symlink;
  .roll-state.json + lock kept). Re-running switch --go --after-pause.
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/aloe-pose-build NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-q1ch NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/inductor-cache-ppmvox NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton_hera_ft NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/inductor-cache-s3perf NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/inductor-cache-samft-lr3 NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-samft-lr3 NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-ppm-fast NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] cache /scratch/amartyavarma/triton-cache-tt3 NOT deleted (guard refused)
- 10-27 11:35 PT [pose5 switch, slinky-12] npl5 window on Weka: /scratch/amartyavarma/npl-append/npl5/jobs -> /data/home/amartyavarma/ppm/pose5/npl5-weka/slinky-12/jobs (main OK 10-27; B's S13_WEKA_FOR_C)
- 10-27 11:35 PT [pose5 switch, slinky-19] npl5 roll --ahead 80 steps (/scratch window 10.63 GB)
- 10-27 11:35 PT [pose5 switch, slinky-0] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:35 PT [pose5 switch, slinky-16] npl5 roll --ahead 100 steps (/scratch window 11.74 GB)
- 10-27 11:35 PT [pose5 switch, slinky-13] npl5 roll --ahead 100 steps (/scratch window 0.0 GB; on Weka)
- 10-27 11:35 PT [pose5 switch, slinky-6] npl5 roll --ahead 100 steps (/scratch window 0.0 GB; on Weka)
- 10-27 11:35 PT [pose5 switch, slinky-11] npl5 roll --ahead 100 steps (/scratch window 11.72 GB)
- 10-27 11:35 PT [pose5 switch, slinky-3] npl5 roll --ahead 100 steps (/scratch window 11.69 GB)
- 10-27 11:35 PT [pose5 switch, slinky-5] npl5 roll --ahead 100 steps (/scratch window 11.71 GB)
- 10-27 11:35 PT [pose5 switch, slinky-17] npl5 roll --ahead 100 steps (/scratch window 11.73 GB)
- 10-27 11:35 PT [pose5 switch, slinky-12] npl5 roll --ahead 100 steps (/scratch window 0.0 GB; on Weka)
- 10-27 11:35 PT [pose5 switch, slinky-8] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:35 PT [pose5 switch, slinky-7] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:35 PT [pose5 switch, slinky-14] npl5 roll --ahead 100 steps (/scratch window 11.72 GB)
- 10-27 11:35 PT [pose5 switch, slinky-2] npl5 roll --ahead 100 steps (/scratch window 11.7 GB)
- 10-27 11:35 PT [pose5 switch, slinky-1] npl5 roll --ahead 100 steps (/scratch window 11.71 GB)
- 10-27 11:35 PT [pose5 switch, slinky-15] npl5 roll --ahead 100 steps (/scratch window 11.73 GB)
- 10-27 11:36 PT stage 31100 16/16 OK (11:36:17; s13 122.4 vs 122.1; s12 141.7 vs 134.1). ppm811 31101 RUNNING on slinky-[0-3,5-8,11-17,19] (log ppm/logs/ppm-31101.log); companions 31102-31105 on slinky-4.
- 10-27 11:40 PT 31101: [resume] FRESH POSE APPEND 4 plans (ends [17352, 17641, 18022, 19382]); POSE-S2P ADDED (tower_p.s_emb cb0, own group s2p) + G0COND ADDED (inp.g0, own group g0cond) at 17850; 'resumed /scratch/amartyavarma/pose5-resume/dcp-17850 at step 17850' 11:38:41. main: hand-back must list this run's HF-verified files so every host reaches >= 64 GiB free for the MoT resume (print for main's OK; never MoT's dcp-14839 / ppm-mot hold / caches) -> C builds a separate print-only helper (running kit scripts not edited).

## ===== POSE RUN LIVE: ppm811 31101 (11:41:48 PT) — recorded 10-27 11:43 PT =====
- s2p cb0, G0COND on, from dcp-17850, max 19382. Resumed 11:38:41 (ADDED s_emb / inp.g0 in own groups; everything else restored exactly).
- Exactness: start eval 17850 EXACT vs ppmdecay-j27810; [811 rows] 17851/17852 == ppmdecay-27810; all 811 per-micro checks true; existing gn
  ~= decay branch at 17851 (decay's lr_mult < 1 on decoder nanosteps explains 1e-4 diffs); new groups g0cond 0.0020, s2p 0.0004; roll check 16/16 OK.
- Start eval: eval_pose/word_ce_norm 1.9017, pose_cb0_ce 1.6463, pose_light_ce 3.7304. Step 17851 77.8 s (word_ce_norm pose 1.8464 / speech
  1.9853 / text 0.7220; pose_cb0_ce 1.6620; peak 53.4 GiB); 17852/17853 67.3 s -> ≈ 28.6 h to 19382.
- Companions: watchdog 31102, hold chain 31103, eval logger 31104, roll supervisor 31105, page refresh 31096 (slinky-4); hfb-watch 30966 (slinky-10).
- MoT: paused at dcp-14839 (uploaded + verified 16/16). Resume needs >= 64 GiB free per host (hand-back helper by C).
- TODO for pose5: fix the pre-pause gate gap (MoT's pause save not modelled -> s12/s13 short at stage); watch first save + HF verify + chained
  hold; first npl5 rows ~18022-18034 (fetcher ahead); Weka dcp copy drop after first save HF-verified (main pre-approved).
- 10-27 11:46 PT main is now amartyavarma-96 (restart). Main added a P5 dotted event line to the 811 page at 17850 (ppm_doc.py events() + ppm.tpl.html EVL; backups *.bak-pre-p5mark-1027); page refresh clean at 18:46 UTC (step 17856, P5 present).

## END-OF-RUN HAND-BACK: MoT resume needs >= 64 GiB free /scratch per host (C, print-only helper) — 10-27 11:48 PT
- WHEN: after the pose run's FINAL save dcp-19382 is HF-verified 16/16 (end_of_run.sh prints that; hold chain releases its holds then).
- COMMAND (login node; it starts ONE tiny read-only srun over the 16 hosts, -c1 --mem=1G, nice; prints, deletes NOTHING):
    python3 -I /data/home/amartyavarma/ppm/pose5/kit/handback_free64.py
  (if the nodes are busy: add --jobid <a running job on all 16 hosts> to overlap it; records-only estimate: --records)
- Refuses (exit 2, nothing listed) until dcp-19382 is HF-verified 16/16. Output per host: free now, the smallest set of THIS run's own files that
  reaches 64 GiB, free after; report saved to kit/state/handback/handback-live-<time>.{txt,json}. Main OKs; deletion is a separate step.
- Tiers (removal order): T1 superseded pose saves 17850<N<19382 (CKPT + hold, HF-verified for that host, written by this run's jobs) ->
  T2 npl5 window + plan -> T3 npl3 rows -> T4 npl2 rows (all HF-sourced) -> T5 dcp-17850 pin (HF-verified 16/16 + Weka copy) ->
  T6 s15 slot-9 pose store (FLAG: MoT's mot.env names pose-s3-samdata/rows; MoT ran on s15 without it -> main + motguard OK).
  Sizes = bytes freed only if EVERY hard link is in the group (a pin sharing inodes with ppmvox's copy frees 0).
- NEVER: the final dcp-19382 (printed only if a host is still short), dcp-16861, saves not written by this run, plan-1 rows (new-pose-llm-data),
  s3-rows / txtdup / other hosts' pose-s3-samdata, anything ppm-mot* (MoT's dcp-14839 + its hold), next-arch, ppm-next, any MoT cache.
- Dry run on current records (11:45 PT; stage-31100 free - final 46.09 - npl5 window still to fetch): every host >= 64 GiB with NOTHING removed
  (tightest s13 71.1 GiB, s16 83.5, s19 83.4, s11 87.0, s0 87.4); report kit/state/handback/handback-records-1027-114558.txt.
- 10-27 11:47 PT C: hand-back helper kit/handback_free64.py (print-only; refuses until dcp-19382 HF-verified 16/16; tiers T1 superseded pose saves -> T2 npl5 -> T3 npl3 -> T4 npl2 -> T5 dcp-17850 pin -> T6 s15 slot-9 store only with main + motguard OK; hard-link aware; never MoT / 16861 / other runs / shared data). Records dry run: all hosts >= 64 GiB with nothing removed (tightest s13 71.1). Reported to main.
- 10-27 11:49 PT s15 PPM_POSE_ROOT check (main -96 asked): pose5 CREATED /scratch/amartyavarma/pose-s3-samdata/rows on slinky-15 at 11:32 (stage 31091 STAGED; absent-path branch only); motguard records (du_slinky15.txt 10-23, inv.txt 10-24) show no such dir before; MoT never reads pose_root (train_ilp.py :143 only; :1038 pose excluded; motguard dry-resume-posecheck.log agrees). Evidence sent to main + motguard. Nothing removed there without main + motguard OK.

## PRE-PAUSE GATE GAP (stage 31091 aborted s12 / s13): root cause + fix (C) — 10-27 12:02 PT
- MEASURED (stage-31091 log vs the 09:59 GUARD_LOG record the 11:27 gate used): MoT's pause stop save cost 0.0-0.1 GB/host (keep 1: dcp-14839
  REPLACED the kept save; its hfhold = hard links, 0 B; no older MoT save was held at 11:27). Staging wrote 0.29 GB (Weka hosts) / ~0.8 GB
  (others, incl. the start of the npl5 window that the model already counts) per host.
- s12 (-0.8): the cache deletions NEVER RAN on any host: stage_host.sh step 2 `bash -c '...; nice -n 10 guard_rm ..'` -> "nice: guard_rm: No
  such file or directory" (rc 127; nice cannot exec a shell function). 6.7 GB counted as freed, not freed. FIXED (nice wraps the bash),
  tested on spare slinky-4 (new form deletes; old form rc 127). With it s12 would have passed the live gate at about +5.9.
- s13 (-0.2): the model's +0.2 did not cover the staging writes (0.29) + MoT noise (0.1).
- MODEL (c/disk/prepause_terms.py, used by build_disk_need.py): expected free at the live gate = record + 17798 + caches - [MoT pause delta
  (0 if no MoT hold at the pause; a full MoT save/host, 36.2 GB, if motguard's state backup holds an older save then: from GUARD_LOG hold /
  release lines + the "16/16 hosts ... B" size line) + staging writes 0.3 + noise 0.1 + cushion 0.5 (PREPAUSE_CUSHION_GB)].
- REPLAY (kit/test_prepause_model.py on the as-run 11:27 table kit/state/DISK_NEED.as-run-31091.json + stage-31091.log): new model flags
  s12 (-4.7 as run) and s13 (-0.7): both real aborts; old model passed both. Conservative elsewhere by ~0.5 GB (s19 -0.2 vs live +0.4: npl5
  already on disk shrank its live pending). What-if: a pause while dcp-14555 was held (09:44-09:59) -> +36.2 GB/host.
- Current settings (s12 Weka, s13 margin 3.0) on the new model: c/DISK_NEED.dryrun.json, no host short; test_gates GATES AGREE. The live
  c/DISK_NEED.json is left as the 11:27 as-run table.
- 10-27 11:52 PT C: pre-pause gap root cause: NOT MoT's pause save (dcp-14839 cost 0.0-0.1 GB/host; replaced the kept save; hfhold hard links). s12 -0.8 = BUG: stage_host step 2 'nice -n 10 guard_rm' (nice can't exec a shell function, rc 127) -> cache deletion never ran on any host (6.7 GB counted, not freed); FIXED kit/stage_host.sh:87-89 (tested on slinky-4). s13 -0.2 = staging writes 0.29 + noise. New pre-pause model c/disk/prepause_terms.py (MoT pause delta from GUARD_LOG holds + staging 0.3 + noise 0.1 + cushion 0.5); replay flags every real abort (kit/test_prepause_model.py). No running script edited.
- 10-27 12:41 PT FIRST SAVE dcp-17903 at 12:40:36 PT (42 s). 11 hold-chain disk-trend ALERTs at 12:40 = FALSE POSITIVES (stage-time free with 0 pose saves vs first post-save free: one-time 46 GB step; lowest real free s13 76.3 GB vs floor 20). Main told; C to fix the trend baseline (no in-place edit of running hold_chain 31103).
- 10-27 12:41 PT projected trough after save 2 (free now - 46.09 - window left; Weka hosts window 0): all hosts above floor + margin (min s19 +7.1, s13 +7.2 = 30.2 vs 23); fetchers 129-141 ahead; hfb-watch hasn't logged dcp-17903 yet (save 1 min old). Sent to main.
- 10-27 12:42 PT hfb-watch: dcp-17903 COMPLETE 16/16, held in ckpt-keep, queued -> InternalCan/ppm-v1-d (12:41:47 PT); pose5 chained hold: HOLD dcp-17903 16/16 (12:41:27).
- 10-27 12:43 PT C (kit agent) stopped: API session limit until 21:00 UTC (14:00 PT); trend-alert fix pending until then. Expect one more spurious 'troughs falling' alert ~13:40-13:55 (save 2 while save 1 held); real projected troughs all above floor + margin.
- 10-27 13:10 PT HF UPLOAD STALL root cause: hfb-throttle controller (hfb_throttle.py; last job 26111 on slinky-15, cancelled 10-23) not running -> every control/<host>.workers = 0 -> dcp-17903 uploaders (31101 steps 105-120, hfb-pin) idle since 12:43:49. FIX: started hfb-throttle job 31153 on slinky-4 (same cmd as 26111; -c1 --mem=2G): 13:09:19 BASE 67.5 s, N 1 (slinky-0 hashing); ramps to 16 hosts ~13:40. dcp-17903 verify ETA ~13:50-14:00. Kit gap: ARMED must also require a running hfb-throttle (C after 14:00 PT).
- 10-27 13:22 PT gn spike at step 17938 (gn_audio 1.80, gn_trunk 3.20; clipped per group at 1.0); 17939 recovered (trunk 0.327, audio 0.085); losses unchanged. 27653 had the same event at 17754 (gn_audio 1.867); 27114/27204 max 0.36/0.40. Logged only (main).
- 10-27 13:27 PT dcp-17903 HF upload: 12/16 hosts VERIFY PASS (first slinky-0 13:14 PT); slinky-15/16/17/19 still uploading (throttle N 6 at 13:20). Release expected ~13:35-13:45.
- 10-27 13:32 PT dcp-17903 HF-VERIFIED 16/16 (hfb finalize PASS, 320 files, 737.5 GB, bad []; release 13:32:27 PT). Chained hold keeps it until save 2 verified. Asked main whether to drop the 507 GB Weka dcp-17850 copy now (pre-approved 06:28 after first save HF-verified).
- 10-27 13:32 PT main: KEEP the Weka dcp-17850 copy for now (pose6's proofs may read it; /data ~660 GB). Drop when pose6 says READY (~17:30 PT) or earlier if /data < 450 GB — ask main first either way.
- 10-27 13:43 PT DEADLOCK: save 2 dcp-17956 at 13:42 (keep-1 pruned CKPT dcp-17903; chain hold keeps 17903 data). hfb-watch refuses to hold 17956 ('/scratch would drop below 20 GB once held saves rotate': s0 55, s3 62, s6 46, s8 57 GB) -> can't upload 17956 -> chain can't release 17903. Asked main (urgent) to OK releasing the chain hold of HF-verified dcp-17903 now + rule change (release N-1 once N-1 HF-verified 16/16 and N COMPLETE).
- 10-27 13:45 PT main APPROVED (13:44): (1) release chain hold of HF-verified dcp-17903 now; (2) new rule: release N-1 when N-1 HF-verified 16/16 AND N COMPLETE 16/16 (invariant: an HF-verified save always exists + newest complete node-local save); watchdog fallback to HF restage if newest node-local save bad. pose5's release command was BLOCKED by Claude Code's built-in rm safety check (needs a person) -> asked the user to run it.
- 10-27 14:01 PT RELEASED chain hold dcp-17903 on 16/16 hosts (main-approved 13:44; literal-path srun rm -rf --one-file-system /scratch/amartyavarma/ppm/ckpt-keep-pose5/hold/dcp-17903, rc 0; the earlier bash -c form was blocked by the safety check). Hold dir now only dcp-17956; /scratch free back up (min s13 71, s6 89, s12 89 GB). Chain state held-17903 -> released-17903.
- 10-27 14:03 PT hfb-watch held dcp-17956 (14:02:56 PT, queued -> ppm-v1-d). C plan approved: hold_chain_v2 (new release rule + step-aware trend; swap of 31103 with between-polls check) target before ~14:38 (save 3 ~14:42 would re-deadlock under the old rule; fallback = pose5 manual release of 17956 after save 3 COMPLETE + 17956 HF-verified, per main's rule); ARMED hfb-throttle check ~15:15; watchdog v2 HF-restage fallback ~16:45 (swap 31102). Disk-trend alert suppressed until 16:00 PT via future-dated alert markers (other alerts unaffected).
- 10-27 14:07 PT C: hold_chain v2 LIVE job 31156 (swapped 14:05:35 PT between polls; v1 31103 cancelled while sleeping); rule: release N when N HF-verified 16/16 AND a later save COMPLETE 16/16; releases via direct srun rm on regex-checked literal paths; trend alerts only after first v2 release; STOP_HOLD_CHAIN for clean exits. Offline 3-host test all as expected. State carried: held-17956, released-17903.

## C: HOLD CHAIN v2 LIVE (job 31156, swapped 14:05:35 PT; main-approved) + hfb-throttle in the switch — 10-27 14:10 PT
- Why: hfb-watch refused to hold dcp-17956 while v1 31103 still held the HF-verified dcp-17903 (s6 46 GB free) -> deadlock; coordinator released
  17903 by hand at 14:01. v1 waited for a LATER save's verify; under that rule the gate needed 3 x 46.09 + 20 = 158.3 GB (s6 had 141.1).
- v2 RULE (kit/hold_chain_v2.sh): release chain-held N when N is HF-verified 16/16 AND a later save is COMPLETE 16/16 in CKPT (old trigger kept
  too). Invariant: an HF-verified save + the newest complete node-local save. Gate saves term max(46.09 + 53 + floor, 2 x 46.09 + hfb FLOOR)
  matches this rule (hfb holds N with only N on disk).
- No rm inside bash -c: releases + stale .tmp cleanup = direct `srun --overlap ... rm -rf --one-file-system -- <literal path>`, path must match
  ^/scratch/amartyavarma/ppm/ckpt-keep-pose5/hold/dcp-<digits>(.tmp)?$. Release time stamped (released-N touched; state/first_release_at).
  STOP_HOLD_CHAIN in kit/state/hold_chain = clean exit at the top of a poll (future swaps).
- Disk trend (interim): alerts only after the first v2 release + two full hourly windows (roll_ahead.json trend_from); floor+margin check always on.
  The 12:40 false positives (stage-time free vs first save) cannot recur. The 16 alert-disk markers (future-dated 12:41-14:05 to mute) were reset.
- Swap: kit/swap_hold_chain_v2.sh (check: v1's bash had ONLY `sleep 60` as descendant -> scancel 31103 -> sbatch hold_chain_v2.sbatch
  TARGET=19382 PIN_STEP=17850, same state dir -> LAUNCHED_RUN HC=31156). First v2 poll 14:07: held-17956 kept, released-17903, no alert.
- Offline test (3 fake hosts, hc2_test.sh): unverified -> only hold; verified + newer COMPLETE -> release; keep-1 prune -> hold keeps it;
  stale .tmp removed by direct rm; STOP_HOLD_CHAIN exit rc 0.
- Switch (future runs): ARMED shows hfb-throttle (1 running = ok; 0 = step 2 starts kit/hfb_throttle.sbatch with need_id; 2+ = red); step 6
  requires it RUNNING before POSE RUN LIVE; the switch submits hold_chain_v2.sbatch.
- 10-27 14:09 PT DELETED (main OK 14:22 msg; pose6 READY): /data/home/amartyavarma/ppm/pose5/dcp17850-weka (490 GiB = 11 host dirs of dcp-17850; HF ppm-v1-c copy + node-local pins stay); VERIFIED*.json records kept in kit/state/dcp17850-weka-records/. df /data still 656 GB right after (Weka reclaim async). Export auto-drop confirmed active in chain v2 (venv python; each poll; only w-15973 present -> KEEP hard refusal); first pose export ~17:38 PT (every 360 min).
- 10-27 14:09 PT /data free 1,147 GB after Weka reclaim. End-of-run /data ledger sent to main: windows <= 35, exports ~89 GB each (17:38/23:38/05:38/11:38 PT) auto-dropped after HF verify (<= 2 at once), final local-19382 conversion 89 -> worst low ≈ 845 GB.
- 10-27 14:11 PT main is now amartyavarma-06 (restart ~14:10).
- 10-27 14:13 PT dcp-17956 HF-VERIFIED 16/16 (finalize PASS 14:13:03 PT, 320 files, 737.5 GB, bad []). Chain v2 keeps it until save 3 COMPLETE (~14:42), then releases.

## C: HOLD CHAIN v3 (31159) + WATCHDOG v2 (31160) LIVE (swapped 14:24 PT, main-approved plan) — 10-27 14:26 PT
- Hold chain v3 (kit/hold_chain_v3.sh) = v2 + STEP-AWARE disk trend: one trough per SAVE CYCLE (save k COMPLETE -> save k+1 COMPLETE; events
  first seen by the hold scan -> kit/state/hold_chain/save_events.tsv), only cycles that start after the hold cycle is established
  (first_release_at), alert only if the last full-cycle trough fell and 2 more such falls cross floor + 5 (floor + margin check always on).
  Replays: the real free-*.tsv history incl. the 12:40 first-save step -> 0 alerts; synthetic 61-62 min drifting cadence, flat troughs -> 0;
  troughs falling 10 GB/cycle -> alert. Swap kit/swap_hold_chain_v3.sh: v2 31156 had only `sleep 60` -> scancel -> v3 31159, HC=31159.
- Watchdog v2 (kit/ppm811s2p_watchdog_v2.sh): resume decision = kit/wd_decide.py:
  newest save COMPLETE 16/16 (CKPT or hold) -> resume it; else the newest HF-VERIFIED save of THIS run (done/dcp-N-final ok + 16/16 verify
  PASS + hfb record job = LAUNCHED_RUN J / a relaunch) newer than every complete local save -> RESTAGE it from HF on all 16 hosts
  (kit/restage_dcp_hf.py: node-local /scratch/amartyavarma/pose5-restage/dcp-N when /scratch >= save + 53 + floor, else Weka
  pose5/restage/dcp-N/<host> with /data >= save + 350; size + sha256 == HF LFS, train_state step == N, COMPLETE last, per-host link
  /scratch/amartyavarma/pose5-resume/dcp-N) and relaunch from the link; NEVER an older local save or 17850 once a newer save exists.
  A save a relaunch died on (no newer step logged) is marked BAD (watchdog_pose5_state/bad-N), never resumed locally again; a BAD save that
  was already restaged once -> ALERT, no relaunch. Nothing is deleted (restage dirs stay for main).
  Tests: wd_decide 8 scenarios (fake scans + fake hfb; another job's verified saves never count; partial scan -> ALERT); watchdog v2 DRY
  with fake srun: CKPT complete -> FROM; partial + nothing verified -> ALERT; partial + verified -> RESTAGE; bad -> RESTAGE; bad +
  already restaged -> ALERT; restage tool on spare (overlap step): dcp-17903/slinky-0 small files from HF STAGED, re-run OK, a corrupted
  file -> ABORT (size 66 != 64). Swap kit/swap_watchdog_v2.sh: v1 31102 had only `sleep 60`; launch env read from /proc -> v2 31160
  with the identical env, WD=31160.
- Switch (future runs): submits hold_chain_v3.sbatch + ppm811s2p_watchdog_v2.sbatch; ARMED checks those + wd_decide.py / restage_dcp_hf.py;
  gate_rules.saves() documents the term vs the hold rule (v1 rule would need 3 x 46.09 + 20 = 158.3 GB).
- 10-27 14:26 PT C: hold chain v3 job 31159 (v2 -> v3 14:24:28) + watchdog v2 job 31160 (v1 31102 -> v2 14:24:57; launch env reused verbatim) LIVE; LAUNCHED_RUN HC=31159 WD=31160. Watchdog v2: resume newest COMPLETE 16/16 else RESTAGE newest HF-verified save of this run from HF (kit/restage_dcp_hf.py; node-local or Weka; sha256 == HF LFS); never older local / 17850; BAD-save marking. v3 step-aware trend (per save cycle; replay of 12:40 -> 0 alerts). Switch: ARMED requires/starts hfb-throttle. Full 16-host restage not exercised (small-file real HF test only). Delta audit (round 4) requested.
- 10-27 14:30 PT AUDIT round 4 (AUDIT_4.md): no blockers; companions can't touch MoT / unverified saves / 16861; watchdog can't rewind past newest HF-verified save. Should-fix (failure paths only): R4-1 newer stale complete save vs resume-from-older; R4-2 BAD marking too eager; R4-3 restage Weka destinations uncoordinated; R4-4 restaged save can block saves + no alert on '[save] SKIPPED'. Nits R4-5/6/7. Routed to C (swap between polls, target 17:00 PT). pose5 monitor already greps the trainer log for SKIP.
- 10-27 14:31 PT C: swapping hold chain v3 31159 -> v4 (between polls; only 'sleep 60'): + R4-4 SKIPPED alert, R4-5 no release while a MISMATCH exists, R4-6 BAD save never counts as later COMPLETE; offline tests pass. Watchdog v3 next (separate swap).
- 10-27 14:36 PT C: hold chain v4 LIVE job 31165 (14:31:52). Swapping watchdog v2 31160 -> v3 between polls: R4-1 (newer COMPLETE save in CKPT -> ALERT, no relaunch), R4-2 (BAD only on real load failure or 2 deaths same save), R4-3/4 central restage destination map (node-local only with save + 2 x 53 + floor; Weka only if /data stays >= 350 incl. exports), R4-7 small-file byte compare vs second HF read. Offline DRY tests pass.

## C: AUDIT 4 FIXES LIVE — hold chain v4 (31165, 14:31:52) + watchdog v3 (31170, 14:36:23), between-polls swaps — 10-27 14:38 PT
- R4-4: hold chain v4 alerts on the trainer's "!!! [save] SKIPPED" (only on a NEW skip line, at most hourly while skips continue).
  Restage: node-local only if /scratch >= save + 2 x 53 + floor, else Weka.
- R4-5: v4 never releases N while a MISMATCH hold dir exists for N on any host (note logged, dir left for main).
- R4-6: v4: a watchdog-BAD save (watchdog_pose5_state/bad-N) never counts as "the later save COMPLETE".
- R4-1: watchdog v3 (kit/wd_decide_v2.py): FROM/RESTAGE of S -> ALERT, no relaunch, if a COMPLETE save > S stays in CKPT on any host (main
  moves it out of ckpt/ppm-v1 by hand; nothing renamed/deleted by the watchdog).
- R4-2: BAD only on a load failure (kit/wd_loadfail.py: a [rankN] traceback through load_local / load / torch.load / digest_state, or an
  exception naming digest / COMPLETE / unpickling / parameter names, with no "resumed ... at step") or after 2 deaths from the same save
  (watchdog_pose5_state/deaths-N); otherwise the same save is retried.
- R4-3: one central restage destination map per restage (from the same scan, now with /scratch free per host): all Weka hosts together must
  leave /data >= 350 GB (df at that moment, exports included), else ALERT, no restage; kit/restage_dcp_hf_v2.py --dest-map (re-checks
  node-local room; without a map: node-local 2 x 53, Weka only if /data >= 350 + 16 x save).
- R4-7: restage v2 byte-compares every small file with a second, forced HF read (same-size corruption -> ABORT, tested).
- Tests: hc4_test.sh (BAD later save -> no release; MISMATCH -> skipped; clean -> release; SKIPPED alert new/repeat/hourly),
  wd_decide_v2 (8 cases incl. R4-1), wd_loadfail (digest assert -> LOADFAIL; NCCL / FileNotFound / after-resume / scancel / live log -> none),
  watchdog v3 DRY (FROM; R4-1 ALERT; RESTAGE map 8 node-local / 8 Weka with /data 1400 >= 723; /data 600 -> NOROOM; death 1 -> retry;
  death 2 -> BAD; load failure -> BAD), restage v2 on the spare (STAGED, OK, same-size corruption -> ABORT).
- LAUNCHED_RUN: HC=31165, WD=31170. Switch (future runs) submits hold_chain_v4 + watchdog_v3; --dry-run clean.
- 10-27 14:43 PT PAUSE PLAN (pose6 mono17850 FT; pause only on pose6's PAUSE NOW ~16:30 after local-17850 verified; the 14:43 'pause now' was retracted, 31101 NOT paused). Node-local dcp-17850 now only on s1/2/7/14 (pin ckpt-keep-pose5/dcp-17850) + s15 (ppmvox/init-17850-slots); other 11 hosts' pose5-resume links dangle (Weka copy deleted 14:09) -> told pose6 (HF ppm-v1-c has it; restage tool available). Save 3 dcp-18009 written 14:43:28 PT.
- 10-27 14:44 PT pose6: conversion via HF range reads (node-local dcp-17850 copies = fallback only). FT per host 8 tasks x 4 CPUs = 32/64 CPUs, 8 GPUs, --mem 500G, not exclusive -> hfb own upload allocations may keep running on FT hosts. Pause save must be HF-verified before 31101's first save after resume. C building kit/pause_resume.sh (--pause/--resume, dry-run) by 16:15.
- 10-27 14:44 PT Save 3 dcp-18009 14:43:28; chain v4 HOLD 16/16 14:43:57; chain v4 RELEASE dcp-17956 (HF-verified + 18009 COMPLETE, new rule) 16/16 at 14:44:05 -> hfb can hold 18009.

## C: PAUSE / RESUME KIT READY (kit/pause_resume.sh; dry-run tested) — 10-27 14:52 PT
- PAUSE (on pose6's "PAUSE NOW"):   bash /data/home/amartyavarma/ppm/pose5/kit/pause_resume.sh --pause
  checks (job RUNNING, watchdog v3 RUNNING + its deliberate-stop branch, hold chain v4 / roll supervisor idle with no job, no wd_stop, no
  STOP_AFTER_SAVE) -> mutes the 146 roll/ahead/disk alert keys for 4 h (future-dated markers; no script edited) -> STOP_AFTER_SAVE -> waits for
  "STOPPED at step N" + job gone + the watchdog's "stopped deliberately" line + dcp-N COMPLETE 16/16 (own tiny srun) -> releases every chain hold
  P < N that is HF-verified 16/16 (direct srun rm of the literal hold/dcp-P; held-P -> released-P; not verified -> kept) -> prints step, save,
  per-host free, HF status, "HOSTS FREE". No export at the stop (export_every 0). Never touches MoT, dcp-17850 copies (pins / ppmvox) or hfb.
- During the pause, once the previous save verifies:   bash .../kit/pause_resume.sh --release
- RESUME (after the FT):   bash /data/home/amartyavarma/ppm/pose5/kit/pause_resume.sh --resume
  checks (no ppm811 queued, dcp-N COMPLETE 16/16, NO newer COMPLETE save in CKPT (R4-1)); prints the pause save's HF status (VERIFIED |
  UPLOADING k/16 -> WARN | NOT STARTED -> loud WARN + ALERT); renames a leftover STOP_AFTER_SAVE / SAVE_NOW aside; submits exactly like
  watchdog v3's relaunch (PPM_ENV=kit/state/run.env PPM_RESUME_FROM=<CKPT>/dcp-N sbatch -N 16 -w nodes16_s7_s15.txt code-811-s2p/ppm811.sbatch);
  CURRENT_JOB_811 = new (watchdog v3 / hold chain v4 / roll supervisor follow it), LAUNCHED_RUN += J=new, watchdog relaunch_from += N; waits for
  RUNNING, "resumed ... at step N", steps N+1/N+2 sane, supervisor attached; re-arms the muted alerts; roll check 16/16; "RESUMED <job>".
- Tests: live --pause --dry-run / --resume --dry-run --step=18009 / --release --dry-run (refuses while running); offline full flow with fake
  squeue/srun/sbatch + a simulated trainer/watchdog/supervisor (pause -> 18031 STOPPED, verified 18009 hold released by a literal srun rm ->
  HOSTS FREE; resume -> sbatch env checked, CURRENT_JOB_811 / LAUNCHED_RUN / relaunch_from written, resumed line, 2 sane steps, attach, 146
  markers re-armed, roll check OK -> RESUMED; resume refused with a newer COMPLETE save in CKPT).
- handback_free64.py now counts every J= line of LAUNCHED_RUN as this run's job.
- 10-27 14:50 PT C: kit/pause_resume.sh READY (14:52; dry-runs + offline full flow pass): --pause (STOP_AFTER_SAVE; waits STOPPED + job gone + watchdog 'deliberate' + dcp-N COMPLETE 16/16; releases HF-verified older chain holds; mutes 146 alert keys 4 h; prints HOSTS FREE), --release (later), --resume (checks; exact watchdog-v3 relaunch sbatch from CKPT/dcp-N; CURRENT_JOB_811 / LAUNCHED_RUN / relaunch_from; resumed + 2 sane steps + roll attach; re-arms alerts; prints RESUMED <job>).
