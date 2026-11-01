## B (npl5 = delivery 5): READY (10-27 09:42 PT)

Records: pose5/npl5/ (SUMMARY.md, TIMES.tsv, logs/, LEG1_SLOT9_HOOK.txt, bin/). Nothing touched the 16 hosts; no B data is left on slinky-10/18.

- Source: HF bucket InternalCan/pose-llm-data/new-pose-llm-data-delivery5.
  - 552,329 parquet + 552,329 release.json, 1.224 TB.
  - 0 overlap with plans 1-3. All releases accepted: 0 pin/sha mismatches.
- Scans: staged on the spares, raw deleted after each scan.
  - 56,945,936 segments, 224,662 h.
  - Products: npl5/scan (5.1 GB). User change #2 sidecars: npl5/scan/origwh (orig_h/orig_w, 0 nulls).
  - /data raw cap of 155 GB enforced in code (bin/stage_lane4.sh + bin/src_fetch_capped.py).
- Plans: npl5/plans/<key>/ppm-fresh-pose-npl5 (PLANS.json).
  - fresh_pose's own cmd_plan; global dedup kept 56,674,353.
  - 1,391,232 rows, 223,599 h. Canonical src_root /scratch/amartyavarma/npl-append/npl5/jobs.
  - Keys = plans 1-3 keys; slot 10 = key slinky-7; slot 9 = key slinky-9, played by slinky-15.
- Dry run, append at 17850: ends [17352, 17641, 18022, 19382] -> PPM_MAX_STEPS = 19382 (+1,532 steps after dcp-17850).
  - Every rank is in npl3 at 17850. npl5 starts at steps 18022-18034. Unused rows 5,237.
  - Flag: speech epoch 2 from 17872 (known).
  - Guards on the real dcp-17850 train_state OK, with code-811 AND with code-811-s2p + the final env (cb0, G0COND): append_at [17352, 17641, 17850]; 4/4 refusals.
  - fresh_pose check 16/16. 16-slot e2e with the real 4 plans per slot: 16/16, 24 micros each.
- Slot 9 (bin/leg1_slot9_hook.sh): plans 1-3 are off-host byte-identical rebuilds; npl3 window 5,434 files / 10.24 GB from HF delivery 3, staged on /data; digests == dcp-17850.
- Rolling fetcher (the hosts cannot hold 75 GB/host): bin/npl5_roll.py v2.
  - Runs as one --overlap step in the trainer job. Supervisor: bin/npl5_roll_super.sbatch on slinky-4.
  - Peak window over the run (back 110): a100 <= 11.74 GB, a80 <= 10.63, a60 <= 9.54, a40 <= 8.42 per host (npl5/roll/WINDOW-*.json).
  - Disk floor MIN_FREE_GB 66 (kit/state override). Markers: roll/<host>.{BEHIND <30, CRIT <20, STARVED, LOWMEM, NOSPACE}, roll/STEP_DIED.
  - Heartbeats: roll/<host>.json (30 s) and roll/SUPER.json.
  - Sim v2 + kill/OOM safety test v2: PASS on the final code.
- Launch-day B steps, inside C's kit/stage_host.sh, ~70 s per host in parallel (rehearsed as slinky-15):
  1. bash pose5/npl5/bin/leg1_slot9_hook.sh
  2. bash pose5/npl5/bin/npl5_prep.sh
  3. Right after the trainer sbatch: `sbatch --export=ALL,END=19382,AHEAD=100,LOW=30 pose5/npl5/bin/npl5_roll_super.sbatch` (C's gate writes kit/state/AHEAD-<host>).
