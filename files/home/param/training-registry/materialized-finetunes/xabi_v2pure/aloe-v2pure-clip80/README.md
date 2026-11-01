# aloe-v2pure-clip80: job 23051 re-run with ONE change, the system-prompt rule (no persona, voice clip on 80%)

PPM 32B monoplex Aloe pose fine-tune on Xabi's single-speaker pose data (release v2, "v2pure"), init PPM local-11488, 200 steps x 64 rows.
Reference run: `ppm/aloe-v2pure-ft` (job 23051, 10-17). User request 10-18: the new general rule "voice clip and persona drawn independently,
80% clip, 20% persona"; this monoplex run starts WITHOUT persona, so persona = 0% here.

## The one change
| | 23051 | this run |
|---|---|---|
| persona text | data cfg `persona_sides` "both" (the v2pure stores carry no persona: 'missing' on every row, so 0% in practice) | `persona_sides` "none" (no persona text at all) |
| voice clip drawn | `h64("ppm-voice\|<seed>\|<row_key>") % 3 == 0` (`voice_mod` 3) | `h64("aloe-v2pure-clip80-voice\|<seed>\|<row_key>") % 10 < 8` (own tag = independent draw) |
| the clip itself | `data_ppm.voice_clips` (a later clip of the same recording's second half, B's speaker) | unchanged |

seed = 20261011 (23051's data seed). Code: `code/data_aloe_pose.py` (`aloe_cfg`, `voice_draw`; env `ALOE_PERSONA_SIDES`, `ALOE_VOICE_RULE`,
`ALOE_VOICE_P10`, `ALOE_VOICE_TAG` in `aloe_clip80.env`; unset env = 23051's cfg exactly).

## Everything else = 23051
- Same stores (read in place from `ppm/aloe-v2pure-ft/stores`, read only), plan seed and order, the same 12,791 rows -> 200 steps x 64 rows,
  LRs, WSD schedule (warmup 7, linear decay to exactly 0 at step 200; decoders x K 8), loss weights / stream multipliers, pose settings,
  eval clips + eval packing (3,481 clips, 486 micros), eval every 25 + export steps, exports at 33/67/100/133/167 (node-local) + 200
  (node-local + consolidated `export-200` on /data), bf16, hourly full resumable saves (node-local, keep 2).
- Step composition: the rows (which clips form each step's 64 rows) and the eval packing are cut with 23051's per-clip position bound,
  so they are 23051's exactly (asserted at launch: rows digest `e606adf327cb0f9d` = items + 64-row groups + eval packing). Only the
  spread of a step's clips over ranks x rounds (memory only; tokens_step normalisers keep the maths) uses this run's larger clip bound:
  4 -> 5 rounds per step at world 24 (2 at world 64).
- Pre-launch check (`tests/plan_clip80_check.py`, `tests/plan_clip80_check.json`): the patched code with 23051's cfg reproduces 23051's
  plan digest `bfdf5ca5beab72db`; with this run's cfg the rows digest is 23051's at world 24 and 64; every micro <= the position budget.
- Trainer additions (checks / logging only): `[prompt]` realised fractions at launch, the rows-digest assert, and a ONE-OFF step-0 eval of
  the init under 23051's prompt rule (`eval_ref23051rule_*`), which must reproduce 23051's step-0 eval.

## Realised prompt fractions (pre-launch check, all clips; the trainer logs the same at launch)
| | clip drawn | clip valid | empty system block | persona |
|---|---|---|---|---|
| this run, train (367,495 clips) | 0.7995 | 0.7422 | 0.2578 | 0 |
| this run, eval (3,481) | 0.8044 | 0.7504 | 0.2496 | 0 |
| 23051's rule, train | 0.3315 | 0.3072 | 0.6928 | 0 (none in the stores) |
| 23051's rule, eval | 0.3232 | 0.2991 | 0.7009 | 0 |

"valid" = drawn and `voice_clips` found a later clip for B (clips late in a recording have none, so ~7% of drawn clips stay empty).

## Resolved-config diff vs 23051 (`[cfgdiff]`, trainer log)
12 keys differ, every other key identical (job 24198 log, 10-18 17:06 PT): 5 prompt-rule keys + 7 run names / paths.
```
data.persona_sides: ref 'both' -> this 'none'
data.voice_mod:     ref 3 -> this '<absent>'
data.voice_p10:     ref '<absent>' -> this 8
data.voice_rule:    ref '<absent>' -> this 'p10'
data.voice_tag:     ref '<absent>' -> this 'aloe-v2pure-clip80'
control_dir, export, local_export_root, log_root, plan_dir, save_root: aloe-v2pure-ft -> aloe-v2pure-clip80
run_tag:            ref 'ppm-aloe-v2pure-ft-j23051' -> this 'ppm-aloe-v2pure-clip80-j24198'
```

## Run
Job 24198, slinky-0/1/2 (world 24), launched 10-18 17:06 PT, run tag `ppm-aloe-v2pure-clip80-j24198`.
- Launch checks: config diff above; rows digest e606adf327cb0f9d == 23051's; plan 90a9782266fd0d3a (23051: bfdf5ca5beab72db: same rows,
  5 rounds per step instead of 4); `[prompt]` fractions as in the table above (persona 0).
- Step-0 eval (normalised word CE, eval_speech/word_ce_norm): **2.2290** under this run's prompt rule (P mean 57). Identity check: the same
  init + eval clips under 23051's prompt rule give **2.2348** = 23051's step 0 (2.2348; raw word CE 2.4409 = 23051's too).
- Step-0 packed vs standalone loss check: max rel err 5.1e-4 (< 2e-3). Peak memory 60.1 GiB per GPU (23051: 60.4).
- Speed: 34-37 s/step from step 2 (23051: ~87-99 s/step on slinky-4/7/15).
- Eval normalised word CE (this run / 23051): step 0 2.2290 / 2.2348, 25 2.1403 / 2.1445, 33 2.1288 / 2.1327, 50 2.1157 / 2.1194,
  67 2.1083 / 2.1121, 75 2.1059 / 2.1097, 100 2.1008 / 2.1045.
- **STOPPED at step 109 (10-18 18:21 PT), not by this run's owner:** `control/STOP_AFTER_SAVE` was touched at 18:21:11 PT and
  `DRAIN_FOR_FT.monoplex` was deleted ~18:23 PT from outside this session. The trainer saved and exited cleanly (rc 0). The TTS fleet then
  reclaimed slinky-0/1/2 (18:23:25 PT). Full resumable save: `/scratch/amartyavarma/aloe-v2pure-clip80/saves/step-00109/` on slinky-0/1/2
  (the step-93 save is also kept). Resume = re-drain the same 3 nodes (re-create `DRAIN_FOR_FT.monoplex`) + `sbatch aloe_clip80.sbatch`.
- Exports on private HF (verified, sha256): `InternalCan/ppm-aloe-v2pure-clip80` export-33/, export-67/, export-100/ (node-local copies kept on
  slinky-0/1/2; Weka copies deleted after verify). Backup controller: `setup-logs/hfb/aloe-bk/clip80/`, job 24215 (tracks job 24198 only).

## Nodes
- 3 TTS-fleet nodes slinky-0, slinky-1, slinky-2 (world 24, as 23051), drained via
  `/data/home/amartyavarma/data/si20k-tts/control/DRAIN_FOR_FT.monoplex`. `release_nodes.sh` (chained job, afterany) deletes ONLY that file,
  and only if the trainer finished 200 steps.
- 8 nodes (world 64: 2 rounds per step instead of 5, same 64 rows per step): `sbatch -N8 -w <8 nodes> aloe_clip80.sbatch` (fresh run only: a resume needs the
  same nodes / world). The run script aborts on slinky-4/7/15/18 and the 14B FT nodes 9-14, 16, 19.
- `/scratch` cleanup on slinky-0/1/2 to fit the saves: `/data/home/amartyavarma/scratch-inspect/slinky012-cleanup-1018/DELETED.log`
  (s3-rows copies verified on private HF first; caches; one superseded Qwen dcp).

## Stop, resume
- Stop (full save, clean exit): `touch /data/home/amartyavarma/ppm/aloe-v2pure-clip80/control/STOP_AFTER_SAVE` (or `scancel -b -s USR1 <job>`).
- Resume: `sbatch /data/home/amartyavarma/ppm/aloe-v2pure-clip80/aloe_clip80.sbatch` (ALOE_RESUME=auto; same nodes; `RUN_TAG.txt` keeps one
  metrics.jsonl).
- Saves: `/scratch/amartyavarma/aloe-v2pure-clip80/saves/step-NNNNN/` per node. Exports: `/scratch/amartyavarma/aloe-v2pure-clip80/exports/`
  (node-local) and `export-200` (consolidated, /data). Backups: private HF `InternalCan/ppm-aloe-v2pure-clip80` (never GCS).
