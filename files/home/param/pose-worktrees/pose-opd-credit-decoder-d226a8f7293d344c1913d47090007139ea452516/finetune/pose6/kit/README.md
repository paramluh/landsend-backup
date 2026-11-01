# pose6/kit: monoplex Aloe pose FT with s2p cb0 + G0COND (launch kit, NOT launched)

Workstream B3 of pose6, 10-27 14:07 PT. The recipe is `ppm/aloe-pose-mono-8020`: job 24272, which produced export-357. Only three things change: the base, `PPM_S2P=cb0` and `PPM_G0COND=1`. Two more items come from main's 10-27 decisions: `--saves final` and the /scratch rule. One knob is flagged for the user: the decp_bb link. The user decides when to launch.

**Pinned code (pose6/code, B1):** `train_aloe_pose.py` 212b1ee9, `data_aloe_pose.py` 47a55ed5, `model_ppm.py` df679691, `data_ppm.py` ec50020e. These are unchanged since my dry runs. If they change, re-run `sbatch -w slinky-4 tests/dry_all.sbatch` and regenerate `CFG_DIFF.txt`.

## Sequence (who triggers what)

| # | step | command | trigger |
|---|---|---|---|
| 1 | 31101 reaches 19382 (~10-28 17:10 PT at 69.3 s/step). `pose5/kit/end_of_run.sh` waits for `dcp-19382` HF-verified 16/16 and stops pose5's companions | - | automatic |
| 2 | **Convert** `dcp-19382` -> `/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-19382` (88.4 GB on /data) | `bash kit/convert_final.sh` (checks, **print-only**), then the command it prints: `sbatch -w <31101's 16 hosts> kit/convert_final.sbatch` | **main** |
| 2v | **Verify**, inside that job: (1) the converter's own `--finalize` (all 128 rank files + `rep.pt` present -> `COMPLETE`); (2) the kit preflight on `local-19382` (2,152 keys = the FT model + `s_emb` [2048, 2560] + `g0` [5120, 8]; bf16; new tensors finite / non-zero; sidecar coverage). Either failing aborts | (in the job) | - |
| 3 | /scratch on the 8 hosts (see "Hosts"): main frees pose5's own listed tiers, plus `dcp-19382` on slinky-17 / slinky-3 once it is converted + HF-verified. Nothing of MoT's is touched | `pose5/kit/handback_free64.py` lists them | **main** |
| 4 | **FT** (~1.5 h) | `bash kit/launch.sh --base newest [--link-on]` (dry run), then `... --go` | **user** decides; main runs |
| 5 | HF backup of the FT's node-local files to the private repo `InternalCan/ppm-pose6-ft-s2pg0` (never GCS) | ops convention `setup-logs/hfb/aloe-bk/posemono/controller_posemono.sh` | main |
| 6 | Main OKs removal of the FT's node-local copies; `handback_free64.py` must show >= 64 GiB on every host | - | main |
| 7 | MoT resume | `pose5/kit/end_of_run.sh` step 4 = `ppm/mot/guard/stop/mot_resume_pause.sh` | main's GO (motguard) |

**Earlier-start alternative (the user's call).** Use the last 6-hourly export instead of steps 1-2:
- `--base newest` before any `local-*` exists picks the newest `ckpt-weka/ppm-v1/w-N`.
- 31101 exports ONLY every 360 min (`PPM_EXPORT_EVERY_MIN=360`, `PPM_EXPORT_EVERY=0`; `train_ppm.py` 1855-1874), and **never at the final step**.
- From the 11:41 PT start, exports land at about 17:40 and 23:40 PT, then 05:40 and **~11:40 PT 10-28 at ≈ step 19,100**. That last one is about 280 steps (~5.5 h) before the end.
- These w-N base runs still need 31101's hosts, so they wait for its end, or for a pause the user asks for and main triggers with STOP_AFTER_SAVE.

**Why the converter is a kit copy.** `kit/tools/ckpt_local_export_alias.py` = `code-811-s2p/ckpt_local_export.py` (md5 87c639ee) plus `--host-as`.
- 31101's saves record LOGICAL hosts (`PPM_HOST_ALIAS`: slinky-15 plays slinky-9, slinky-7 plays slinky-10). The stock script would assert "holds no rank" on those two hosts. monoft's `ckpt_node_export.py` (`CKX_HOST_AS`) did the same for dcp-17850.
- The stock `.sh` also needs a running job to overlap, which 31101 no longer is at that point. So `convert_final.sbatch` is its own CPU-only job on the 16 hosts.
- The values are cast exactly as the stock script does: fp32 `model` -> bf16, as every `local-N` is made.

**/data ledger.** It must stay >= 300 GB:
- **Checked, fail loud, before the conversion:** free - 89 GB must be >= 300 (`convert_final.sh`, `convert_final.sbatch`). The converter's own assert checks the same on rank 0's host.
- Now: 657 GB free.
- 31101 adds 4 more exports x 88.4 GB (KEEP=0 keeps them all), which leaves ≈ 303 GB at its end. **The conversion then FAILS the ledger (303 - 89 = 214) unless main prunes superseded w-N exports first.**
- The FT's own final consolidated `export-357` (88.4 GB) is refused by the trainer below the same floor; its node-local copy then stays the only one.

## One command (DEFAULT = 64 ranks, the recipe's layout)

```
bash /data/home/amartyavarma/ppm/pose6/kit/launch.sh --base newest          # DRY RUN: prints everything, runs the checks, submits nothing
bash /data/home/amartyavarma/ppm/pose6/kit/launch.sh --base newest --go     # submit (only if every check passes)
```

| flag | default | meaning |
|---|---|---|
| `--base` | (required) | `newest` = the newest COMPLETE 31101 checkpoint carrying `s2p` cb0 + `g0cond` 1, i.e. `local-19382` after step 2, or else the newest w-N. Or give any dir |
| `--hosts` | `slinky-15,14,5,7,1,2,17,3` | 8 production hosts; spare **slinky-8**; never slinky-9 |
| `--rank0` | `slinky-15` | the host that runs rank 0, which also writes `rep.pt` / `meta.json` of every export (+13 GiB). Placement only: `task_ppm.sh` rank = 8 x index in `PPM_HOSTS` + local id. mono8020 used `sort -V` order |
| `--saves` | **final** (main 10-27) | no hourly saves; the final FULL save (masters + AdamW) at step 357. `hourly` = the recipe's (keep 2); `none` = no FULL save. Save cadence only, maths unchanged |
| `--disk-rule` | **floor** (main 10-27) | need per host = the FT's measured write + `DISK_FLOOR_GIB=30` (pose5's floor for 31101). `recipe` = mono8020's 1500 \| 1000 \| 500 / 8 + 50 |
| `--link-on` | off | train the D33 decp_bb link ON. **USER CALL** (see "Decisions") |
| `--nodes 1` | 8 | the FALLBACK one-node kit (see "Fallback"); never the default |
| `--resume`, `--stage` | | resume a stopped run; stage a DCP base node-local (bit-exact) |

**Checks.** The dry run checks the following; `--go` refuses unless every check passes. The run script repeats checks 2-5 inside the job.
1. **Base:** COMPLETE and its format.
2. **Base keys and shapes** (`tools/preflight.py`): see "Verify" in step 2v.
3. **Sidecar coverage:** 27,517 + 531 items.
4. **Hosts:** every host fully idle.
5. **/scratch, live:** `df` in a tiny srun (`-c1 --mem=1G`) on each host. It **refuses below the need: 165 GiB, or 178 on the rank-0 host**. No stale saves.
6. **/data headroom:** warning only.

At the end of the run, the run script logs **the actual write per host** (`[disk] <host> wrote exports / saves / cache / total GiB`, also in `logs/disk-<job>.txt`), which confirms the measured-write rule. The kit never deletes anything.

### Disk rule (GiB per host, df -BG units)

| item | size | derivation |
|---|---|---|
| FULL save | 73 | sharded 43.03B params x 12 B (fp32 master + AdamW m, v) / 8 = 60.1, plus `node.pt` with the replicated decoders (1.15B, fp32 + AdamW) 12.9. From 24272's `[build]`: 44.176B total |
| export slice | 10.0 (+2.2 `rep.pt` on rank 0) | 88.4 GB bf16 per export / 8 hosts (24272: `88.4 GB bf16 shards`) |
| write, `--saves final` | **135** (rank 0 **148**) | 1 save + 6 exports |
| write, `hourly` / `none` | 208 (221) / 62 (75) | 2 saves kept / 0 saves |
| **need, floor rule** | **final: 165 (rank 0 178)**; hourly 238 (251); none 92 (105) | write + 30 |
| need, recipe rule | final 175; hourly 237; none 113 | mono8020's 1000 \| 1500 \| 500 / 8 + 50 |

Cross-check: mono8020 (hourly, 2 saves) went from 249 GiB free to 45 at its minimum, i.e. ~204 GiB written vs the 208 model above.

## Hosts

Projected /scratch free at 31101's hand-back, in GiB. Source: `pose5/kit/state/handback/handback-records-1027-114558.txt` (pose5's `handback_free64.py --records`, 11:46 PT).
- **"+tiers"** = pose5's own removable tiers T2-T5: npl5 / npl3 / npl2 rows, the dcp-17850 pin.
- **"+final"** = also `dcp-19382`, which can go only after step 2 and HF verify.
- Nothing of MoT's is ever counted, and nothing on slinky-3 other than pose5's own listed files.

| host | as-is | +tiers | +final | need (floor, final) | what main must free first |
|---|---|---|---|---|---|
| **slinky-15** (rank 0) | 563.0 | 606.0 | 649.0 | 178 | nothing |
| **slinky-14** | 208.1 | 285.5 | 328.4 | 165 | nothing |
| **slinky-5** | 159.2 | 193.7 | 236.6 | 165 | pose5 T2 (npl5, ~11 GiB) |
| **slinky-7** | 129.6 | 207.0 | 249.9 | 165 | pose5 tiers incl. T5 (T2-T4 = 34.5 is 0.9 short) |
| **slinky-1** | 108.6 | 186.0 | 228.9 | 165 | pose5 tiers incl. T5 |
| **slinky-2** | 104.9 | 182.3 | 225.2 | 165 | pose5 tiers incl. T5 |
| **slinky-17** | 100.2 | 134.7 | 177.6 | 165 | pose5 tiers + `dcp-19382` (after step 2 + HF verify) |
| **slinky-3** | 95.0 | 129.4 | 172.3 | 165 | pose5 tiers + `dcp-19382` (after step 2 + HF verify). MoT's caches untouched |
| slinky-8 (spare) | 90.3 | 124.8 | 167.7 | 165 | pose5 tiers + `dcp-19382` |
| slinky-0 / 11 / 16 / 19 / 12 / 6 / 13 | 87.4 / 87.0 / 83.5 / 83.4 / 89.0 / 88.5 / 71.1 | | <= 164.8 | 165 | do not fit |
| slinky-9 | never | | | | |

The launcher prints these projections. At `--go`, only the live `df` counts.

## MoT interplay (MoT resume needs >= 64 GiB free /scratch per host)

The FT's node-local files on the 8 hosts all sit under `/scratch/amartyavarma/pose6-ft/`:

| path | size per host |
|---|---|
| `exports/export-{60,119,178,238,298,357}` | 60 GiB (+13 on rank 0) |
| `saves/step-00357` (`--saves final`) | 73 GiB |
| `cache/{triton,inductor}` | small (the FT's own compile caches; never MoT's) |
| `base-*` | only with `--stage` |

/data holds `kit/export-357` (88.4 GB), plus `runs/`, `logs/`, `plan/`, `control/`.

Before MoT resumes: HF backup to `InternalCan/ppm-pose6-ft-s2pg0` with verify PASS per host (never GCS). Then main OKs removal, then `handback_free64.py` live (>= 64 GiB), then MoT resume on main's GO. `launch.sh` section 7 prints this list; the kit deletes nothing.

## Files

| file | what |
|---|---|
| `ft_s2pg0.env` | `aloe_mono8020.env` VERBATIM, plus the `[S2PG0]` lines: names / dirs, `QQA_REF_CFG2` (logging), `PPM_S2P=cb0`, `PPM_G0COND=1`, `ALOE_G0_SIDECAR`, the flagged `QQA_DECP_BB_OFF=1`, and the FALLBACK-only one-node overlay |
| `tools/ft_overrides.sh` | the user-call switches `FT_LINK` / `FT_SAVES` / `FT_EXPORTS`, applied after the env by the run script, the launcher and the tests alike |
| `ft_s2pg0_run.sh`, `ft_s2pg0.sbatch` | mono8020's run script / sbatch. Changes are in the header, plus: rank-0 pin, the floor rule, the end-of-run write log, compile caches under `pose6-ft/cache` |
| `launch.sh` | the one command |
| `convert_final.sh`, `convert_final.sbatch`, `tools/ckpt_local_export_alias.py` | step 2 (main) |
| `CFG_DIFF.txt` | resolved-config diff vs job 24272, every variant |
| `tools/preflight.py`, `tools/dry_cfg.py`, `tools/stage_base.py` | base / sidecar check; resolved config via the trainer's own code; optional bit-exact staging |
| `ref/`, `tests/` | 24272's cfg and the FT key table; `dry_all.sbatch` (job 31155, slinky-4), `kit_tests.sbatch`, `test_stage.py`, `tests/out/` |

## Base loading

- **mono8020** read `local-11488` (`ppm-local-bf16-shards-v1`, world 128) in place from /data: `load_init` resharded 128 -> 64.
- **`local-19382`** is the same format and world, read the same way. That is the default base after step 2.
- **A w-N** (consolidated DCP) is read directly by pose6/code's `load_init_dcp` (B1).
- **`--stage`** is a bit-exact fallback (tests S1 / S2).
- 31101 carries no pose_head keys. `load_init` accepts that through the Rosie fix below.

## Rosie fixes, pose-s2p reuse

| fix | needed? | why | maths-neutral: proof |
|---|---|---|---|
| `load_init` unloaded keys may be `[]` | **yes** | 31101 has no pose_head keys (D37) | assert only; same rule in aloe-v2pure-main-17850 (8,1,1 init, `keys NOT loaded: []`) and Rosie 30847 |
| `ALOE_SKIP_FINAL_FULL` | `--saves none` / fallback only | disk / host RAM | runs after the last step + final eval (Rosie 30847 log) |
| AdamW host offload | fallback only | 38 + 574/8 = 110 GiB per GPU at W=8 | bit-identical: optoff.py md5 1024fab9; job 28249 `0 differ` |
| eval pad | inert | 529 eval micros >= ranks | zero-weight dummies |
| `ALOE_MONO_ITEMS` / `ALOE_EPOCHS` | not carried | Rosie data | rows digest asserted |

From `pose-s2p`: I reused the dry-cfg tool, the world-8 overlay (fallback only) and the /scratch-need idea. I did not reuse `PPM_S2P=1` (full) or the zero-init on local-11488.

## Fallback: one 8-GPU node (`--nodes 1`), documented only

- **Layout:** world 8 turns `micro_layout` into `lpt` (9.17 rounds per step, 212 zero-weight dummies). Same rows digest, same objective.
- **Not bit-identical:** the decoder nanostep frame partition is a different seeded draw, and the fp order differs.
- **Memory:** AdamW host offload needed (110 GiB per GPU without).
- **Saves:** no FULL saves (host RAM).
- **Disk:** 528 GiB /scratch on one host.
- **ETA:** ~14 h (Rosie 30847 timings).

Key-by-key maths table, still accurate: same normalisers (`train_aloe_pose.py` 1225-1236), dummies zero-weight (811-812), packing loss check < 2e-3, the same `train_step_multi` path; FLAG decoder chunk randperm (723-727) and fp order.

## Decisions

1. **decp_bb link: OFF (default) or `--link-on`.** pose6 recommends OFF.
   - mono8020 trained it ON because its base had it on.
   - 31101 has had it OFF since 10-20 (frozen; exports alpha 0).
   - I scanned every other base-state knob: CD, dec attn, decoder K, decoder mode, pose head, key set / shapes, frozen parts. None differs.
2. **Base:** `local-19382` (main's decision), or the earlier w-N ≈ 19,100 (user's choice).
3. **/data:** main prunes superseded 31101 w-N exports before step 2. Otherwise the ledger check fails.

## Tests (CPU only; no test data on slinky-18 or slinky-9)

| test | result |
|---|---|
| D0: dry mode faithful (ORIGINAL mono8020 code + env, 64 ranks vs 24272) | **0 keys differ** |
| **D1: DEFAULT** (64 ranks, `--saves final`, link off), job 31155 | **17 keys:** 7 names / paths; `init`; s2p / lr.s2p / clip.s2p; g0cond / g0_sidecar / lr.g0cond / clip.g0cond; flagged `decp_bb_off`; `save_every_min` 60 -> 100000 (`--saves final`). Plan digest a5d995bcec45dc6f = 24272's; rows digest 3681cc26ea97cad7 |
| D4 / D5 / D6 | `--link-on` 16 (no `decp_bb_off`); `--saves hourly` 16 (no `save_every_min`); `--saves none` 17 |
| D2: fallback | 18 (+ `micro_layout` lpt) |
| D3: control, all mods off + hourly | **8: names / paths + init only** |
| P1 / P2: preflight w-15973 / export-357 | FAIL as expected: exactly the 2 new keys missing; 0 extra / 0 shape / all bf16; sidecar 27,517 / 27,517 + 531 / 531 |
| S1 / S2: `stage_base` + the `load_init` loop | 0 / 272, 0 / 544, 0 / 32, 0 / 32 differ |
| S3: preflight logic | 5 / 5 |
| `convert_final.sh` | runs clean (FAILs as expected now: 31101 at step 17977, hosts busy; ledger ok at 657 GB) |
| `launch.sh` dry runs (default, recipe rule, fallback, flags) | clean (the FAILs are expected: no base yet, hosts busy) |

**Pending:** the positive preflight runs on `local-19382` inside the convert job (verify 2) and again at launch.
