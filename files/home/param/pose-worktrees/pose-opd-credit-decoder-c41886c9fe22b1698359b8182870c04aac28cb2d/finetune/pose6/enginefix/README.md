# pose6 enginefix (prepared, NOT deployed)

This directory holds two fixes:
1. A recorded `decp_bb_off` forces the D33 link alpha to 0.
2. Checkpoints carrying the pose6 mods are refused instead of silently dropped (section "Fix 2").

pose6 workstream A1, 10-27 12:38 PT. Per main, this goes in with the first model swap, and only on the user's OK.

## The bug

The 8,1,1 run 31101 trains with the D33 backbone-to-pose-decoder link OFF (`PPM_DECP_BB_OFF=1`). The trainer never computes the link: the pose decoder is called with `zb=None`. The link scale must therefore be 0 at inference.

None of the three serving engines reads the `decp_bb_off` record. Each takes alpha only from `decp_bb_alpha`:

| engine | local copy (`ppm-local-bf16-shards-v1`) | DCP export |
|---|---|---|
| deployed monoplex `aloe-pose-monoplex-infer/ppm/fast_ppm.py` | `decp_bb_alpha_of(meta.cfg)` | `train_cfg.json["decp_bb_alpha"]` |
| `pose-s2p/infer/ppm/fast_ppm.py` | same | same |
| duplex `aloe-posedup-live/live_ppm_pose.py` (lines 198, 215; `ppm/code/infer_ppm.decp_bb_alpha_of` lines 87-92) | `decp_bb_alpha_of(meta.cfg)` | `decp_bb_alpha_of(train_cfg.json["cfg"] or train_cfg.json)` |

What each kind of link-off checkpoint gives today:

| link-off checkpoint | record | monoplex / pose-s2p today | duplex today | correct |
|---|---|---|---|---|
| 31101 DCP export (`ckpt-weka/ppm-v1/w-N`) | side `{decp_bb_alpha: 0.0, decp_bb_off: 1}`, no `cfg` | 0.0 (correct by accident) | 0.0 (correct by accident) | 0 |
| ppm-local copy of a 31101 save (`ckpt_local_export.py`: meta.cfg = training cfg) | cfg `{decp_bb_alpha: 1.0, decp_bb_off: 1}` | **1.0** | **1.0** | 0 |
| FT DCP export, link off (`pose6/code/train_aloe_pose.py:665` writes top-level `decp_bb_alpha: 1.0` + `cfg`) | side alpha 1.0; `decp_bb_off` top-level or in `cfg` | **1.0** | **1.0** | 0 |
| FT local export, link off (meta.cfg = FT CFG) | cfg `{1.0, decp_bb_off: 1}` | **1.0** | **1.0** | 0 |

## The patch (identical in all three)

Each patch makes three edits:
- In `ckpt_info`, both alpha expressions, local and dcp, become `_link_alpha(path, record, <the same expression>)`.
- A 12-line helper `_link_alpha` is added. If `decp_bb_off` is truthy in the record or in its nested `cfg`, it returns 0.0, logging if the recorded alpha was non-zero. Otherwise it returns the alpha unchanged.
- No other line changes.

With alpha 0 the link adds exactly 0: `zb = decp_bb(hn) * 0.0`, so slot 0 = `proj(hv)` bit-for-bit, all finite. That is the trainer's zb=None graph (pose6/infer TESTS.md LK).

| patched copy | patch | from (md5) | patched md5 |
|---|---|---|---|
| `deployed/fast_ppm.py` | `deployed_fast_ppm.patch` | `aloe-pose-monoplex-infer/ppm/fast_ppm.py` (b5476bd67f60e89dd42d6d16d5ac3bc5) | ecec48b000b24dc2a4dd7fd8a85947fd (fixes 1 + 2) |
| `pose-s2p/fast_ppm.py` | `pose-s2p_fast_ppm.patch` | `pose-s2p/infer/ppm/fast_ppm.py` (9c0729943767e22de7b431c2d5b5a445) | ebe8d1ae095ee82a6bdd66d73ef18c30 (fixes 1 + 2) |
| `duplex/live_ppm_pose.py` | `duplex/live_ppm_pose.patch` | `aloe-posedup-live/live_ppm_pose.py` (f445c7b92711275b315c17076ece394e) | 7e874acf63b119a7d64be706c0cec083 (fixes 1 + 2) |

`pose6/infer/ppm/fast_ppm.py` already carries the same helper.

## Fix 2: refuse checkpoints that carry the pose6 mods (main's request, 10-27 ~13:05 PT)

**The hazard.** The old engines do not know `inp.g0.weight` (G0COND) or `tower_p.s_emb.weight` (s2p). They build the model without these tensors and load the rest:
- DCP exports log the tensors as "not used";
- local shards skip them.

So a 31101 export or a pose6 FT checkpoint would be served with its mods silently dropped.

**The fix.** `build_model` now calls `_refuse_pose6_keys(ci)` right after `ckpt_info`, before anything is built. It raises `RuntimeError` naming the keys and the engine to use: "... Serve it with the pose6 engine ppm/pose6/infer/ppm/fast_ppm.py". The keys refused:

| engine | refused keys |
|---|---|
| deployed monoplex | `inp.g0.weight`, `tower_p.s_emb.weight` |
| pose-s2p | `inp.g0.weight` only (it serves `s_emb` itself) |
| duplex | `inp.g0.weight`, `tower_p.s_emb.weight` |

How the keys are read:
- DCP: from the checkpoint's metadata (`ci["shapes"]`);
- local: `rep.pt` + `rank0.pt` keys, mmap (the same read `ckpt_info` does).

`ckpt_info` itself is untouched by fix 2, so hot-swap watchers that only inspect candidates do not raise.

**Diff.** Each patch now:
- removes the same 2 alpha lines;
- adds their 2 replacements, 1 call line in `build_model`, and 3 top-level definitions: `_link_alpha`, `POSE6_KEYS`, `_refuse_pose6_keys`. This is checked by parsing the added lines (job 31151).

## Proofs (CPU): ALL PASS

The scripts are `test_fix.py` (monoplex and pose-s2p) and `test_fix_duplex.py` (duplex, run in its own process). Jobs:
- 31143 and 31145 on slinky-4: full runs; synthetic records on slinky-4 /scratch, a few KB;
- 31146 on slinky-18: duplex real models only, read-only, nothing written to /scratch;
- 31137 on slinky-18: md5 of the export-498 serving copy.

Output: `test_fix.json`, `duplex/test_fix_duplex_slinky-{4,18}.json`, `logs/`.

### (D) Line diff

| engine | lines removed | lines added |
|---|---|---|
| monoplex | the 2 alpha lines | 14 (the 2 replacements + the helper) |
| pose-s2p | the 2 alpha lines | 14 |
| duplex | the 2 alpha lines (198, 215) | 14 |

### (R) Every deployed / configured model: `ckpt_info(original) == ckpt_info(patched)`

The whole returned dict is compared. The engines read nothing else from a checkpoint, and every other line is unchanged, so they serve byte-identical outputs. No model records `decp_bb_off`.

| model | who loads it (from the units / configs) | format | engine(s) | alpha orig = patched | decp records |
|---|---|---|---|---|---|
| `aloe-pose-mono-8020/exports/export-357` | genweb `worker.py` (`GENWEB_CK` default) | local | monoplex, pose-s2p | 1.0 = 1.0, dict identical | meta.cfg `{decp_bb_alpha: 1.0}` |
| `aloe-pose-rosie/exports-local/export-18` | genweb `rosie_unit.sh` | local | monoplex, pose-s2p | 1.0 = 1.0, identical | meta.cfg `{decp_bb_alpha: 1.0}` |
| `/scratch/amartyavarma/aloe-posedup-live/export-498-dcp` (slinky-18) | live demo `SWAP_TO.txt` / `LOADED.json`; stage-lb instances C, P, B; `CK_LABELS.json`; `demo.sbatch` CK; genweb `worker_dup.py` | dcp | duplex | 1.0 = 1.0, identical | side `{decp_bb: 1, decp_bb_alpha: 1.0}`, side.cfg `{decp_bb_link: 1, decp_bb_alpha: 1.0}` |
| `aloe-posedup-live/ck/export-498-dcp` (/data; same `train_cfg.json` 8e84a8d3... and `.metadata` 18971473... as the slinky-18 copy) | stage-lb inst2 / instS4 `LOADED.json` + `SWAP_TO.txt`; `switch_1027/restore.sh` posedup-lb-P | dcp | duplex (and monoplex / pose-s2p, checked too) | 1.0 = 1.0, identical in all three | same |
| `/scratch/amartyavarma/pdsamples/export-498` (slinky-4 and slinky-18) | posefix/exact sbatches | local | duplex | 1.0 = 1.0, identical | meta.cfg `{decp_bb_link: 1, decp_bb_alpha: 1.0}` |
| `/scratch/.../aloe-posedup-live/export-333`, `/scratch/.../export-498` (CK_LABELS bring-up/fallback entries), `aloe-ft-32-samsep/poseduplex/export-poseduplex-v5-498` | `CK_LABELS.json` only | - | - | **not checked**: absent on slinky-4 and -18 and on /data; per RESTART_CMD.txt the fallback sits on slinky-10 /scratch, which had no free CPU | - |

### (G) Fix 2, jobs 31151 (slinky-4) + 31152 (slinky-18, read-only)

**Real models.** On every model in the (R) table, the guard passes:
- export-357 and Rosie export-18 (monoplex and pose-s2p);
- export-498 /data copy and slinky-18 serving copy, and pdsamples/export-498 (duplex).

None of them carries either key.

**Synthetic checkpoints** with the keys, in both formats (DCP and local):

| keys | deployed | pose-s2p | duplex | original engines |
|---|---|---|---|---|
| `inp.g0` only | build_model raises (names the pose6 engine) | raises | raises | `ckpt_info` succeeds, no guard (they would build and drop it) |
| `s_emb` only | raises | guard passes (it serves s_emb) | raises | same |
| both | raises | raises | (covered by the single-key cases) | same |

### (S) Synthetic records: patched gives alpha 0 exactly when `decp_bb_off` is recorded; the original gives the knob

| record | monoplex / pose-s2p orig → fix | duplex orig → fix |
|---|---|---|
| dcp side `{alpha 0, off 1}` (31101 export) | 0 → 0 | 0 → 0 |
| dcp side `{alpha 1, off 1}` | **1 → 0** | **1 → 0** (off at top level, cfg `{alpha 1}`) |
| dcp side `{alpha 1}` + cfg `{alpha 1, off 1}` (FT export, flag only in cfg) | **1 → 0** | **1 → 0** |
| dcp side `{alpha 1}` (= export-498) | 1 → 1 | 1 → 1 |
| local cfg `{alpha 1, off 1}` | **1 → 0** | **1 → 0** |
| local cfg `{alpha 1}` (= export-357 / Rosie) | 1 → 1 | 1 → 1 |

## Deploying (only on the user's OK, with the first model swap)

For each engine:
1. Copy the patched file over the live one, keeping `.bak-pre-linkoff-1027`, and check the md5 against the table above.
2. Restart make-before-break.
   - The duplex live backend and the single avatar/instance backends need main's OK and a standby (no single-copy restarts).
   - The genweb units are rolled one idle backend at a time.
3. Every currently served model keeps alpha 1.0 and an identical `ckpt_info`.
