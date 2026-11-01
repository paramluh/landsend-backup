# pose6/infer: the engine with BOTH mods (s2p cb0 + G0COND). Prepared, NOT deployed

Workstream A1 of pose6, 10-27. Tests and their numbers are in `TESTS.md`.

## Files

| file | what it is |
|---|---|
| `ppm/fast_ppm.py` | `pose-s2p/infer/ppm/fast_ppm.py` (= the deployed `aloe-pose-monoplex-infer/ppm/fast_ppm.py` + s2p) + **G0COND** + the **link-off alpha** fix + checkpoint-flag checks. Diffs: `infer_fast_ppm_vs_pose_s2p.diff`, `infer_fast_ppm_vs_deployed.diff`. |
| `ppm/*.py` (all others), `infer.py` | byte-identical copies of `pose-s2p/infer` (= the deployed files). The md5s are in `ORIG_MD5.txt`. |
| `tests/` | `cpu_load.py`, `cpu_slot_train.py`, `cpu.sbatch`, `cpu_small.sbatch`, `test_engine6.py` + `.sbatch`, `gpu_guard.py`; `out/` (json, micros), `logs/` |

The deployed engine and pose-s2p are untouched. The patch that fixes the link-off alpha in those two is in `pose6/enginefix/`.

## G0COND in the engine

The term is the trainer's `model_ppm.Inp.forward` G0COND term (pose5/code-811-s2p `model_ppm.py:253-255`; B1's `pose6/code/model_ppm.py:245-247` is the same code):
```
inp.g0 = Linear(8 -> 5120, bias=False)
trunk_input = trunk_input.index_add(0, [SLOT], inp.g0(f.to(bf16)).to(trunk_input.dtype))
```
- **f** is `data_ppm.g0_feat`'s 8 features: `[r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w, log(H/W)]`, stored as float32. It is cast to the bf16 weight dtype; under FSDP the `inp` unit's `cast_forward_inputs` does the same cast.
- **Where it runs.** `FastPPM._prefill_row(..., g0=(SLOT, f))`, on every row of the instance. The SLOT is always a prefill position, so the per-frame step and its CUDA graphs do not change at all.
- **Flag off.** A model without `inp.g0` means `self.g0 is None`, and `_prefill_row` runs its original line. `start()` only adds a check that refuses an instance carrying `g0f`.

### Model build follows the checkpoint

`build_model` turns G0COND on iff `inp.g0.weight` is in the checkpoint; otherwise `PPM_G0COND=1` turns it on.

| case | result |
|---|---|
| checkpoint has the key | module built and loaded exactly (shape [5120, 8] asserted) |
| `PPM_G0COND=0` + checkpoint has the key | AssertionError |
| no key + `PPM_G0COND=1` | ZERO init, logged, `binfo["g0cond_zero_init"]` (tests only) |
| no key + env unset | no module; today's path |

`binfo` gains `g0cond` and `g0cond_zero_init`. `FastPPM(..., g0=None | False | module)` follows the model by default. Passing a module is for tests that share one copy of the weights.

### Instance API (= `pose6/briefs/CONTRACT.md`)

```python
eng.start([dict(x, a_mode="empty", g0f=f)], seed, params, f0=0)   # f = np.float32[8], one per instance
```

`start()` checks every instance before it changes any state:

| condition | result |
|---|---|
| G0COND engine, `g0f` missing or None | `ValueError` |
| `g0f` wrong shape, not a float, or not finite | `ValueError` |
| engine without G0COND, `g0f` given | `ValueError` (never ignored silently) |
| position P-1 is not the system block's codes-free `'\n'` | `AssertionError` (SLOT check below) |

- `f` is cast to float32, the dtype `g0_feat` stores.
- `eng.g0_rows` lists the (row, SLOT, f) actually applied at the last `start()`.
- `infer.py`, the CLI copy, is not G0-aware. On a G0COND model it raises the `ValueError` above.

## SLOT

**FT / monoplex layout (export-357, the playground).** SLOT = **P - 1**, per decisions D1 and D2. This is the system block's last position, inside the prefill.

`sysprompt.build_prefix` ends every system block with the footer `<|im_end|> \n` on channel B. The inference copy (`infer/ppm/sysprompt.py`) and the FT trainer's copy (`pose-s2p/code/lib/sysprompt.py`) are verbatim apart from a header line, and their final assert is `ids[1,-2] == IM_END` with no codes at position -1. So at P - 1 both inference and training have:

| field at P - 1 | value |
|---|---|
| `ids_B` | 198 (`'\n'`), with `ids_B[P-2]` = 151645 (`<|im_end|>`) |
| `present` | `[0, 1]`; channel A is absent (PAD, present 0) |
| Mimi codes (`depth`) | none, either channel |
| pose codes (`pdepth`) | none |

Pose frame 0 is labelled at P + CD = P + 2. Every prompt type checked has this layout:

| prompt type | inference `system_block` P | training item P (real Aloe eval item) |
|---|---|---|
| empty block | 5 | 5 (item [4, 1772, 0]) |
| voice clip | 134 (10 s clip) | 70 (item [0, 698, 0], 61 clip frames) |
| persona | 26 | 929 (item [0, 2203, 0]) |
| clip + persona | 155 | 212 (item [3, 336, 0]) |

- The rule held on **all 528 Aloe eval items** scanned (`cpu_slot_train.py`; FT trainer data path = mono8020's).
- B1's FT data code puts `g0pos = offset_k + P_k - 1` (`pose6/code/data_aloe_pose.py:737`, checked at :724-733), which is the same position.
- The trunk input at P - 1 is `emb_B(198) + g0(f)` in both training and inference.

**Base 8,1,1 layout (`pose_s3q`).** A pose row is packed clips `[SLOT 1 | clip | GAP 4]`; each clip is its own attention segment with pid restarting at 0. There is no system block. `gpos` is the segment's first position (`lib/pose_s3q.py:264`). The lane there is `lane[s0]`, and its codes and pose are -1.

Mapped onto the engine's coordinates, that is a 1-position "system block" (P = 1) with SLOT = P - 1 = 0: clip frame 0's lane sits at SLOT + 1 and its codes at SLOT + 1 + CD, exactly as in the FT layout. So the rule is "SLOT = P - 1" in both layouts.

**This engine and the playground never serve the base layout.**
- `start()` asserts P >= 5, and every caller builds a system block (`infer.system_block` → `sysprompt.build_prefix`).
- genweb `worker.py` serves only FT checkpoints:
  - export-357 (`worker.py:35`);
  - Rosie export-18 (`rosie_unit.sh:9`).
- The duplex playground and live demo run `live_ppm_pose.py` (export-498), a different engine.
- `ppm-pose-server` has its own engine copy (`fast_ppm_srv`), for pose-FT exports.

So nothing in the base layout needed support. A base 31101 export loaded here (the load check) runs in the FT layout with SLOT = P - 1. Base pose rows never have a system block, so serving a base model this way is off-distribution with or without G0COND.

## Uncond rows get the SAME f as cond (D3): proof from the training code

**Rows in this engine.**
- The only uncond rows are the pose-CFG rows (`cfg_rows`). They keep the same lane, system-block text and pose history, and every Mimi code becomes digital-silence codes.
- There are **no audio-CFG rows**: `cfg_rows = cfg_rows and self.pose`, and `_text` and `_speech` read rows `[:NI]` only. The Voxtral front's uncond table is asserted off in this vendored engine.
- The SLOT position P - 1 carries no Mimi codes, so the silence substitution leaves it untouched. The uncond row's input at P - 1 is the cond row's input exactly when f is the same; GPU test (w1) checks this at the KV level.

**What training does with f.**

1. **FT trainer** (`pose-s2p/code` = mono8020 + Rosie fixes + s2p; B1's `pose6/code` adds G0COND):
   - There is no CFG-dropout or uncond construct. A grep for `uncond|silence|cfg_drop|cond_drop|p_uncond|null_cond|drop_p|drop_prob` over `train_aloe_pose.py`, `data_aloe_pose.py`, `data_ppm.py`, `model_ppm.py` and `lib/` finds only:
     - the run-family name "aloe-uncond-ft", which means channel A is empty on EVERY row (`data_aloe_pose.py:1-8, 64, 221`; not a dropout);
     - `build_v2pure_rows.py`'s silence partner codes (another store; the FT uses `ALOE_STORE_DIR=aloe-pose-ft/stores`, `aloe_mono8020.env:136`);
     - the `codep_s3q` docstring.
   - The FT trainer does not use codep, xdep or silence at all: a grep of `train_aloe_pose.py` is empty.
   - B1's port attaches g0pos/g0f to EVERY pose micro, unconditionally: single item at `pose6/code/data_aloe_pose.py:335-336`, packed at `:564-565`, from the item's sidecar record at `:316-317`.
2. **8,1,1 trainer** (`pose5/code-811-s2p`, run 31101):
   - f comes from `data_ppm.g0_feat` (`data_ppm.py:257-272`). f = 0 ONLY when frame 0 is invalid (`:271-272`), and that is logged loudly.
   - It is attached to every pose micro of the pose datasets (`data_ppm.py:288-299`; `train_ppm.py:1192` `pcfg`, passed to PoseDS / FreshPoseDS train and eval).
   - It is applied on every pose micro (`model_ppm.py:553-558` → `:253-255`).
   - No construct drops or zeroes f together with the audio. The only place f is altered is the EVAL-ONLY codependence probe: `train_ppm.py:522-536` `codep_g0`, run inside `evaluate()` (`:1582`, `:1609`):
     - `pose_shuf` gives the partner clip's f;
     - `pose_zero` gives f = 0, the pose-input ablation;
     - `speech_shuf` and `speech_sil` keep the clip's OWN f.
   - `speech_sil` (Mimi codes absent, lane PAD; `lib/codep_s3q.py:12`) is the training-side analog of an audio-off condition, and it keeps f.

**Conclusion.** f is a property of the clip's frame-0 pose geometry, not of the audio. The uncond (silence) row gets the same f as the cond row. D3 is confirmed and implemented.

## Load path for the 8,1,1 exports (31101) and FT exports

**31101 exports.**
- Location: `PPM_EXPORT_ROOT=/data/home/amartyavarma/ppm/ckpt-weka`, `PPM_RUN=ppm-v1`, so `ckpt-weka/ppm-v1/w-<step>`, every 360 min (`PPM_EXPORT_EVERY_MIN=360`, `PPM_EXPORT_KEEP=0`).
- Format: a consolidated bf16 DCP (`safety_ppm.export_weights`: `dcp.save` of `get_model_state_dict`) + `train_cfg.json` side `{step, decp_bb: 1, decp_bb_alpha: 0.0, decp_bb_off: 1, s2p: "cb0", g0cond: 1}` (`train_ppm.py:1872-1876`) + `COMPLETE`.
- This is the same format as the existing `ppm-v1/w-15973`.
- The 31101 model built from code-811-s2p with its flags (meta) has exactly w-15973's keys and shapes plus `inp.g0.weight` [5120, 8] and `tower_p.s_emb.weight` [2048, 2560]; the engine's model has no key outside that set. So the engine loads a 31101 export with no other change (`TESTS.md` L2).

**FT exports.**
- Local shards (`ppm-local-bf16-shards-v1`, like export-357): `rank<r>.pt` dim-0 shards + `rep.pt` + `meta.json` cfg.
- Or a DCP export.
- The new keys need no special handling. The local loader concatenates any key's rank shards; DCP reads any key.

**Engine checks.**
- `ckpt_info` detects s2p (from s_emb rows) and G0COND (from key presence).
- If the record (DCP side, or local meta cfg) carries `s2p` / `g0cond`, it must agree with the tensors; otherwise AssertionError.
- Both formats are tested exactly on synthetic checkpoints (L1).

**LINK OFF (31101 `PPM_DECP_BB_OFF=1`).** The trainer never computes the D33 link: the pose decoder gets `zb=None`. The engine now uses alpha 0 whenever a record says `decp_bb_off: 1`, in both formats (`_link_alpha`).
- Before this, a local copy of a 31101 save (cfg `decp_bb_alpha: 1.0` + `decp_bb_off: 1`) or an FT link-off export would have run with alpha 1.0.
- With alpha 0, slot 0 = `proj(hv_p)` bit-for-bit, all finite (`TESTS.md` LK).
- The same fix for the deployed and pose-s2p engines is in `pose6/enginefix/`; the duplex `live_ppm_pose.py` has the same bug.
- **For B1:** the FT DCP export side hardcodes `decp_bb_alpha: 1.0` (`pose6/code/train_aloe_pose.py:665`). A link-off FT must record `decp_bb_off: 1`.

**Hazard: never serve the new checkpoints with the old engines.**
- The deployed engine does not know `tower_p.s_emb` or `inp.g0`. It would build a model without them and still load: the DCP path logs them as "not used"; the local path skips them.
- The pose-s2p engine knows s_emb but not inp.g0.
- Both would silently drop the mods.

## s2p cb0

This is pose-s2p's code, unchanged (`pose-s2p/README.md`). Speech is split at the cb0 draw, and pose runs after the cb0 draw, concurrently with the speech light decoder. Uncond rows read silence cb0.
