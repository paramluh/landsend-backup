# pose6/code: monoplex Aloe pose FT with both 8,1,1 mods (workstream B1)

This is a copy of `ppm/pose-s2p/code`, which is aloe-pose-rosie's env-gated fixes on top of the mono8020 trainer, plus `PPM_S2P`. On top of that copy it adds:
- G0COND, ported from `pose5/code-811-s2p` (job 31101's code);
- the own optimizer groups `s2p` / `g0cond` (decision D4);
- the base init rules (D5);
- LINK OFF, ported from 811's `PPM_DECP_BB_OFF`.

Source md5s are in `ORIG_MD5.txt`. Nothing is launched.

**Proofs: ALL PASS** (job 31133, CPU, 10-27 12:31-12:50 PT; numbers in `TESTS.md`):
- (1) flags off = mono8020's trainer, bit for bit: cfg, forward terms, every gradient, decoders, and the AdamW step.
- (2) init from a base with the new keys is exact, with FSDP resharding 3 -> 2, from both the DCP format and the local-shards format; the D5 rules fail loud.
- (3) the 811 model vs pose6, with non-zero s_emb / g0, on Aloe packed / EMPTY / CLIP / PERSONA / BOTH and a real TikTok 811 pose micro: terms and grads max abs 0.0.
- (4) the new params get gradients only through their own path, and every FT micro is a pose micro.
- (5) zero new params = flags off.
- LINK OFF: matches 811's link-off path, decp_bb gets no gradient, and the exports record alpha 0 / decp_bb_off 1.

Pinned md5s are in `CODE_MD5.txt`; diffs are in `diffs/`.

## Flags

Every flag is off by default. With all of them off, the trainer is mono8020's trainer exactly: the same maths, and the same resolved config. TESTS.md (1) checks this.

| env | effect | cfg keys (only when on) |
|---|---|---|
| `PPM_S2P=cb0` | `tower_p.s_emb` (pose-s2p), now in its OWN main-AdamW group `s2p` (D4; pose-s2p had it in `tower_p`) | `s2p`, `lr.s2p`, `clip.s2p` (= tower_p's) |
| `PPM_G0COND=1` + `ALOE_G0_SIDECAR=/data/home/amartyavarma/ppm/pose6/g0aloe/g0aloe.parquet` | `inp.g0` = Linear(8 -> 5120, no bias), own group `g0cond`; trunk input at each item's SLOT += g0(f) | `g0cond`, `g0_sidecar`, `lr.g0cond`, `clip.g0cond` (= audio's) |
| `QQA_DECP_BB_OFF=1` | D33 link frozen, never computed, no grad; exports record `decp_bb_alpha 0` + `decp_bb_off 1` | `decp_bb_off` |
| `ALOE_TEST_NEWPARAM_RANDOM=<seed>` | TEST ONLY: when the base lacks the new keys, fill them with seeded random non-zero values | `test_newparam_random` |

How the own groups behave (`model_ppm.JOIN = {g0cond: audio, s2p: tower_p}`):
- LR and clip value are the JOIN group's.
- They use the same schedule lambda as every group (this trainer warms every group up).
- Each group has its own clip norm.
- So with zero-valued new params, every other tensor's update is bit-identical to flags off, including after the optimizer step (TESTS.md (5)).

## Files changed (vs pose-s2p/code)

- **`model_ppm.py`**:
  - `G0_DIM`.
  - `Inp.forward(g0=)`: the index_add, byte-for-byte the same op as 811.
  - `PPM(g0cond=)`: `self.g0cond`, `inp.g0`.
  - `PPM.forward`: asserts that a pose micro carries `g0pos` / `g0f` when on.
  - `GROUPS` + `JOIN` + `group_of` (`inp.g0.weight` -> `g0cond`, `tower_p.s_emb.weight` -> `s2p`), `init_rule` (g0 zero), `build(g0cond=)`.
  - `decp_bb_on` (link-off gate, default True).
- **`data_ppm.py`**: `G0_DIM` / `G0_FIELDS` / `G0_STATS` / `g0_problem` / `g0_feat`, copied byte-identically from `pose5/code-811-s2p/data_ppm.py:243-276`. TESTS.md checks the bytes.
- **`data_aloe_pose.py`**: the G0COND section. The flag is read from env, as `QQA_POSE_B` is:
  - `g0_table` / `g0_item` / `g0_coverage`: the sidecar reader, which fails loud.
  - `g0_slot_check`: what sits at P - 1, asserted per item.
  - `g0_arrays`: g0pos = packed offset + P - 1.
  - Hooks: `Stores.conv` (per-item f), `as_micro` / `pack_micro` (g0pos / g0f).
- **`train_aloe_pose.py`**:
  - the flags, groups and cfg keys above; `ARR` + g0pos / g0f;
  - `load_init` (D5) + `load_init_dcp` (a 31101 `w-<step>` DCP export, aloe-v2vox-asr-w15973's pattern);
  - the startup coverage assert; LINK OFF in run_micro / micro_losses / train_step_multi;
  - the frozen set, and the export side keys (DCP `train_cfg.json` + node-local `meta.json` `side`).

## D1: what sits at the SLOT

### (a) The FT SLOT = P - 1, for every prompt type

`sysprompt.build_prefix` (`lib/sysprompt.py:175`, `:209-222`) always opens with `<|im_start|> system \n` and closes with `<|im_end|> \n`. It asserts that the last position carries no codes.

So for every prompt type, P - 1 is the footer's `\n`:

| prompt type | system block | P |
|---|---|---|
| EMPTY | header + footer | 5 |
| CLIP | header + `<sosb>` lane(L) PAD PAD `<eosb>` + footer | L + 9 |
| PERSONA | header + `<sop>` ids `<eop>` + footer | |
| BOTH | header + clip + persona + footer | |

At P - 1, on every type:

| field | value |
|---|---|
| lane B | 198 (`\n`); P - 2 = 151645 `<\|im_end\|>` |
| present_B | 1 |
| channel A | PAD, present 0 |
| speech codes | none, on both channels (depth -1) |
| pose | none (pdepth -1) |
| text target | none: targets start at P (`data_aloe_pose.py:268`, `sel >= P`) |

Conversation frame 0's lane sits at P. Its first speech code and first pose label sit at P + CD = P + 2.

`g0_slot_check` (`data_aloe_pose.py:723`) asserts all of this on every item. TESTS.md A records it per prompt type on real items.

### Base SLOT (job 31101)

`lib/pose_s3q.py`:
- The SLOT is `gpos` = each clip segment's FIRST position (`:264`). It is also the start of the attention segment: pid 0, and it attends to nothing before it.
- Lane: PAD, or NEW_WORD when the clip's first word starts at clip frame 0 (`qwen-1ch-data/pose-s3/POSE_LAYOUT.md:25,34`).
- present_B 1, no codes (`:202`, q < CD), no pose.
- It HAS a text target, predicting clip frame 0's lane (`:239-240`).

### Timing: the same SLOT -> frame-0 offset

| | lane of frame f | pose label of frame f | where |
|---|---|---|---|
| base | gpos + 1 + f (`pose_s3q.py:174`: start = slot + SLOT) | gpos + 1 + f + pose_delay (`pose_s3q.py:207-208`: pdepth[q] = pose of row frame s0 + q - pd) | pose_delay = SP.CD = 2 (`code-811-s2p/data_ppm.py:95`) |
| FT | P + f = SLOT + 1 + f | P + pose_delay + f = SLOT + 1 + f + pose_delay (`data_aloe_pose.py:189-192`) | pose_delay = 2 (asserted = CD, `:186`) |

The offset matches the base's, and it is asserted per item: the first pose label is at SLOT + 1 + CD.

### Differences from the base SLOT

The SLOT is a different position in two ways:
1. **Token.** The FT SLOT holds `\n` (198); the base SLOT holds PAD / NEW_WORD. g0(f) is additive, so at FT step 0 the base-trained W_g0 lands on a different token embedding.
2. **Context.** The FT SLOT is the end of the system block; it sees the voice clip and persona. The base SLOT is a segment start that sees nothing.

Neither changes the timing of when the information is available.

### (b) Trainer and inference agree exactly

Inference builds the system block with the same `build_prefix`:
- `pose6/infer/infer.py:106-121` `system_block`, then `assemble` puts the conversation at P;
- `infer/ppm/sysprompt.py` `build_prefix` is byte-identical to the training copy (diffed).

At P - 1 that gives the same footer `\n` on B, present 1, channel A PAD / present 0, no codes, no pose.

Sibling A1's engine (`pose6/infer/ppm/fast_ppm.py`):
- adds g0 at SLOT = P - 1 inside the prefill, with the same op: `e.index_add(0, gpos, self.g0(gf.to(w.dtype)).to(e.dtype))`, `:1358`;
- asserts the footer `\n` / `<|im_end|>` / present flags there (`:1398-1400`);
- gives the cond row and the pose-CFG uncond row the same f (D3).

One difference does not touch P - 1: the inference voice clip uses a blank (PAD) lane, while training clips may keep their lane (sibling-earlier rule).

### (c) Is there a cleaner equivalent? Not implemented, needs pose6's OK

| option | what it is | verdict |
|---|---|---|
| (i) dedicated PAD SLOT | insert one PAD-lane, codes-free, target-free position between the footer and frame 0 (P' = P + 1, SLOT = P) | the only closer analog: the SLOT token becomes PAD, as in the base; it would still not be a segment start |
| (ii) position 0 (`<\|im_start\|>`) | gives the segment-start property | the SLOT -> frame-0 distance would vary with P (5..2000): worse |

Cost of option (i):
- **Layout / recipe:** every conversation position shifts by 1. `tok_ub` bounds grow by 1, so the 64-row cut and `ALOE_ROWS_DIGEST 3681cc26ea97cad7` may change. The micros are no longer mono8020's, so the "recipe copied exactly" claim is lost.
- **Inference:** `infer.py assemble` / `system_block` and the engine's P handling need +1, on BOTH the FT engine and the playground worker. Existing FT checkpoints (the export-357 lineage) would then need the old layout, so there would be two layouts to keep apart.

Recommendation: keep D1 (P - 1).

## f per Aloe item

f comes from B2's sidecar `pose6/g0aloe/g0aloe.parquet`, keyed (store, row, swap), with item_key = row_key|b_side:
- all 28,048 items are covered (27,517 train + 531 eval);
- 0 have frame0_ok false;
- items with chunk_index > 0 use the cut-frame-0 anchor (B2's decision (a)).

The data code asserts, for every item:
- the record exists;
- its row_key / b_side / item_key match the store row;
- every G0 field is present and the frame-0 source is present;
- frame0_ok true implies usable values;
- the stored f equals `g0_feat(fields)` byte-exactly.

f = 0 only where frame0_ok is false (the base's rule), and that case is logged. At startup the trainer asserts coverage of every plan item, train and eval.

## Init (D5)

`QQA_INIT` takes either of two formats:
- a ppm-local-bf16-shards-v1 dir (`meta.json`; mono8020's `local-11488`, FT node-local exports);
- a consolidated bf16 DCP export (`.metadata` + COMPLETE). That is 31101's `/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-<step>` (PPM_EXPORT_EVERY_MIN 360, PPM_EXPORT_KEEP 0).

Every model tensor is loaded exactly, including s_emb / g0. The new params are never re-initialised. Any failure here is loud, and the full rule set is in the trainer docstring.

For a DCP base with a `train_cfg.json`, the recorded s2p mode must equal `PPM_S2P`.

The optimizer state starts fresh, as in mono8020.

## Open issues for pose6 / main

1. **LINK OFF in the launch env.** `QQA_DECP_BB_OFF` is ported and proven; pose6 has escalated it with the recommendation OFF (= the base). If it is set, the resolved-cfg diff gains `decp_bb_off`.
2. **Inference gap (A1).** The engine's local-copy path (`pose6/infer/ppm/fast_ppm.py:171`, `ppm_io.decp_bb_alpha_of`) takes alpha from `cfg['decp_bb_alpha']` and ignores `cfg['decp_bb_off']`. So a link-off local copy (an FT node-local export, or an 811 `ckpt_local_export` of a link-off run) would run the link at alpha 1.0.
   - The DCP path reads `train_cfg.json` and is right.
   - The FT's node-local `meta.json` now also carries `side.decp_bb_alpha 0` / `decp_bb_off 1`, but the engine does not read `side` yet.
3. **31101's link-off point.** The base's `decp_bb` weights are frozen from the link-off point onward (10-20), so they are stale if the link is ever re-enabled.
