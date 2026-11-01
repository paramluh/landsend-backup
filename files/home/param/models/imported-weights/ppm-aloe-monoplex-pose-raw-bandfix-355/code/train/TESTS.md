# pose6/code tests (workstream B1): ALL PASS

The proofs ran as job **31133**: slinky-4, CPU only, 2 cores, 16 GB, 10-27 12:31-12:50 PT. The earlier jobs 31121 / 31124 / 31128 failed only at stage A's item selection (no short item has a voice clip); stage D passed in each of them.
- Runner: `tests/run_tests.sbatch`. Log: `tests/logs/run_tests-31133.log`. JSON: `tests/out/stage*.json`.
- The code under test is pinned in `CODE_MD5.txt` (also listed at the end of this file).
- The base for every comparison:
  - the mono8020 reference = `aloe-pose-mono-8020/code`, the export-357 run's code;
  - the base model = `pose5/code-811-s2p`, job 31101's code.

## Test setup

**Model.**
- Tiny shapes, as in the 811 tests: backbone 2 x H128, towers 2 x H64, decoders 1 x H64, real vocab 151,936 / 151,682.
- bf16 compute.
- Stage B: FSDP2 on a CPU world-1 gloo mesh (fp32 masters, bf16 params, fp32 reduce, the production wrapping).
- Stage D: FSDP2 at world 3 (save) and world 2 (load).

**Weights.** Every tensor is seeded per NAME, so every code variant holds bit-identical shared tensors.

**CPU replacements.** The GPU kernels (fused rms / add_rms / qk_prep / silu_mul, FA3 varlen, bf16 CE GEMMs, cuda autocast, pinned memory) are replaced by torch reference code in the test process only. Every variant of every comparison runs the same replacements.

**Micros.** These are REAL Aloe micros, built by each code dir's own data path (`Stores.conv` -> `pack_micro`, the StepDS path) with the mono8020 env. They are the shortest eval item of each prompt type:

| type | item (store, row, swap) | n frames | P | positions |
|---|---|---|---|---|
| EMPTY | (6, 1646, 1) `9fe89449.../seg00/c01` | 153 | 5 | 160 |
| CLIP | (0, 698, 0) `01a4adf4.../seg00/c00` | 7500 | 70 | 7572 |
| PERSONA | (6, 1646, 0) `9fe89449.../seg00/c01` | 153 | 785 | 940 |
| BOTH | (3, 336, 0) `036aa116.../seg00/c00` | 7500 | 212 | 7714 |

How they are packed:
- **Packed micro 1:** EMPTY + CLIP + PERSONA, 8,672 positions, g0pos [4, 229, 8516].
- **Packed micro 2:** BOTH, g0pos [211].
- Each g0pos equals the item's packed offset + P - 1 (asserted).

## Stage A: data path and SLOT (`stageA_select.json`, `stageAB_compare.json`)

| check | result |
|---|---|
| flags off: pose6 micros vs mono8020's data code, every key and array | **byte-identical** |
| flags on minus {g0pos, g0f} vs flags off | **byte-identical** (G0COND adds only those 2 keys) |
| g0f vs the sidecar's stored f (B2's request), on 28 real micros (4 above + 12 random eval + 12 random train) | **byte-exact** (float32 bytes), g0pos = [P - 1] |
| sidecar coverage of every train / eval item (`splits_v2` = the plan's items) | **27,517 / 27,517 train, 531 / 531 eval; f = 0 on 0 items** |
| an item missing from the sidecar | **fails loud** (AssertionError) |

**What sits at the SLOT = P - 1 (D1), on real items of all four prompt types:**

| | EMPTY | CLIP | PERSONA | BOTH |
|---|---|---|---|---|
| lane B [P-3 .. P+1] | 198, 151645, **198**, 151670, 151670 | 151679 `<eosb>`, 151645, **198**, PAD, PAD | 151681 `<eop>`, 151645, **198**, PAD, PAD | 151681, 151645, **198**, PAD, PAD |
| channel A at P - 1 | PAD, present 0 | same | same | same |
| present_B at P - 1 | 1 | 1 | 1 | 1 |
| codes at P - 1 (any channel) / pose | none / none | none / none | none / none | none / none |
| text target at P - 1 | no (the first target is at P) | no | no | no |
| speech codes in the system block | 0 | 61 (voice clip) | 0 | 74 |
| first conversation speech code = first pose label | P + 2 | P + 2 | P + 2 | P + 2 |
| **first pose label - SLOT** | **3** | **3** | **3** | **3** |

The last row is 3 = 1 + CD = the base's gpos -> frame-0 pose-label offset (`lib/pose_s3q.py:174,207-208`).

## Stage B: the trainer (`stageB_*.json`, `stageAB_compare.json`)

**What each variant runs.** One full optimizer step of each code dir's own trainer:
- prep with tokens_step normalisers;
- `train_step_multi`, 2 rounds: pass A forward, the joint K = 8 nanostep updates of BOTH decoders with their own AdamW, then pass B forward + backward;
- `make_opt`;
- the per-group clip;
- the main AdamW step at the groups' peak LRs;
- both exports.

**Variants:**

| variant | code | flags | new params |
|---|---|---|---|
| mono | mono8020 | n/a | n/a |
| off | pose6 | all off | n/a |
| on0 | pose6 | `PPM_S2P=cb0 PPM_G0COND=1` | zero |
| onnz | pose6 | same as on0 | non-zero |
| on0bboff | pose6 | on0 + `QQA_DECP_BB_OFF=1` | zero |

### (1) Flags off = mono8020's trainer: PASS

| compared | result |
|---|---|
| resolved CFG | identical (0 keys differ) |
| forward terms | identical |
| stats vector + loss (float.hex) | identical (loss 206.264618) |
| EVERY parameter gradient (sha256) | identical |
| decoders after their nanosteps | identical |
| per-group clip norms | identical |
| every parameter after the AdamW step | identical |
| export `train_cfg.json` (sans cfg) + node-local `meta.json` (sans save_time) | identical content; no `side` key |

### (5) Zero new params = flags off: PASS

On every pre-existing tensor, all of these are bit-identical: forward terms, stats, loss, gradients, decoders, the clip norms of the 7 old groups, and every parameter after the AdamW step. Every other update stays unchanged because each new group has its own clip norm (D4).

The new params themselves:

| param | grad norm | own group (LR = JOIN group's) | after the step |
|---|---|---|---|
| `inp.g0.weight` | 0.2128 | `g0cond` (LR = audio's) | non-zero (abs max 6.0e-5) |
| `tower_p.s_emb.weight` | 0.9881 | `s2p` (LR = tower_p's) | non-zero (abs max 1.5e-5) |

Resolved-cfg diff, off -> on, is exactly 7 keys: `s2p`, `g0cond`, `g0_sidecar`, `lr.s2p`, `lr.g0cond`, `clip.s2p`, `clip.g0cond`. The lr / clip values equal tower_p's / audio's.

### (4) The new params get gradients only through their own path: PASS

**s_emb.** Its gradient rows are a subset of the cb0 codes B's depth carries in the step: 1,650 non-zero rows = 1,650 distinct codes present.

**inp.g0 (hooks, both rounds).**
- Its output gradient equals the trunk-input gradient at the K SLOT positions, **bitwise** (index_add's backward).
- dW = sum_k dy_k (x) f_k, with max abs diff 6.0e-5 on max |dW| 0.0285 (rel 2.1e-3, bf16 GEMM rounding).
- So no other position feeds it.

**Speech / text rows** (811's real speech / gemma / chat micros through the pose6 model, flags on, non-zero new params):
- both new params get **grad None**;
- the pose tower is not run;
- the trunk gets gradient (emb grad abs max 6.75 / 4.0 / 3.86).

**Every FT micro is a pose micro** (counted over the mono8020 plan, from its own log `aloe-pose-mono-8020/logs/posemono8020-24229.log`):

| | count | kind |
|---|---|---|
| plan a5d995bcec45dc6f | 27,517 train clips -> 22,825 rows -> 357 steps x 64 | |
| train micros at world 64 | 22,848 = 22,825 rows + 23 dummies | all `pose` |
| eval micros | 529 packed (dummy-padded to 9 per rank) | all `pose` |
| train micros at world 8 (pose-s2p dry plan, the same rows digest 3681cc26ea97cad7) | 26,192, of which 212 dummies | all `pose` |

Why every micro is `pose`:
- The trainer asserts `QQA_POSE_B=1` and `QQA_A_EMPTY=1`.
- Every micro comes from `pack_micro`, which sets kind `pose`.
- `prep` asserts `kind == "pose"` on every train and eval micro.

So no speech or text row exists in the FT. Dummies carry zero loss weights, and the trainer asserts their loss == 0. They add nothing to any gradient, the new params included.

### Non-zero new params (onnz)

Forward terms differ from flags off: text, cb0, hv_s, hv_p, pcb0 (pcb0 4.064394 -> 4.065927). Both mods act.

### LINK OFF (`QQA_DECP_BB_OFF=1`, port of 811's `PPM_DECP_BB_OFF`): PASS

| check | result |
|---|---|
| zb_p computed | no |
| `decp_bb.weight` / `.bias` grad | None |
| decp_bb after the AdamW step | unchanged |
| group `decp_bb` built | no |
| cfg diff vs on | `decp_bb_off` only |
| export `train_cfg.json` side | `decp_bb_alpha 0.0, decp_bb_off 1, s2p cb0, g0cond 1` |
| node-local `meta.json` `side` | the same |
| flag unset | `decp_bb_alpha 1.0`, no `decp_bb_off` |
| unset knob vs mono8020 | identical (B(1) above) |

(At tiny init the link's zb is exactly 0 in this seeded model, so the step's forward terms equal on0's. The link-off path is shown below against 811's link-off model.)

## Stage C: parity with the base's own model (`stageC_parity.json`): PASS

**Keys.** At REAL size (meta device), the FT model's state-dict keys and shapes **equal** the 811 model's (31101 config: decp_bb, no pose head, s2p cb0, g0cond): 2,152 keys, `inp.g0.weight` (5120, 8), `tower_p.s_emb.weight` (2048, 2560).
- Against the real export `ckpt-weka/ppm-v1/w-15973` (bf16 DCP, pre-mods): the FT model needs exactly those 2 keys more, and no other key differs; shapes are equal on every common key.
- So a 31101 export, which carries both keys, covers the FT model exactly.

**Loss parity.** The 811 `model_ppm` (job 31101's) vs pose6's, plain modules, the same bf16 weights with NON-ZERO s_emb and g0. Backward = lw_word x text + cb0 + pose cb0, plus fixed seeded upstream gradients into hv_s / hv_p / zb_p (the nanosteps' feedback):

| micro | positions | SLOTs | forward terms max abs | all 128 grads max abs | text | cb0 | pose cb0 | new-param grad norms (g0, s_emb) |
|---|---|---|---|---|---|---|---|---|
| Aloe packed (EMPTY+CLIP+PERSONA) | 8,672 | 3 | **0.0** | **0.0** | 20.224012 | 7.781498 | 7.972465 | 0.585, 15.00 |
| Aloe EMPTY | 160 | 1 | **0.0** | **0.0** | 11.937158 | 7.799590 | 7.955291 | 4.248, 3.787 |
| Aloe CLIP | 7,572 | 1 | **0.0** | **0.0** | 20.240778 | 7.780118 | 7.976628 | 0.198, 14.63 |
| Aloe PERSONA | 940 | 1 | **0.0** | **0.0** | 27.683746 | 7.831087 | 7.785592 | 0.338, 2.875 |
| Aloe BOTH | 7,714 | 1 | **0.0** | **0.0** | 17.830391 | 7.897294 | 7.860975 | 0.160, 16.20 |
| TikTok 811 pose micro (real, `pose5/g0-test-micros/pose.pkl`) | 7,497 | 23 | **0.0** | **0.0** | 25.193045 | 7.766720 | 7.707139 | 2.001, 15.76 |

The 811 and pose6 values are bit-identical in every row (forward terms and grads both 0.0), so each value column is shown once.

**LINK OFF parity** (811 `decp_bb_on False` = 31101's path vs pose6's QQA_DECP_BB_OFF path):

| micro | zb computed (811, pose6) | hv_p | pose-decoder CE | decp_bb grad | all grads |
|---|---|---|---|---|---|
| Aloe packed | no, no | bit-identical | 7.795907 = 7.795907 | None on both | identical |
| TikTok 811 | no, no | bit-identical | 7.780424 = 7.780424 | None on both | identical |

## Stage D: init (D5), FSDP resharding (`stageD_init.json`): PASS

**Setup.**
- The base is built by the BASE's code (`code-811-s2p` `model_ppm.build`, FSDP2, world 3, 31101 config, every tensor non-zero).
- It is written by the export code paths:
  - DCP = `safety_ppm.export_weights`' path, the `w-<step>` format, with `train_cfg.json`;
  - local shards = the `ckpt_local_export` format.
- The pose6 trainer's `load_init` then loads it into a world-2 FSDP2 model (resharding 3 -> 2).
- Every LOCAL shard is compared bitwise with the expected values (the seeded full tensor -> bf16 -> fp32, sliced to the rank's rows). Both ranks agree.

| case | result |
|---|---|
| flags on <- DCP / <- local shards | 128 / 128 tensors exact on both ranks, incl. `inp.g0.weight` and `tower_p.s_emb.weight` (non-zero) |
| flags on <- base without the new keys (DCP / local) | FAILS loud (D5 message) |
| + `ALOE_TEST_NEWPARAM_RANDOM=7` | new params = the seeded full tensor's slice (world-invariant, non-zero); all others exact |
| test override with the keys present | FAILS loud ("unset it") |
| flags off <- base WITH the new keys (DCP / local) | FAILS loud (never dropped silently) |
| flags off <- pre-mods base (mono8020's path) | 126 / 126 exact |

Not tested on real weights here: the load of a real 31101 export. The first one is due at ~17:40 PT (PPM_EXPORT_EVERY_MIN 360 from 11:41). Stage C shows its key set and shapes cover the FT model, and stage D proves the DCP path.

## Code md5s (= `CODE_MD5.txt`; the files job 31133 ran)

```
47a55ed52ae3032b6da0f5c82a439796  data_aloe_pose.py
ec50020ea7dce8565c05a395ef541b59  data_ppm.py
7510d28481d52ea4c9c0958a53b56f2b  data_qqa.py
df67969167470e7399b7b5f3fb2ce6df  model_ppm.py
212b1ee9dc4e0532f37121c940111b6c  train_aloe_pose.py
1024fab9f3304a1b519307902199240f  optoff.py
aadecad76fe3fcf8893cd0a88fae1758  task_ppm.sh
786f9d6874918781da5b4df5d1a24071  lib/pose_s3q.py
c47b9d1c3e1646562715c8f2c95c0328  lib/sysprompt.py
1a0699bfa27f5ddfd4aaefcc4b755aca  lib/heads_s2q.py
ed6360cd1a224e343ba94d870a89d955  lib/speech_q.py
6abfd45288cccab9a20a85f66ed1040f  vendor/fused_triton.py
985aa1cc843752d951d3411d4787edad  ../g0aloe/g0aloe.parquet (B2 sidecar)
```
The full list (lib, vendor, tests) is in `CODE_MD5.txt`. Diffs vs pose-s2p and vs mono8020 are in `diffs/`.
