# pose6/infer tests

The design is in `README.md`. Raw results are in `tests/out/*.json` and `tests/logs/`.

## Summary

| test | where | result |
|---|---|---|
| S: inference SLOT layout, start()'s G0COND checks | CPU (jobs 31120, 31123) | PASS |
| SLOT in the FT trainer's real micros (528 Aloe eval items) | CPU (job 31120) | PASS |
| L1: load path, both checkpoint formats, exact tensors, fail-loud cases | CPU (jobs 31120, 31123) | PASS |
| LK: link OFF (decp_bb_off) gives alpha 0, and slot 0 = proj(hv_p) bit-exact | CPU (job 31123) | PASS |
| L2: real key sets (w-15973, export-357, the predicted 31101 export) | CPU (jobs 31120, 31123) | PASS |
| (i) flags off == deployed engine; (ii) zero new params == deployed engine | GPU, deployed layout (job 31140) | **PASS** (bit-identical, floor 0) |
| (w) G0COND wiring (locality, uncond same f, f = 0 == off, fail loud) | GPU (job 31140) | **PASS** |
| (iii) engine vs training forward, non-zero new params, + floor | GPU (job 31140) | at the bf16 floor on every stream |
| (iii-strong) same at 10x / 30x scale: effect large, parity at the floor; (loc) the clip uncond hn outlier localised | GPU (steps in 31150) | **PASS**: effect 14-60x parity (max abs); the outlier is an ill-conditioned point, not a mismatch |
| (iv) latency, today / s2p cb0 / s2p cb0 + G0COND | GPU (job 31140) | 61.0 / 66.2 / 66.3 ms (free, graphs) |

Engine versions under test:
- `ppm/fast_ppm.py` md5 9ebe66985bf7f02fb26934f2b05fa16f: CPU job 31123 and GPU job 31140.
- **Final** md5 25160cdf3a815987255bb4959aa5d7f0. The only change since is that `_link_alpha` also reads a nested `cfg` record (enginefix parity), which affects only records carrying `decp_bb_off`; export-357 carries none. Every CPU check (S, L1, LK, L2) was re-run on this version in job 31148 and all PASS.

## CPU checks

### S: inference SLOT (`tests/cpu_load.py` S; `infer.system_block` + `assemble`, as genweb worker.py builds the prompt)

| prompt | P | ids_B[P-2:P] | ids_A[P-1] | present[:, P-1] | Mimi codes at P-1 | pose codes at P-1 | voice-clip code positions | `_g0_inst` -> (SLOT, f) |
|---|---|---|---|---|---|---|---|---|
| empty block | 5 | 151645, 198 | PAD | 0, 1 | 0 | 0 | 0 | (4, f) |
| voice clip (10 s) | 134 | 151645, 198 | PAD | 0, 1 | 0 | 0 | 125 | (133, f) |
| persona | 26 | 151645, 198 | PAD | 0, 1 | 0 | 0 | 0 | (25, f) |
| clip + persona | 155 | 151645, 198 | PAD | 0, 1 | 0 | 0 | 125 | (154, f) |

start()'s fail-loud checks (`FastPPM._g0_inst`), all raised as designed:

| case | raised |
|---|---|
| G0COND engine, no g0f | `ValueError` |
| G0COND engine, g0f None | `ValueError` |
| shape (7,) | `ValueError` |
| NaN | `ValueError` |
| int64 vector | `ValueError` |
| P-1 not `'\n'` | `AssertionError` |
| engine without G0COND, g0f given | `ValueError` |

An engine without G0COND and no g0f returns None (today's path). A float64 f is cast to float32.

### SLOT in the FT trainer (`tests/cpu_slot_train.py`, pose-s2p/code data path, `aloe_mono8020.env`, job 31120)

**Coverage.** Every Aloe eval item with n >= 320 frames was checked: 528 items. On all 528:
- ids_B[P-2:P] = (<|im_end|>, '\n');
- present = (0, 1) at P-1;
- no Mimi or pose codes at P-1;
- the first pose label is at P + 2.

**Parity micros** saved to `tests/out/micros/`, one per prompt type:

| type | item (store, row, swap) | row_key | P | n_frames | clip code positions |
|---|---|---|---|---|---|
| empty | [4, 1772, 0] | ALOE/s2handover-pose/b36af357-.../seg00/c00 | 5 | 776 | 0 |
| clip | [0, 698, 0] | ALOE/s2handover-pose/01a4adf4-.../seg00/c00 | 70 | 7500 | 61 |
| persona | [0, 2203, 0] | ALOE/s2handover-pose/fadd1e80-.../seg00/c01 | 929 | 356 | 0 |
| clip + persona | [3, 336, 0] | ALOE/s2handover-pose/036aa116-.../seg00/c00 | 212 | 7500 | 74 |

B1's port (`pose6/code/data_aloe_pose.py:724-737`) puts g0pos at offset + P - 1 and asserts the same footer, which matches the engine.

### L1: load path on synthetic checkpoints (tiny widths, every loader line as deployed; node /scratch only)

The synthetic checkpoint is a PPM with pose, the D33 link, `tower_p.s_emb` cb0 and `inp.g0`, with seeded random bf16 weights. Two formats:
- the 8,1,1 export format: `dcp.save` + `train_cfg.json` side `{decp_bb, decp_bb_alpha 0.5, s2p cb0, g0cond 1}`, plus an unused pose_head key;
- `ppm-local-bf16-shards-v1`: 4 dim-0 rank shards + `rep.pt` decoders + a meta cfg carrying s2p / g0cond.

| case | dcp | local |
|---|---|---|
| build_model: every tensor bit-equal to saved (incl. s_emb, inp.g0); binfo g0cond True, s2p cb0, alpha 0.5; inp.g0 has no bias | PASS (0 unequal, 0 missing) | PASS |
| `PPM_G0COND=0` with the key | AssertionError | AssertionError |
| side records g0cond 1, key absent | AssertionError | (n/a) |
| no new keys, env unset | no module, no s_emb, all tensors equal | same |
| no new keys, `PPM_G0COND=1` | inp.g0 = 0 (`g0cond_zero_init`), only missing key = inp.g0.weight | same |

### LK: link OFF, as 31101 runs (`PPM_DECP_BB_OFF=1`; job 31123)

| record | engine alpha |
|---|---|
| DCP side `{decp_bb_alpha 0.0, decp_bb_off 1}` (the 31101 export side) | 0.0 |
| DCP side `{decp_bb_alpha 1.0, decp_bb_off 1}` (FT export side as written today + off) | 0.0 |
| DCP side `{decp_bb_alpha 0.7}`, no off (unchanged path) | 0.7 |
| local cfg `{decp_bb_alpha 1.0, decp_bb_off 1}` (local copy of a 31101 save / FT link-off local export) | 0.0. **The deployed and pose-s2p engines give 1.0 here: the bug** (`pose6/enginefix`). |
| local cfg `{decp_bb_alpha 1.0}`, no off | 1.0 |

**Slot 0 with alpha 0.** The engine's `zb = decp_bb(hn) * alpha` gives all zeros and all finite values. `FusedDec.slot0(hv, zb)` is bit-identical to `slot0(hv, None)` and to the trainer's `proj(hv)` (the zb=None graph): **0 differing bit patterns**. There is no NaN path, because the weights are asserted finite at load and hn is RMS-normed. With alpha 1 the link would add up to 3.7 here.

### L2: real checkpoints' key sets (job 31123)

| checkpoint | engine model keys missing (engine with s2p cb0 + G0COND) | shape mismatches | extra (unused) |
|---|---|---|---|
| `ckpt-weka/ppm-v1/w-15973` (8,1,1 consolidated DCP; side `{decp_bb 1, alpha 1.0}`; 2,150 keys) | exactly `inp.g0.weight`, `tower_p.s_emb.weight` | 0 | none |
| `aloe-pose-mono-8020/exports/export-357` (local, world 64) | exactly `inp.g0.weight`, `tower_p.s_emb.weight` | 0 | none |

**Predicted 31101 export.** This is the `code-811-s2p` model with `s2p=cb0, g0cond=True` on meta, i.e. the export's `get_model_state_dict`. Compared with w-15973, the new keys are exactly `inp.g0.weight` [5120, 8] and `tower_p.s_emb.weight` [2048, 2560]. No key is removed and no shape changes. The engine needs no key outside that set and every shape matches.

**Conclusion.** The engine loads 31101's exports as they are written, and the first real export is the final check.

## GPU proofs (deployed layout: 2 GPUs of one node, ONE process, export-357 loaded once; job 31140 on slinky-4 GPUs 4,5)

The guard passed: physical GPUs 4 and 5 ran a 25 s bf16 burn at 682 and 690 TFLOPS (median SM 1380 and 1410 MHz, power-capped), with no thermal reason.

Job 31140 ran 12:33 to 13:01 PT. Results are in `tests/out/engine6_results_31140.json` and the log in `tests/logs/gpu-31140.log`; the tables below are rendered by `tests/report6.py`.

**Verdicts**
- **(i) and (ii) PASS.** All 6 mode × scenario cells are bit-identical to the deployed engine, and so is the orig-vs-orig floor:
  - flags off;
  - s2p cb0 at zero;
  - s2p cb0 + G0COND at zero with a non-zero f.
- **(w) PASS.**
- **(iii):** engine-vs-trainer parity with non-zero s_emb and inp.g0 is at the bf16 floor on every stream, for the cond row and for the pose-CFG uncond row (details below the tables).
- **(iv):** G0COND adds nothing per frame. Its cost is one Linear in the prefill.

### (i) flags off and (ii) zero new params vs the DEPLOYED engine (one process, orig-vs-orig floor first)

| mode / scenario | frames | floor orig vs orig | (i) orig vs off | (ii) orig vs s2p cb0 (SE = 0) | (ii) orig vs s2p cb0 + G0COND (SE = 0, G0 = 0, f = F_A) | g0 rows applied | pass |
|---|---|---|---|---|---|---|---|
| eager/free | 50 | 0 (all identical) | 0 (all identical) | (graphs only) | 0 (all identical) | [(0, 133), (1, 133)] | True |
| eager/free_cfg | 50 | 0 (all identical) | 0 (all identical) | (graphs only) | 0 (all identical) | [(0, 133), (1, 133)] | True |
| eager/poseonly | 50 | 0 (all identical) | 0 (all identical) | (graphs only) | 0 (all identical) | [(0, 133), (1, 133)] | True |
| graphs/free | 150 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | [(0, 133), (1, 133)] | True |
| graphs/free_cfg | 150 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | [(0, 133), (1, 133)] | True |
| graphs/poseonly | 149 | 0 (all identical) | 0 (all identical) | 0 (all identical) | 0 (all identical) | [(0, 133), (1, 133)] | True |

(i)/(ii) all pass: **True**. Compared per frame: B lane, speech codes, pose codes (readback) + hv_p, pose cb0 logits (both rows), hv_s, speech cb0 logits, hn (both rows); 0 = bit-identical.

### non-zero parameter scales (used in (w), (iii), (iv))

- inp.g0 ~ N(0, sigma^2), sigma = 0.06: chosen so ||g0(F_A)|| = the median per-frame ||inp.pose(pose codes)|| of Person 01's opening = 6.79 (p10-p90 6.49-7.22); realised ||g0(F_A)|| = 6.77, ||g0(F_B)|| = 13.9 (||F_A|| 1.58, ||F_B|| 3.23); for scale: inp.pose row norm median 0.983, emb_B('\n') row norm 2.05.
- tower_p.s_emb ~ N(0, (0.02 / sqrt 32)^2) (= pose-s2p, the tower code_emb init scale): row norm median 0.179.
- F_A = [0.5765, 0.5305, -0.2011, -0.7049, 0.5178, -0.6487, 0.574, 0.5754] (Person 01 opening frame 0 root / width, 9:16), F_B = [0.42, 0.61, -1.0498, -1.2, 1.6571, -1.7429, 1.1143, -0.5754].

### (w) G0COND wiring (non-zero G0, SE = 0)

- w1 prefill locality (P = 134, SLOT = 133), new vs off engine, trunk KV:
  - row0/layer0: K differs at 1 positions, first [133]; V at 1, first [133]
  - row0/last: K differs at 1 positions, first [133]; V at 1, first [133]
  - row1/layer0: K differs at 1 positions, first [133]; V at 1, first [133]
  - row1/last: K differs at 1 positions, first [133]; V at 1, first [133]
  - uncond row's layer-0 K/V at the SLOT == the cond row's (same f): True; g0 rows applied [(0, 133), (1, 133)]; hn_prev change cond 0.734, uncond 1.5; pass True
- w2_f_zero_equals_off/eager: 0 (all identical); pass True
- w2_f_zero_equals_off/graphs: 0 (all identical); pass True
- w3 f matters (graphs, free, 60 frames): F_A vs F_B lane_b 13; dep_b 1420; pdep 1854; hv_p 7.23; zp0 18.8; hv_s 33.2; z0 25; hn_prev 70.2; F_A vs off lane_b 24; dep_b 1416; pdep 1789; hv_p 9.66; zp0 20.7; hv_s 9.38; z0 25.5; hn_prev 92.4; F_A repeat floor 0 (all identical); frame-0 hn change vs off 1.42; pass True
- w4 fail loud on the real engines: new_without_g0f: ValueError: G0COND model: every instance needs g0f (np.float32 [8] frame-0 features; pose6/infer/g0src.py); off_with_g0f: ValueError: instance carries g0f but this engine has no G0COND (model without inp.g0): refusing to ignore it; s2p_only_with_g0f: ValueError: instance carries g0f but this engine has no G0COND (model without inp.g0): refusing to ignore it; orig_ignores_g0f_key (deployed engine, documented): no exception; new_slot_not_newline: AssertionError: ("G0COND SLOT = P - 1 must be the system block's codes-free footer '\\n' on channel B", 134, 151670); pass True

(w) all pass: **True**

### (iii) engine vs the TRAINING forward, teacher-forced from conversation frame 0 (cond row + pose-CFG uncond row)

Training model: `/data/home/amartyavarma/ppm/pose6/code/model_ppm.py` (B1 port: True; G0COND in the training class: True).

| params / micro / mode | P | frames | text top1 / TV mean / max abs | speech cb0 top1 / max abs | pose cb0 cond top1 / TV / max abs | pose cb0 UNCOND top1 / TV / max abs | hn cond / uncond max abs (ref absmax) | hv_p / zb max abs | pose dec e2e top1 / max abs (same-input floor) |
|---|---|---|---|---|---|---|---|---|---|
| zero_new_params_floor/clip/graphs | 70 | 300 | 1.0 / 4.00e-03 / 0.375 | 0.99667 / 0.219 | 0.99667 / 5.14e-03 / 0.125 | 0.99333 / 4.82e-03 / 0.125 | 1 / 1 (109.5) | 0.125 / 1 | 0.96371 / 0.75 (0.96371 / 0.789) |
| zero_new_params_floor/persona/graphs | 929 | 300 | 0.99667 / 4.62e-03 / 1.25 | 0.99667 / 0.312 | 1.0 / 2.75e-03 / 0.25 | 0.99667 / 2.77e-03 / 0.25 | 2.5 / 2.12 (95.0) | 0.188 / 1 | 0.97177 / 1.88 (0.97446 / 1.75) |
| zero_new_params_floor/empty/graphs | 5 | 300 | 1.0 / 3.61e-03 / 0.625 | 0.99333 / 0.125 | 0.99333 / 7.94e-03 / 0.219 | 0.99333 / 9.62e-03 / 0.188 | 1.14 / 1 (92.5) | 0.125 / 1 | 0.95565 / 0.812 (0.96237 / 0.758) |
| zero_new_params_floor/clip+persona/graphs | 212 | 300 | 0.99333 / 3.49e-03 / 0.502 | 0.99333 / 0.172 | 0.99 / 6.29e-03 / 0.125 | 0.99333 / 6.85e-03 / 0.125 | 1 / 1 (90.5) | 0.125 / 1 | 0.96102 / 1.5 (0.95968 / 1.5) |
| nonzero_new_params/clip/graphs | 70 | 300 | 0.99667 / 3.82e-03 / 0.625 | 0.99333 / 0.125 | 0.99667 / 4.39e-03 / 0.125 | 0.99333 / 5.05e-03 / 0.125 | 1 / 6.5 (109.5) | 0.188 / 1 | 0.95833 / 0.75 (0.96505 / 0.688) |
| nonzero_new_params/clip/eager | 70 | 100 | 1.0 / 2.64e-03 / 0.375 | 0.99 / 0.125 | 1.0 / 6.12e-03 / 0.125 | 0.99 / 6.19e-03 / 0.125 | 1 / 5.25 (91.0) | 0.188 / 0.5 | 0.97715 / 0.562 (0.97715 / 0.5) |
| nonzero_new_params/persona/graphs | 929 | 300 | 0.99667 / 4.84e-03 / 1.25 | 1.0 / 0.219 | 0.99333 / 2.35e-03 / 0.141 | 0.99667 / 2.78e-03 / 0.188 | 2.22 / 1 (93.5) | 0.156 / 0.75 | 0.97312 / 1.5 (0.97177 / 1.56) |
| nonzero_new_params/empty/graphs | 5 | 300 | 0.99667 / 3.12e-03 / 0.5 | 0.99333 / 0.125 | 0.99 / 8.70e-03 / 0.188 | 0.99667 / 9.05e-03 / 0.188 | 1.5 / 1.5 (92.5) | 0.156 / 1 | 0.95699 / 0.938 (0.96774 / 0.719) |
| nonzero_new_params/clip+persona/graphs | 212 | 300 | 1.0 / 2.89e-03 / 0.594 | 0.99667 / 0.156 | 0.99333 / 6.78e-03 / 0.125 | 0.99333 / 6.23e-03 / 0.125 | 1 / 1.5 (90.0) | 0.125 / 1 | 0.95968 / 1.25 (0.96371 / 1.5) |

Effect size of the mods in the TRAINER itself (non-zero vs zero new params, same micro): the parity errors above must be far below these.

| micro / mode | hn max abs | text top1 / TV mean / max abs | pose cb0 top1 / TV mean / max abs | speech cb0 top1 / max abs |
|---|---|---|---|---|
| nonzero_new_params/clip/graphs | 2.12 | 0.99667 / 4.92e-03 / 0.76 | 1.0 / 6.24e-03 / 0.283 | 0.99333 / 1.12 |
| nonzero_new_params/clip/eager | 2.17 | 1.0 / 4.83e-03 / 0.625 | 1.0 / 8.90e-03 / 0.301 | 0.99 / 1.19 |
| nonzero_new_params/persona/graphs | 3.12 | 0.99 / 4.49e-03 / 0.688 | 0.99333 / 3.33e-03 / 0.164 | 0.99667 / 0.188 |
| nonzero_new_params/empty/graphs | 6.58 | 0.99667 / 5.05e-03 / 2.2 | 0.99333 / 1.17e-02 / 0.75 | 0.99 / 2.22 |
| nonzero_new_params/clip+persona/graphs | 1.5 | 0.99333 / 3.21e-03 / 1.25 | 0.98333 / 7.93e-03 / 0.188 | 1.0 / 0.312 |

### (iv) per-frame latency (same 2 GPUs, same process; sync = both GPUs synchronised every frame, median / p90; async = no per-frame sync)

| mode / scenario | today (deployed engine) | s2p cb0 | s2p cb0 + G0COND | start() prefill ms (today / s2p / +G0) |
|---|---|---|---|---|
| graphs/free (302 frames) | sync 60.96 / p90 61.15; async 60.64 | sync 66.18 / p90 66.39; async 65.88 | sync 66.31 / p90 66.52; async 66.0 | 155.4 / 152.8 / 152.9 |
| graphs/free_cfg (302 frames) | sync 64.99 / p90 65.16; async 64.64 | sync 69.99 / p90 70.21; async 69.65 | sync 70.06 / p90 70.26; async 69.74 | 151.8 / 154.1 / 151.5 |
| graphs/poseonly (151 frames) | sync 61.2 / p90 61.34; async 60.6 | sync 66.35 / p90 66.43; async 65.73 | sync 66.47 / p90 66.57; async 65.84 | 156.3 / 151.8 / 150.8 |
| eager/free (32 frames) | sync 584.65 / p90 595.25; async 554.89 | sync 586.83 / p90 607.84; async 547.48 | sync 579.23 / p90 594.46; async 551.3 | 152.0 / 147.3 / 148.2 |
| eager/free_cfg (32 frames) | sync 590.43 / p90 604.42; async 543.26 | sync 580.81 / p90 590.21; async 561.17 | sync 587.39 / p90 594.81; async 560.87 | 155.8 / 149.2 / 189.5 |
| eager/poseonly (32 frames) | sync 588.77 / p90 595.85; async 553.56 | sync 590.19 / p90 601.9; async 564.14 | sync 579.63 / p90 585.93; async 546.39 | 150.6 / 151.0 / 152.3 |


### Reading (iii)

**Training model.** It is B1's `pose6/code/model_ppm.py`: its G0COND port, picked up automatically. The training forward runs piecewise over the 2 GPUs with the training classes:
- `Inp.forward(g0=(gpos, gf))` → `Block` x 64 → the head's final norm (`Head.forward`'s hn);
- `Tower.forward` for speech, then `Tower.forward(sdep=B's codes at T)` for pose;
- `shift1(decp_bb(hn))` → `Decoder.inputs` + `.model`.

These share the engine's weight tensors (no copy). The trainer gpos is P - 1, which is what B1's data code emits (`g0pos = offset + P - 1`).

**Uncond reference.** The training forward on the silence micro, i.e. every Mimi code replaced by digital silence, with the SAME f. The trainer has no uncond rows; README explains why the same f is the rule.

**Parity at the floor.** With non-zero s_emb and inp.g0, every stream sits at the same level as the zero-param floor, which is today's engine-vs-trainer bf16 difference:

| stream | non-zero params | zero-param floor |
|---|---|---|
| text TV mean | 2.6-4.8e-3 | 3.5-4.6e-3 |
| text top-1 | 0.993-1.0 | (same range) |
| speech cb0 max abs | 0.13-0.22 | 0.13-0.31 |
| pose cb0 cond TV mean | 2.4-8.7e-3 | 2.8-7.9e-3 |
| pose cb0 cond max abs | 0.125-0.19 | 0.125-0.25 |
| pose cb0 UNCOND TV mean | 2.8-9.1e-3 | 2.8-9.6e-3 |
| pose cb0 UNCOND max abs | 0.125-0.19 | 0.125-0.25 |
| hv_p max abs | 0.125-0.19 | 0.125-0.19 |
| pose decoder e2e top-1 | 0.956-0.977 | its same-input floor 0.96-0.977 |

**One outlier: hn of the UNCOND row on the "clip" micro with non-zero params.** Its max abs is 6.5 (graphs) and 5.25 (eager), against 1.0 at zero params. hn's absmax is about 109, so this is about 13 bf16 ulps on a single element; the cond row of the same run has 1.0.

It is not a wiring error:
- (w1) shows the uncond row's SLOT input is bit-identical to the cond row's;
- the uncond row's only consumer, the pose cb0 logits of the CFG mix, matches at the floor (max abs 0.125, TV 5.1e-3);
- the other three micros' uncond hn are 1.0-1.5.

Now localised in the **(loc)** section below: an ill-conditioned point (frame 1, position P + 1), not a mismatch.

**Power of the test.** At the realistic scale used here, the mods' own effect on the trainer is small after frame 0:
- hn max change 1.5-6.6;
- text TV 3-5e-3, about 1-1.5x the parity noise;
- pose cb0 max 0.16-0.75.

So (iii) alone is only a weak detector of a mis-wired G0COND. Exact wiring is proven bit-for-bit by (i), (ii), (w1) and (w2):
- KV changes at exactly P - 1, in every layer, for both rows;
- f = 0 is identical to flag off;
- f changes the output, and a repeat of the same f is identical.

(iii) shows there is no extra error beyond the bf16 floor. **(iii-strong)** below closes the power gap: at 10x / 30x scale the effect is 14-60x the parity on max abs, and parity stays at the floor.

### Reading (iv)

**CUDA graphs** (the live / playground path):

| scenario | today | s2p cb0 | s2p cb0 + G0COND |
|---|---|---|---|
| free (sync) | 60.96 ms | 66.18 | 66.31 |
| free_cfg | 64.99 | 69.99 | 70.06 |
| poseonly | 61.20 | 66.35 | 66.47 |

All are within the 80 ms live budget. These match pose-s2p's own measurement (61.0 / 66.3 ms on slinky-10).

**G0COND costs about +0.1 ms**, which is run-to-run noise: the step graphs are the same graphs. Prefill (`start()`) takes about 150-155 ms in every engine.

**Eager** runs at about 550-590 ms/frame in every engine. It is Python- and launch-bound and is not a serving path.

## (iii-strong) + (loc): steps inside pose6's allocation 31150 (slinky-4 GPUs 4,5; guard OK at 682 / 690 TFLOPS)

Two steps ran:
- `iiis`, 13:04-13:08 PT (log `tests/logs/iiis-31150.log`, json `tests/out/engine6_results_31150_iiis.json`);
- `loc2`, 13:08-13:10 PT (log `tests/logs/loc2-31150.log`, json `tests/out/engine6_results_31150.json`).

Setup: the clip micro (P 70, voice clip) and the empty micro (P 5), 150 teacher-forced frames, CUDA graphs, cond row and pose-CFG uncond row. Three scales:
- floor: zero new params;
- realistic: as in (iii);
- strong: g0 sigma x10, so ||g0(F_A)|| = 67.7 instead of 6.77, and s_emb std x30.

**Effect** = the trainer with non-zero vs zero new params. **Parity** = engine vs trainer at the same scale.

### At strong scale, the effect is large and parity stays at the floor

Max abs per stream:

| micro | stream | effect | parity | effect / parity | parity at the zero-param floor |
|---|---|---|---|---|---|
| clip | text logits | 16.8 | 0.5 | 34x | 0.375 |
| clip | speech cb0 logits | 9.81 | 0.37 | 27x | 0.125 |
| clip | pose cb0 cond | 3.34 | 0.125 | 27x | 0.125 |
| clip | pose cb0 UNCOND | 4.89 | 0.125 | 39x | 0.125 |
| clip | hn cond / uncond | 7.75 / 57.0 | 0.56 / 1.0 | 14x / 57x | 1.0 / 1.0 |
| empty | text logits | 16.3 | 1.0 | 16x | 0.75 |
| empty | speech cb0 | 8.64 | 0.145 | 60x | 0.125 |
| empty | pose cb0 cond | 5.59 | 0.125 | 45x | 0.156 |
| empty | pose cb0 UNCOND | 5.38 | 0.125 | 43x | 0.125 |
| empty | hn cond / uncond | 10.1 / 10.1 | 0.92 / 1.0 | 11x / 10x | 1.5 / 1.0 |

TV mean (effect / parity), and parity against the floor's TV:

| stream | clip | empty |
|---|---|---|
| text | 0.0192 / 0.0033 = 5.9x | 0.0203 / 0.0041 = 5.0x |
| speech cb0 | 0.0433 / 0.0062 = 7.0x | 0.0707 / 0.0071 = 10.0x |
| pose cond | 0.0547 / 0.0068 = 8.0x | 0.0973 / 0.0094 = 10.3x |
| pose uncond | 0.0454 / 0.0060 = 7.6x | 0.1098 / 0.0087 = 12.6x |
| parity TV vs floor TV (all streams) | within 0.86-1.23x | same range |

Top-1 effect at strong scale: pose cb0 cond 0.88-0.97 and uncond 0.87-0.95, against parity top-1 0.98-1.0.

**Text is the weakest stream:** its TV ratio is about 5-6x, because one SLOT position changes text little. Its max-abs ratio is still 16-34x.

### (loc) the clip-micro uncond hn outlier

**Where it is.** Uncond row, conversation frame 1, i.e. absolute position P + 1 = 71, channel 1660:
- trainer -34.25, engine -39.5, diff 5.25;
- next largest: channel 2594 (ref 42.75, diff 2.75) and channel 4645 (ref -12.44, diff 2.56), on the same frame;
- the zero-param floor run's diff at that same frame and channel is **0.0** (its frame-1 row max is 0.0625);
- that frame's hn row absmax is **40.75**, against about 12 on every other uncond frame.

**It does not grow with scale.** At strong scale:
- the whole uncond hn parity max is 1.0;
- frame 1 is 0.63;
- the top entries are spread over frames 1, 72, 108 and 109 at ≤ 1.0, with floor diffs of 0-0.375 at the same spots.

**Per frame (loc2)**, uncond row, row-max |diff| on frames 0-11 at realistic scale:

| comparison | frame 0 | frame 1 | frames 2-11 |
|---|---|---|---|
| served step path (graphs) vs trainer | 0.09 | **1.5** (5.25 in the iiis run, whose trainer pass covered L = P + 150 instead of P + 40) | 0.03-0.66 |
| engine prefill path (eager FA3-varlen, the trainer's kernel family) vs trainer | 0.06 | **3.75** | 0.03-0.31 |
| engine step vs engine prefill | 0.06 | **5.25** | 0.06-0.88 |
| trainer vs trainer with f[0] moved by ONE bf16 ulp | 0.06 | **2.81** | 0.04-0.81 |
| cond row: step vs trainer / prefill vs trainer / 1-ulp sensitivity | ≤ 0.125 | ≤ 0.125 | ≤ 0.19 |

At strong scale the same uncond comparisons are ≤ 1.0 at frame 1 and ≤ 0.81 elsewhere.

**Conclusion: an ill-conditioned point, not a mismatch.**
- At this position the trainer's own output moves by 2.8 for a 1-ulp change of one input feature.
- Three valid bf16 evaluations of the same function disagree there by 1.5-5.25, including the trainer against itself at a different sequence length (5.25 vs 1.5).
- One of those three, the engine's prefill path, uses the trainer's FA3-varlen kernels.
- Everywhere else, and in the cond row, all paths agree to ≤ 0.9.
- It disappears when the effect is scaled up 10x.
- The uncond row's pose cb0 logits, the only thing the uncond hn feeds, stay at the floor (max 0.125).

### The trainer-side uncond reference (checked in the run, `iiis/*/uncond_reference`)

- Channel B's depth is replaced by `silence_depth`, with the frame index restarting per coded run, at exactly:
  - the voice clip's coded positions in the system block: 61 for clip, 0 for empty;
  - every conversation position from P + CD = P + 2 on: 148 of 150.
- Those conversation silence codes are **equal** to the engine's per-frame uncond rule `sil_t[T - P - CD]` (array-equal).
- Channel A is unchanged (absent); ids, present and pdepth are unchanged.
- g0 is added at the same SLOT P - 1 with the same f (F_A) as the cond row. In the engine, the cond row and the uncond row get bit-identical layer-0 K/V at the SLOT (w1).
