# ZOOM_POSE: pose-duplex FT + zoom text co-train (trainer side), 10-30

User-approved (10-29 ~23:01 PT): job 30074's pose-duplex recipe plus zoom-strangers-5k text co-train, 1:1 per step, on 8 nodes. 4 Aloe hosts and 4 zoom hosts, 2 slots per GPU, logical world 64 per group.

This note covers only the trainer, `code/train_samsep_posedup.py`. No other code file changed (`model_ppm.py` and `data_zoom.py` are untouched). The data change (raw codes) and the env, sbatch and launcher belong to the parent and other helpers.

## Why a change was needed
The trainer refused the combination. Line 258 of the pre-edit file asserted `not POSE or ALOE_ONLY`. Even without that assert, three things would have failed on a zoom micro:
- The zoom micro is `kind 'speech'` with no `pdepth`, `pcb0` or `plight`, and `prep` asserts `kind == 'pose'`.
- `own` has 3 columns, and the step loop asserts `NOWN = 5`.
- The zoom branch of `run_micro` never ran `tower_p`, `decp_bb` or the `dec_p` collective.

So zoom ranks had never run the pose parts, and an FSDP and collective mismatch would have hung or corrupted the step.

## What changed (line numbers are in the edited file)
| where | change |
|---|---|
| L1-7 | Module doc: a ZOOMPOSE header. |
| L235 | Cotrain mode with pose only: the `cotrain.zoom_pose` description key, which is self-describing. It falls under `cotrain.*`, so the 26232 diff allows it. |
| L262-276 | The pose assert now allows cotrain mode. It still requires that `GMAX == 1` or dec_p K = 1 (the env has K = 1). Added `ZOOMPOSE = POSE and not ALOE_ONLY`; `FRAMING_COND` is refused with zoom ranks, because `inp.framing` would run on Aloe ranks only. Added `POSE_ONLY_PREFIX`. |
| L348-368 | `zoom_pose(m)` and `ZoomPoseDS`. A zoom micro becomes exactly a pose-less Aloe B side (= `data_aloe_posedup.pose_arrays(pose_b=None)` + `as_micro`): `kind 'pose'`, `pdepth [N,32] = -1` (the code tables add exactly 0), `pcb0 = -100`, `plight = []`, `own` (1,5) all zeros, `pose_ok 0`. The text part is unchanged. |
| L391-398 `prep` | A zoom micro gets `pcb0_w` = exact zeros, and its no-pose state is asserted. The Aloe branch is unchanged, except that the `lam == 1` assert is replaced by `pcb0_w *= lam` when `lam != 1` (W_ALOE on the pose term; it never fires at W_ALOE = 1). |
| L617-633 `run_micro` zoom branch | Asserts no pose labels or frames. Runs `dec_noop(dec_p)` with `Mj_p`, after dec_s's, in the same order as an Aloe micro. Backward roots are `[loss, hv_s, hv_p, zb_p]`; every root except `loss` gets a zero gradient, the same graph as an Aloe micro. Adds `dec_p_lrs/M/N` to `extra`, which the step loop asserts and logs. |
| L657 | The D33 link root's gradient is multiplied by `lam` when `lam != 1`, as for hv. It is bit-identical at W_ALOE = 1. |
| L1462, L1478 | Zoom train stream (`ZoomPoseDS`) and zoom eval micros go through `zoom_pose`. |
| L756-761, L1377 | Test only (`COTRAIN_TEST_DEPTH`): the deep `tower_p` and `dec_p` layers count as skipped deep layers, not pose skips. PROD never has deep keys. |
| L1866-1915 `gradcheck` | Check (1) now also covers `tower_p.`, `dec_p.` (the reduced flat buffer, through its grad views), `decp_bb.` and `inp.pose.`. It logs the pose-only max separately and zeroes dec_p's buffer between passes. |

aloe_only (30074's mode): no zoom micro exists, so no ZOOMPOSE branch runs. The only changes on that path are the two relaxed asserts, and `lam` is always 1.0, so the maths is byte-identical. Backup of the pre-edit file: `tests/train_samsep_posedup.py.orig-pre-zoompose`.

## FSDP parity: every rank runs the same units in the same order
- **Forward:** a zoom micro now has `kind 'pose'`, so `model_ppm.PPM.forward` runs the same units as for an Aloe micro: `inp` (`inp.pose` on all -1), `vadapter`/`vgain` (zero latent, `vmul` 0), layers, head, `tower_s`, `tower_p`, `decp_bb`.
- **Collectives:** after the forward, `dec_s` then `dec_p` each do one `_dec_update` all-reduce at the last slot (`dec_noop` on zoom ranks, `nanosteps_tw` or `dec_noop` on Aloe ranks). Then comes one backward with the same root set.
- **Step-0 loss check:** `pack_loss_check_zoom` does 1 + Kmax forwards plus the all-reduce. Its `prep` now gives the pose-shaped zoom micro zero pose weights.
- **Eval, calibration and band mask:**
  - Evals and the s calibration call the same `run_micro`/`prep`. The eval pose keys come from the Aloe stats vector, to which zoom ranks add `[0]*NVA`.
  - The band mask lives only in the data module (Aloe micros). An Aloe micro whose pose frames are all masked already takes the `light.numel()==0` path, which is `dec_noop` with zero roots.
- **Step-0 gradient check:** the "pose tensors without a gradient" check holds on zoom ranks, because zero roots keep the tensors in the graph.

## Normalisers: L_aloe pose terms = 30074's
- `tot[4]` / `tot[5]` (pose cb0 labels and pose frames) and `Mj_p` are all-reduced over all ranks. Zoom micros add 0, so these are the Aloe group's own pose totals.
- The Aloe pose cb0 weight is `1 / (max(tot[4],1) / WORLD)`, and FSDP averages over WORLD. That gives the token mean over the step's Aloe pose labels.
- The dec_p chunk weight is `WORLD / (31 Mj_p)`, which gives the mean over Aloe pose frames.
- These are 30074's rules exactly. WORLD = 64 in both runs; 30074 had 1 slot per GPU, and this run has 32 Aloe GPUs x 2 slots, which is exact accumulation.
- Result: pose-only tensors (tower_p, dec_p, decp_bb, inp.pose) and the other Aloe-only tensors get exactly 30074's gradient for the same micros. The shared trunk and head get it plus the zoom term.

## Tests
### GPU test T1: job 33086, PASS
Setup:
- `tests/zp_test.sbatch` with `tests/zp_t1.env`, on spare slinky-9 GPUs 0-1 (`CUDA_VISIBLE_DEVICES=0,1` from the gres allocation). Log: `tests/logs/zp-t1-33086.log`.
- 1 Aloe rank and 1 zoom rank, each with 2 slots: logical worlds 2 and 2. This is the production slot structure, GA = GZ = 2.
- The pose path is on, as in 30074 (link on, dec_p K 1), plus `POSEDUP_ITEMS=pose_ok` and `POSE_BAND_MASK=1`.
- Reduced depth 2,2,2 from PPM w-15973 plus the ASR adapter w-1441. This is the zoom-cotrain tests' init; `/scratch/.../poseduplex-init/w-18100` is not staged on any spare node.

Results:
- **Gradcheck (1), exact zeros.** In the zoom-only pass, the max |grad| over tower_s, vadapter, vgain, inp.aud, tower_p, dec_p, decp_bb and inp.pose is 0.000e+00 on every rank. The pose tensors alone are also 0.000e+00, over 116 local tensors. The reduced dec_s buffer is 0, and no tensor lacks a grad tensor.
- **Gradcheck (2), additivity.** grad(both) vs grad(aloe) + grad(zoom) has a global relative L2 error of 1.122e-04, against a nondeterminism floor of 1.126e-04 (the both pass run twice). The worst tensor is inp.pose.weight at 2.7e-3; it is Aloe-only, so the difference is pure backward-atomics noise. The dec buffer max |diff| is 3.7e-09 (floor 7.5e-09).
- **Gradcheck (3), group losses.** Identical across passes: aloe 29.625932, zoom 12.394907.
- **Step-0 loss check:** exact, 0.00e+00 on text, cb0, light, pose_cb0 and pose_light.
- **Steps and evals:** 2 real steps (all pose gn finite: gn_tower_p 74.7 / 38.6, gn_decp_bb 0.108 / 0.074) and evals at steps 0 and 2. No hang, exit 0.
- **Timing:** step 2 took 4.9 s (step 1 was 115 s, mostly decoder compiles). Peak 66.9 GiB at world 2.
- **Cleanup:** no ckpt was written, and nothing was left on /scratch.

### CPU dry for the 4+4 layout (`tests/zp_cfg_dry.sh`): PASS
Run on slinky-10. Log: `tests/logs/zp_cfg_dry.log`. Output: `tests/dry-zp/`.

Layout:
- Hosts slinky-11,12,13,14 are Aloe; slinky-15,16,17,19 are zoom.
- ALOE_LOGICAL_WORLD = ZOOM_LOGICAL_WORLD = 64; NA = NZ = 32; GA = GZ = 2.
- QQA_STEPS 426, pose_ok, W 1 / 1, steps mode aloe, band mask on.

Config diffs:
- PROD True.
- Against 26232 (the abort rule): 106 keys differ, **0 BAD**, VERDICT PASS. Every cotrain, zoom, accum and posedup key is allowed.
- Report-only diffs: 82 keys vs 26467, 78 vs 28511 and 89 vs 29345.
- The resolved config on rank 32 (zoom, slinky-15) is identical to rank 0's.

Zoom data, `/data/home/amartyavarma/zoom-charisma/full-5k`:
- Train: 4,900 conversations. Eval: 100. Both pass the asserts against the HF push: sizes 60,115,866 B and 1,228,694 B, sha256 prefixes 6c40f770… and afc014d9….
- Table cfg_key f33e97dc0358aa68. 4,900 packs per epoch, 100 eval packs.
- **Over 426 steps: 426 x 64 = 27,264 zoom packs = 5.56 zoom epochs.** One zoom epoch is 77 steps, and each epoch has its own seeded order.

Plan digest, world 64, 426 steps, pose_ok:
- **With POSE_BAND_MASK=1:** `838ce1e2dabeafee`, 27,260 train micros, 13,566 items per epoch (12 band-dropped lanes leave), 212 steps per epoch. So 426 steps = 2.01 epochs; 424 would be exactly 2.
- Without the band mask: `d3e5e9624ef3e682`, 27,156 micros. This is the env table's `426:pose_ok` line, so that line is wrong for a band-mask run.
- Both digests come from the current data module and the sam-sep stores. Re-run `tests/zp_cfg_dry.sh` after the raw-codes data change: its last block prints the digest.

## Env lines for the parent (on top of 30074's env)
```
COTRAIN_GROUP_MODE=hosts
COTRAIN_ALOE_HOSTS=slinky-11,slinky-12,slinky-13,slinky-14
PPM_HOSTS=slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19
PPM_HOST_NRANKS=8,8,8,8,8,8,8,8
ALOE_LOGICAL_WORLD=64
ZOOM_LOGICAL_WORLD=64
COTRAIN_W_ALOE=1
COTRAIN_W_ZOOM=1
COTRAIN_STEPS_MODE=aloe
QQA_STEPS_SET=426
POSEDUP_ITEMS=pose_ok          # run_poseduplex.env hard-sets POSEDUP_ITEMS=all BEFORE its plan table: change it there
POSE_BAND_MASK=1
POSEDUP_PLAN_DIGEST=838ce1e2dabeafee   # band mask on (recompute after the raw-codes change)
ALOE_EXPECT_TRAIN_MICROS=27260         # band mask on
QQA_DEC_K=1 / POSEDUP_DEC_P_K=1        # required with 2 slots (asserted); already 30074's
FRAMING_COND unset / 0                 # refused with zoom ranks
```
- `run_poseduplex.sbatch` line 19 aborts unless `COTRAIN_GROUP_MODE=aloe_only`. The parent must lift that guard.
- The init subset is staged per host index at world 64, the same world as 30074, so stage it in the new PPM_HOSTS order; zoom hosts need it too.
- The PROD ledger check on each Aloe host reads `DONE_v2vox_node{l}.json` for both of its logical nodes (l and l + 4, because of the 2 slots).
