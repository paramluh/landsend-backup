# Monoplex SAM re-run on 12 nodes (world 96): ranks, code changes, speed, proof, launch check
(w96 subagent of monosam, 10-26; saved by monosam)

**Verdict.**
- World 96 gives the same per-step update as job 24272, but the original code **crashes at step 0** on the dummy ranks (step-0 loss check: `KeyError 'singles'`). One env-gated fix is required.
- **12 nodes is not meaningfully faster than 8:** about 77 min against the measured 80.2 min, and about 80-81 min with the launch-time identity check. Every step still forwards one row per real rank.
- **Recommendation: 8 nodes.** That is 24272's exact layout and code path, and it leaves 4 hosts free.

## 1. Rank layout at world 96 (`plan_w96_check.py`, all PASS, `out/plan_w96_check.json`)
The trainer's own `make_plan_v2` was run on the original stores at world 64 and at world 96.
- **Rows digest:** `3681cc26ea97cad7` at both worlds.
- **`plan_digest`:** `a5d995bcec45dc6f` at world 64 and `a97a0950d765f6d6` at world 96. It includes the per-rank micro list and is used only for resume.
- **Steps:** 357, layout `row_per_rank`, R = 1 at every step.
- **Rows on ranks:** at every step, rank j < 64 gets the same row at both worlds.
- **Dummies at world 96:**
  - ranks 64..95 at every step;
  - ranks 41..95 at step 357 (55, against 23 at world 64);
  - 11,447 in total.
- **Eval packing:** the same 529 micros, 8-9 per rank at world 64 and 5-6 per rank at world 96.
- **Rank-independent data:** a micro's arrays are byte-identical when built with a different rank number.
- **Which hosts run only dummies:** rank = 8 × (host index in the sort -V host list) + local rank, so the last 4 hosts do only dummy work.
- **Dummy cost:** a dummy micro is the plan's first item, (7, 2219, 1): 2,360 frames, at most 4,362 positions. Every step has a real 7,500-frame row, so a dummy rank is never the slowest.

## 2. What else was checked
- **Rank-keyed RNG.**
  - The decoder nanostep frame permutation seed is `(dec_seed*1000003 + micros*1009 + RANK)*64 + r`, so it is identical for real ranks 0..63.
  - The eval decoder subsample uses seed 1234, and there is no dropout.
  - Light frames are keyed on `row_key` (light_frac 1.0). The voice and persona draws are keyed per item.
- **Normalisers.**
  - Text and cb0: `tot/WORLD`.
  - Decoder chunk weights: `31 M_j / WORLD`.
  - Decoder all-reduce: `/WORLD`.
  - FSDP2: an fp32 average over WORLD.
  - The factor cancels in every term.
- **Init reshard 128 → 96.** The shards are uneven, and some ranks get none:

  | dim-0 size | ranks with no shard |
  |---|---|
  | 128 | 64-95 |
  | 1024 / 2048 | 94-95 |
  | 2560 / 5120 | 95 |
  | 1 | 1-95 |

  - The 1-row case already ran in 24272.
  - The 128 → 3 and 128 → 4 reshards gave bit-identical fingerprints in the test.
- **Saves and exports.**
  - A world-96 resume needs the same 12 hosts in the same order (asserted).
  - `infer_ppm` and the HF `finalize.py` read the world from `meta.json`.
  - Peak memory is about 42 GiB per GPU, against 45.5.

## 3. Code changes (in the -sam copy only; `code_changes.diff`, 2 files)
Each change is gated by an env variable whose default reproduces the original code.

| where | change | env (default) | status |
|---|---|---|---|
| `data_aloe_pose.StepDS` | A dummy micro at the loss-check step gets `singles` = the fallback item's real micro. | `ALOE_LOSSCHECK_DUMMY` (0) | **required at world 96** |
| `train_aloe_pose.pack_loss_check` | A dummy rank runs the same forwards on that stand-in, with K = 0, and reports its errors as 0. | same | required at world 96 |
| `train_aloe_pose`: `NORM_W` / `set_norm_world` | The normalisers, decoder `den` and `div_`, the FSDP2 `set_gradient_divide_factor` and the logged loss all use NORM_W. | `ALOE_NORM_WORLD` (unset = WORLD; only WORLD or 64 is accepted) | recommended = 64 |
| `train_aloe_pose`: `STOP_AT` | Stop after step N, with no eval, save or export. | `ALOE_STOP_AT_STEP` (0) | launch-time check only |

None of these knobs is in the CFG, so the resolved-config diff against 24272 is unchanged. The trainer logs them on a `[sam]` line.

## 4. 12 nodes against 8 nodes (24272 measured 80.2 min, 17:28:30 → 18:48:43)

| phase | 8 nodes (measured) | 12 nodes (estimate) |
|---|---|---|
| launch, plan, build, init | 14 + 12 + 17 s | ~15 + 10 + 15 s |
| two step-0 evals | 24.4 + 21.8 s | ~16 + 15 s |
| step-0 loss check | 23 s | 23 s |
| 357 steps (step 1: 52.5 s with compile, then a mean of 11.50 s) | 4,145 s | ~4,145 s (±2%) |
| 20 evals | 408 s | ~272 s |
| exports and saves | 137 s | ~95 s |
| consolidated export and teardown | 21 s | 21 s |
| **total** | **80.2 min** | **~77 min (76-79)** |

There is no exact way to speed up the step itself. Splitting a row across GPUs is a maths rewrite. Moving decoder frames onto the dummy ranks would change only the reduction order, but it would need new cross-rank code.

## 5. Small-scale proof (`step_equiv.py`, `compare.py`, `out/compare_*.json`)
**Setup.**
- The trainer's own data path, build, `train_step_multi`, clipping, AdamW, schedule and step-0 loss check.
- Full width, depth 1/1/1 (5.54B params), initialised from local-11488.
- The vocab tables and `inp.aud.0` are frozen in both layouts so the model fits on the shared GPUs. The `inp.aud.0` gradient is 0 anyway.
- Steps 1-2 on slinky-4 GPUs 0-3.

**The four runs:**
- **W3:** world 3, all ranks real.
- **W4:** world 4, rank 3 a dummy.
- **W4n:** W4 with `ALOE_NORM_WORLD=3`.
- **W3b:** the default code path after all the edits.

Each was run twice from a re-read init.

| (rel L2 per optimizer group) | noise floor (same layout) | W3 vs W4 (plain) | W3 vs W4n (knob) |
|---|---|---|---|
| forward stats, eval sums, loss check, counts, LRs | bit-identical | bit-identical | bit-identical |
| logged loss | 0 | 7e-8 | 0 |
| light CE | 0 | 4e-7 | 6e-8 |
| grad norm, max over groups | 3.1e-6 | 1.5e-3 | 7.3e-6 |
| grad step 1: trunk / tower_s / audio | 3.5e-5 / 1.3e-5 / 2.1e-4 | 2.4e-3 / 1.8e-3 / 1.9e-3 | 2.0e-4 / 1.6e-4 / 6.6e-4 |
| grad step 1: proj / spec / decp_bb | 0 / 0 / 0 | 2.2e-3 / 1.3e-3 / 1.7e-3 | 2e-8 / 0 / 4.4e-5 |
| update step 2: trunk / tower_s / audio | 3.2e-4 / 3.6e-5 / 1.5e-3 | 4.2e-3 / 5.0e-3 / 1.2e-2 | 1.1e-3 / 1.3e-3 / 4.2e-3 |
| decoder update, step 1 | 0 | 4.9e-3 | 1.5e-4 |

**What this shows:**
- The dummies add exactly nothing.
- Without the knob, the gradients differ by one bf16 rounding (the W'/W scale on each rank's loss weights).
- With the knob, only the fp32 reduction order is left: 1-4× the FA3 noise floor.
- W3b against W3 is at the noise floor, so the default path is unchanged.

## 6. Launch-time identity check (designed and implemented by monosam; optional, about 3.5 min)
**What it runs:** a 1-step world-96 run of the -sam code on the ORIGINAL stores. It uses `idcheck.env`: the -sam env plus `ALOE_STORE_DIR=<original>`, `ALOE_STOP_AT_STEP=1`, `ALOE_RESUME=never`, and its own run tag, logs and dirs. It writes no save and no export.

**Checker:** `check_launch_identity.py <run dir> <log>` compares it against 24272 by tolerance class:

| class | tolerance | what is compared |
|---|---|---|
| EXACT | equal | counts, LRs, rows digest, prompt fractions, token totals; loss-check rank 0 [0.66991, 1.25048, 3.26354, 1.04616, 3.78573] |
| EVAL | ≤ 1e-6 | step-0 evals 2.011178055650361 and 1.9574 |
| FWD | ≤ 1e-5 | step-1 loss 23.7407, word_ce_norm 2.0532, cb0 1.1145, pose cb0 1.2920 |
| DEC | ≤ 1e-4 | light 2.8651, pose light 4.0827 |
| GN | ≤ 5e-3 | gn_trunk 57.11, gn_tower_p 43.06 |

`CHECK_MODE=plain` loosens DEC and GN for a run without the knob. Run on 24272 against itself, the checker passes everything except the deliberate world / dummy-count line.
