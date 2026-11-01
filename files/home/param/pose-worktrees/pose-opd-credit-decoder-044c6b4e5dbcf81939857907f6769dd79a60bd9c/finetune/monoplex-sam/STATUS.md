# aloe-pose-mono-8020-sam: status (tmux agent monosam), 10-26 ~06:05 PT

**State: READY except /scratch.** Everything is staged and dry-run. MoT has not been touched. One blocker needs main's OK: freeing node-local /scratch on the chosen hosts. Then the user's go.

## Results
1. **SAM-code stores** (`ppm/aloe-pose-ft-sam`, README there): built in 2 min 16 s.
   - 17,872 rows; **0 missing**, 0 T mismatches. An independent verify passes on all 8 stores.
   - Every non-codes file is byte-identical. The voice clips are SAM codes too, since they are cut from B's own codes.
   - **Fraction of code frames that changed, on the trained B tracks** (136.5M frames): all codebooks 78.9 %, cb0 20.5 %, cb1 34.1 %, cb2 57.8 %, cb3 63.9 %, cb4..31 70.8-89.2 %. Eval is about the same: cb0 19.9 %.
   - Speaker sanity: 0 of 22,886 conversation sides are flipped.
   - SAM "swapped" chunks cover 0.017 % of the trained B frames; they were kept, as specified.
2. **Plan / config.** `tests/plan_sam_check.py` runs the trainer's own plan, CFG and schedule code.
   - **SAM vs original at world 64: identical.** That covers rows digest 3681cc26ea97cad7, plan digest a5d995bcec45dc6f, items, rows, step groups, eval packing, 357 steps, warmup 12, exports, and every step's LR and decoder-LR multiplier.
   - **Resolved config vs 24272:** 8 keys differ, all paths or tags.
   - **Micros:** 48 train and 8 eval micros built by the trainer's builder differ only in the Mimi-code arrays.
3. **12 vs 8 nodes** (`tests/w96/REPORT.md`):
   - **Wall time:** 12 nodes ≈ 77 min, or ≈ 80-81 min with the full-scale identity check. 8 nodes = 80.2 min (24272 measured). The steps are 69 min either way, since each real rank still forwards one row.
   - **Recommendation: 8 nodes.** That is 24272's exact layout and code path, and it frees 4 hosts.
   - **12 nodes needs code changes.** The original code crashes at world 96 (step-0 loss check on a dummy rank). The fix is env-gated (`ALOE_LOSSCHECK_DUMMY=1`), plus `ALOE_NORM_WORLD=64` so that ranks 0..63 compute world 64's bf16 numbers.
   - **Small-scale maths proof** (depth 1, full width, world 3 vs world 4 with one dummy):
     - forward, evals, loss check, counts and LRs are bit-identical;
     - gradients with the knob differ by 2e-8 to 6.6e-4 relative L2, i.e. 1-4× the FA3 run-to-run floor (which is 0 to 2.4e-4);
     - without the knob they differ by 1e-3 to 3e-3 (one bf16 rounding).
   - **Full-scale check (12 nodes, default on):** 1 step at world 96 on the original stores, compared to 24272's step-0 evals 2.0112 / 1.9574 and its step-1 line. It costs ~3.5 min.
4. **GO.sh** (`--dry-run`, `--nodes=12|8`, `--no-idcheck`). The flow:
   - mot_stop.sh (STOP_AFTER_SAVE; dcp-S COMPLETE on 16/16; markers so the watchdog, guard and state backup stand still);
   - the FT, held on its hosts behind MoT and gated on READY;
   - the launch checks;
   - the HF backup controller (private `InternalCan/ppm-aloe-pose-mono-8020-sam`);
   - FT end with export-357 verified;
   - mot_resume.sh, with its verification.
   - motguard reviewed the stop/resume part and its 2 fixes are applied (`go/MOTGUARD_REVIEW.md`).
   - **Dry-runs, 12 and 8 nodes: every check passes except the per-host /scratch space.** That includes mot_stop and mot_resume dry-runs at rc 0. Logs: `go/GO_DRYRUN_12.log`, `go/GO_DRYRUN_8.log`.

## Blocker: /scratch (needs main's OK)
- **Space needed per host:** about 157 GB (12 nodes) or 222 GB (8 nodes) for the FT, plus 64 GiB that must still be free for MoT's resume pre-flight. Today's free space is 73-221 GiB (630 on slinky-15).
- **Candidates:** `go/SCRATCH_CANDIDATES.md` and `go/scratch_candidates.tsv`; read-only survey, with the traps listed there.
  - **12 hosts** (0,1,2,3,5,6,7,8,14,15,17,19): ≥ 240 GiB each after 1,553 GiB of SAFE deletions.
  - **8 hosts** (1,2,5,6,7,14,15,17): ≥ 300 GiB each after 1,322 GiB.
  - The SAFE set includes HF-verified full training states: ppmvox v5b dcp-18163, samsep dcp-249, poseduplex dcp-498.
- **Script:** `go/scratch_cleanup.sh --option 12|8`. It is a dry run by default; `MAIN_GO=1 ... --execute` deletes through `scratch_guard.sh`.
  - It can run now, inside MoT, at 1 CPU and ionice idle (`--jobid 30289`), so GO does not wait for it.
