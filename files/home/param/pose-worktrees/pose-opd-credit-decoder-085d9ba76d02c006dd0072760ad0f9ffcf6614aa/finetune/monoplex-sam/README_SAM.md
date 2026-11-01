# aloe-pose-mono-8020-sam: the latest monoplex (job 24272) retrained on the SAM audio codes

The user asked on 10-26 ~05:08 PT, via main: "how long will it take to train my latest monoplex on the sam version?" and "rebuild expecting to train on 12 nodes; we will at some point pause the ongoing run, train, and quickly resume the ongoing run".

- **Recipe:** job 24272's exactly (README.md here is 24272's own): PPM 32B monoplex pose FT, init PPM local-11488, 357 steps of 64 rows, row_per_rank, the 80/20 prompt rule, and the same items, rows, order, prompts, persona, pose and words.
- **The ONLY data change is the Mimi codes.** They come from `ppm/aloe-pose-ft-sam/stores`: 22091's stores with `codes_assistant` and `codes_user` swapped for the SAM-separated, crossbleed-removed isolated stems (SPEAKER_00 and SPEAKER_01). That dir's README.md has the build, the checks and the per-codebook change.
- **Voice clips:** the trainer cuts the voice clip from B's own codes (a later chunk of the same conversation), so **the voice clips are SAM codes too**.

## What differs from 24272, and nothing else
| file | difference |
|---|---|
| `aloe_mono8020.env` | paths and names only: store dir, run tag, log, plan, control, export, save and local-export dirs, and QQA_REF_CFG2 = 24272's resolved cfg |
| `aloe_mono8020_run.sh` | run dir, scratch dir and store assert; forbidden hosts are now slinky-4/9/10/18; the node-count switch (below); idcheck support (own run-tag files, original stores only with ALOE_STOP_AT_STEP=1) |
| `aloe_mono8020.sbatch` | job name posemonosam, log, -N12 (GO.sh passes -N and -w), the optional IDCHECK stage |
| `code/` | env-gated world-96 changes only (`go/CODE_DIFF.txt`, `tests/w96/REPORT.md`). Every default is 24272's code path |

- **Resolved config vs 24272** (the trainer's CFG block, `tests/plan/CFG_DIFF_vs_24272.txt`): 8 keys, all paths or tags: control_dir, export, local_export_root, log_root, plan_dir, run_tag, save_root, stores.
- **Plan** (`tests/plan_sam_check.py`, the trainer's own code): SAM stores vs original stores are identical at world 64.
  - Identical: rows digest 3681cc26ea97cad7, plan digest a5d995bcec45dc6f, items, rows, step groups, eval packing, 357 steps, warmup 12, exports, and every step's LR and decoder-LR multiplier.
  - At world 96, ranks 0..63 get world 64's exact micros at every step, and ranks 64..95 are dummies.
- **Micros** (`tests/micro_diff_check.py`): 48 train and 8 eval micros built by the trainer from both stores. Every field is identical except the Mimi-code arrays (`cb0`, `depth`, and the loss-check copy `singles`).

## 8 or 12 nodes (the run script switches on SLURM_NNODES; any other count aborts)
| | 8 nodes (world 64) | 12 nodes (world 96) |
|---|---|---|
| layout | 24272's exactly, with the same code path (no knob) | 64 real ranks + 32 zero-weight dummy ranks; `ALOE_LOSSCHECK_DUMMY=1` (required: the original code crashes at the step-0 loss check on a dummy rank) and `ALOE_NORM_WORLD=64` (ranks 0..63 compute world 64's bf16 numbers; only the fp32 reduction order differs) |
| maths | identical | identical; small-scale proof in `tests/w96/REPORT.md` section 5; optional full-scale 1-step identity check vs 24272 (~3.5 min) |
| wall | 80.2 min (24272 measured) | ~77 min, or ~80-81 min with the identity check |
| /scratch per host | ~222 GB + 64 GiB left for MoT's resume | ~157 GB + 64 GiB |

## GO
`GO.sh` (with `--dry-run` first; `--nodes=12|8`): pause MoT with a clean save → train → resume MoT. See its header and `go/` (mot_stop.sh, mot_resume.sh, MOT_NOTES.md, MOTGUARD_REVIEW.md). Backups go to private HF `InternalCan/ppm-aloe-pose-mono-8020-sam` through the controller in `setup-logs/hfb/aloe-bk/posemono-sam` (never GCS).
