# monosam: retrain the latest monoplex on the SAM audio codes (from main, 10-26 ~05:08 PT)

You are the tmux agent **monosam**. The user, verbatim:
> "okay how long will it take to train my latest monoplex on the sam version?"
> "okay rebuild expecting to train on 12 nodes we will at somepoint pause the ongoing run train and quickly resume the ongoing run"

## Goal
Get everything READY so that, when the user says go, we can:
1. pause MoT (job 30289) with a clean save;
2. train the monoplex SAM run on 12 nodes;
3. resume MoT quickly from that save.

DO NOT pause MoT or launch the training yourself. Main relays the user's go.

## The run to reproduce
- **Run:** ppm/aloe-pose-mono-8020 (job 24272, export-357; README.md there): PPM 32B monoplex pose FT.
  - 357 steps, 64 rows/step, row_per_rank layout, 80/20 prompt rule.
  - Init: PPM local-11488.
  - Data: stores ppm/aloe-pose-ft/stores p0..p7, built 10-16 by ppm/aloe-pose-ft/code/build_poseB_rows.py from data/aloe-handover-full/DATA.
- **Last time:** it ran on 8 nodes (64 GPUs) in 80 min wall.

## The ONLY change: the audio codes
- **Source of the SAM codes:** data/aloe-sam-sep/ (README.md there; HF InternalCan/stage2-sam-sep-v1-mimi, already on disk; no downloads).
- **Which stream:** use the per-speaker isolated, crossbleed-removed stems, `SPEAKER_00_isolated_npy` / `SPEAKER_01_isolated_npy`. Same mimi-stream-v5 recipe, same keys, same T.
- **Keep everything else byte-identical:** same rows, items, order, prompts, persona, pose tokens, words and steps.
- **Voice clips:** they come from B's own codes, so they also use the SAM codes. Say so in the README.
- **Build:** new stores at ppm/aloe-pose-ft-sam/stores p0..p7 with only the codes arrays swapped. Assert per row:
  - same (session, seg, speaker) key and same T;
  - pose/words/index byte-identical to the original stores.
- **Report:**
  - the fraction of code frames that differ, per codebook;
  - any row whose SAM codes are missing. If any are missing, STOP and tell main; never silently drop or fall back.
- **Speed:** use max CPU parallelism (64 workers per store, all stores at once) on spare-node CPUs via srun --overlap with --mem. Target ≤30 min.

## Run dir: ppm/aloe-pose-mono-8020-sam
- A copy of the original run's code/env/sbatch, with ONLY the store dir, run tag and output paths changed.
- **Plan check:** the trainer's own plan code over the new stores must give the identical item list, row cut, step count (357), warmup and LR schedule as the original. Diff the plan digests.
- **Config check:** diff the resolved config against job 24272's. Only paths/tags may differ.

## 12 nodes, same maths (USER HARD RULE: never change the maths)
- The recipe is 64 rows per step with row_per_rank: one row per rank on 64 ranks. On 12 nodes = 96 GPUs the global batch MUST stay 64 rows/step with the identical update.
- Work out exactly how the trainer runs on 96 GPUs:
  - e.g. 64 active ranks + 32 zero-weight dummy ranks (identical maths, but no speedup);
  - or an exact way to use the extra GPUs (e.g. wider FSDP sharding, so memory/comm is lower and the step faster).
- Estimate the wall time on 12 nodes vs 8 nodes from the original logs. If 12 is no faster than 8, say so plainly: the user may prefer 8 and keep 4 for something else.
- Prove maths identity: run step 1 on both layouts with the same seed (loss, grad norm and updated-param checksum equal to bf16 noise or exact). Do it on spare GPUs at small scale where possible. NEVER on the MoT hosts while MoT trains.

## Pause / train / resume script, staged and NOT run
Write ppm/aloe-pose-mono-8020-sam/GO.sh that, when main runs it:
(a) asks MoT for a clean save-and-exit at the next save boundary. Use the run's documented STOP_AFTER_SAVE / USR1 mechanism (memory stop-runs-with-save; coordinate with motguard amartyavarma-3c, which owns MoT's guard), wait until the save is COMPLETE on all 16 hosts, and record the step;
(b) launches the monoplex SAM run on 12 of those nodes (never slinky-9; always --mem);
(c) after its final export-357 is written + verified (and HF-backed up like the original), immediately relaunches MoT resume from the recorded save on its 16 hosts, with the same env/recipe as now;
(d) reports each phase to main.
Dry-run every check in GO.sh except the actual stop/launch. Get motguard to review the MoT stop/resume part.

## Rules
- **Commands.** Always pass --mem. NEVER slinky-9 (-x slinky-9). No login-node compute. No `cd … &&`.
- **Data and checkpoints.** No GCS. No downloads. bf16 only. Never delete checkpoints.
- **Parallelism.** Max-parallelise (subagents for store build / plan check / equivalence test in parallel).
- **Reporting.** Message main (name on line 3 of ppm/control/OPS_PLAYBOOK.md) ONCE when everything is READY with:
  - the build stats;
  - the plan/config diff;
  - the 12-vs-8 node estimate and maths proof;
  - GO.sh dry-run results.
  Or message on a blocker. Keep STATUS.md in the run dir.
