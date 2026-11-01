# Production monitoring — 2026-10-29

All **seven scheduled checks passed**. At the final observation, **11:48:22.960229 UTC**, all 56 ranks were live at **step 75/284**. The window covered **35.0160 minutes wall time**, or **31.9932 minutes excluding the planned save/reload gap**. Training remained running after monitoring. The [completion summary](completion-summary.json) records every validation and node result.

The monitored run is `/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029`, [W&B `tfk0zyzj`](https://forge.coreweave.com/wandb/canopy-labs/param-pose-llm/runs/tfk0zyzj). It uses seven nodes, 56 H100s, real microbatch two/global batch 112, 284 optimizer updates over two epochs, shared 10% speech/pose depth sampling, and the selected chunk-512/direct-addmm text path. There are no evaluations. Normal checkpoint policy is every 7,200 seconds plus final save, retaining the latest resume state and older BF16 exports.

## Scheduled observations

All times below are UTC. Rank-step ranges reflect sequential node inspection, not a synchronized training barrier. Raw snapshots retain full timestamps, process identities, GPU ownership, logs, metrics, disk, and RAM observations.

| Scheduled | Actual observation | Completed rank steps | Result |
| --- | --- | --- | --- |
| 11:18:22 | 11:18:22.995659 | 11–11 | Pass |
| 11:23:22 | 11:23:23.566909 | 23–24 | Pass |
| 11:28:22 | 11:28:23.510800 | 36–36 | Pass; planned checkpoint transition |
| 11:33:22 | 11:33:22.952693 | 39–39 | Pass |
| 11:38:22 | 11:38:22.981003 | 51–51 | Pass |
| 11:43:22 | 11:43:23.015862 | 63–63 | Pass |
| 11:48:22, extension | 11:48:22.960229 | 75–75 | Pass |

The [monitor implementation](monitor_live_training.py) verifies coverage of ranks 0–55, live owned processes and expected GPU process sets, fresh/advancing progress, finite losses/metrics, absence of targeted error signatures, writable run paths, and at least 700 GB free disk. It records available host RAM. During the explicitly declared save/restart transition, process liveness/progress checks account for the planned stop; unrelated error logs, metric validity, and disk checks remain active. The transition has an expiration, so it cannot indefinitely suppress failures.

## Planned checkpoint transition

Training deliberately saved and stopped at step 36 to install tested checkpoint recovery. The measured save/reload gap was **181.371109 seconds**, from the first save-start event at **11:28:17.126880** to the first resumed batch at **11:31:18.497989**. The transition was marked complete after all ranks had resumed and advanced. The extension compensates for this planned interruption; it is not an unexpected crash or lost optimizer progress.

All 126 checkpoint payload files across seven nodes, totaling 716,134,475,911 bytes, were verified against SHA256 manifests. Every rank resumed step 36 and completed steps 37–39 under the same W&B ID. The previous writer performed the live normal save; the patched runtime performed the resume. The patched writer’s failure handling is covered separately by 65 CPU tests, 11 distributed fault cases, and three fault/retry/resume cases. See the [recovery report](../checkpoint-recovery/RECOVERY_REPORT.md), its portable full-scale source snapshots, and [round-trip events](checkpoint-resume-roundtrip-events.json).

## Learning rates and W&B cleanup

The [learning-rate audit](learning-rate-audit.json) confirms every main parameter group and both depth-decoder peak rates match the production pose-duplex finetune. The new complete-coverage schedule has nine warmup updates and a 284-update horizon. On the first resumed update, the LR scale was exactly 248/275, with no warmup restart.

Only two prior failed/stopped distillation W&B records were deleted: `y7yzkawo` (one completed update) and `qflr2d51` (21 completed updates), both short of their planned 664 updates. A fresh API listing verified their absence and preservation of the current run plus all eight unrelated runs. Associated artifacts and local logs/checkpoints were not deleted. The [cleanup summary](wandb-cleanup-summary.json) records the scoped evidence without exporting unrelated run metadata.

## Evidence and limits

The `snapshots/` directory contains unchanged scheduled and additional diagnostic snapshots. [The original schedule](schedule-state.json) preserves all six requested checks; [the transition record](planned-transition.json) records its explicit reason and completion. The [extension schedule](schedule-state-extension.json) and [completion summary](completion-summary.json) preserve the seventh observation. The observation reference began at 11:13:22 UTC; the exact planned save/reload gap is subtracted rather than counted as active training time.

At step 75, aggregate KL was **0.025412828**, the latest rank-zero update took **24.5554 seconds**, and all logged learning rates were positive. The benchmark-based estimate was approximately **88 more minutes of updates**, plus save overhead: 209 remaining updates × 25.192114 seconds. [The estimate record](eta-at-final-check.json) preserves its calculation and scope.

This monitoring window establishes startup and continued-training health, with a verified checkpoint round trip. It does not establish two-epoch completion, convergence, or immunity to later hardware/transport failures. Training remains running after monitoring; completion estimates exclude future checkpoint time and vary with conversation lengths.
