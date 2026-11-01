# Checkpoint recovery audit — 2026-10-29

The revised writer synchronizes local checkpoint failures before peers enter the next collective. Periodic saves can fail without stopping training; required final/stop saves retry within a fixed limit. Resume restores model, optimizer, RNG, step, and the existing W&B run ID. These changes affect checkpoint/control behavior, not the distillation objective or batch schedule.

## Failure evidence

The harness invokes the actual checkpoint writer with a small sharded DTensor model and AdamW on two CPU/Gloo ranks. Each fault uses a fresh process group with a four-second collective timeout. The failing worker stays alive while its peer reports, so early process exit cannot hide a mismatched collective. An already verified previous checkpoint must remain available.

The [before report](before.json) reproduced three asymmetric failures: an exception preparing model state, creating checkpoint directories, or enumerating old resume directories caused one rank to raise immediately while its peer timed out around four seconds later. Write, metadata, marker, and individual-prune failures already had consistent outcomes.

The [after report](after.json) passes **all 11 cases**:

| Injected failure | Revised result on both ranks |
| --- | --- |
| Model-state preparation | Same typed `CheckpointSaveError` |
| Byte-size calculation | Same typed `CheckpointSaveError` |
| Directory creation | Same typed `CheckpointSaveError` |
| Rank-file write | Same typed `CheckpointSaveError` |
| Hashing during write | Same typed `CheckpointSaveError` |
| Verification checksum | Same typed `CheckpointSaveError` |
| Metadata write | Same typed `CheckpointSaveError` |
| Final directory rename | Same typed `CheckpointSaveError` |
| Latest-resume marker | Verified new checkpoint returned; warning, no old-resume pruning |
| Individual old-resume removal | New checkpoint returned; warning, old resume retained |
| Enumerating old resumes | New checkpoint returned; warning, old resumes retained |

Every case preserved the previous checkpoint and avoided transport timeout. The [recovery report](recovery.json) additionally checks directory, metadata, and verification failures followed by a successful all-reduce on the same group, a new attempt at the same step, and resume from that retry. These three cases pass on both ranks. An independent two-rank DTensor test verifies exact AdamW continuation after save/resume and successful old-resume retention policy.

## Trainer policy

Only the typed error representing a globally agreed local save failure is recoverable. A periodic failure returns to training and schedules the next attempt after 60 seconds. A final or graceful-stop save makes at most three attempts, five seconds apart; exhaustion raises visibly. Raw transport errors or unexpected exceptions propagate, because peer agreement cannot be assumed.

Attempt zero uses `step-NNNNNN`; subsequent attempts use `step-NNNNNN-retry-01`, `-retry-02`, and so on. Distinct directories prevent partially written or already renamed files from colliding with the next attempt. Failed attempts are retained for inspection. Successful saves still verify rank hashes, commit node-local latest markers, and prune only older validated resume directories; BF16 exports remain. A failed marker update does not invalidate the newly verified checkpoint, but its explicit path may be needed until a later marker succeeds.

Checkpoint event logging cannot prevent entry into the save collective: failures in the disk logger fall back to stdout, and a failing stdout fallback is also suppressed. Ordinary logging and unrelated process failures are outside this recovery guarantee.

W&B resume reads existing `wandb.json` from the output directory or original checkpoint’s run directory. It requires a nonempty recorded ID and the requested project; conflicting source/output IDs are rejected. Initialization uses that ID with `resume="must"` and broadcasts initialization errors across ranks. Fresh runs do not reuse old IDs. Replayed updates at or below already logged W&B steps may be ignored by W&B until history catches up.

## Tests and scope

All **65 CPU unit tests** passed, including 15 focused tests of the trainer’s actual helpers extracted from its AST: W&B identity preservation, new output-directory lookup, invalid metadata rejection, bounded retries, periodic continuation, unexpected-error propagation, and logger/stdout failures. Other tests cover the exact retry suffix, safe retention, optimizer continuation, data coverage, and loss math.

The tested writer SHA256 is `fdb174953cabdb85233cb0fcd643a4771aaf1dc34d527f632b1a6cb5a2ff69b4`; the before version is recorded in its report. The harness is `finetune/duplex_distill/tests/distributed_checkpoint_faults.py`. Reports retain exact source hashes, outcomes, and timings.

## Full 56-GPU save and resume

A controlled production save/stop at **step 36** completed on all seven nodes. The independent [checkpoint audit](full-scale/checkpoint-step-000036-audit.json) verified all **126 payload files**, totaling **716,134,475,911 bytes**, against their SHA256 manifests: 56 resume rank shards, 56 BF16 export rank shards, and one replicated file per format on each of seven nodes. All node manifests agreed. Rank zero’s save took **74.830313 seconds**, measured between its save-start and save-complete events.

The restarted process loaded the revised recovery implementation. Every one of 56 ranks reported `resume_complete` at step 36 and then completed **step 37**, as recorded in the [full-scale resume audit](full-scale/full-scale-resume-audit.json). All teachers remained gradient-free. Maximum allocated memory across those first resumed updates was **68.8396 GB**. W&B continued under the existing ID **`tfk0zyzj`**, and the first resumed learning-rate scale was exactly **248/275**, consistent with update index 36 on the 284-update schedule. See [the first resumed metric](full-scale/resumed-first-metric.json).

The initial resumed update included cold-start overhead. Subsequent rank-zero updates 38 and 39 took **24.4986** and **27.2208 seconds**. These are individual live-update observations, not a new throughput benchmark. [The compact round-trip summary](full-scale/roundtrip-summary.json) retains the exact save events, those updates, and source hashes. The [resumed source/config audit](full-scale/source-config-audit-resumed-source.json) verifies identical source, configuration, and plan across all seven nodes. Original and resumed snapshots are preserved under `full-scale/launch-source/` and `full-scale/resumed-source/`.

The distinction is deliberate: **the live checkpoint was written by the previous writer’s normal path, then read by the patched runtime**. This proves a full production save/restart round trip and W&B identity preservation. The patched writer’s injected-error and retry behavior is established by the separate CPU/Gloo tests above; no save failure was deliberately injected into the live 56-GPU job. Neither validation proves recovery from broken NCCL transport, a dead host, or every filesystem failure. Complete two-epoch training was still in progress at this audit.
