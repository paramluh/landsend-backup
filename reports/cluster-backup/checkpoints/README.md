# Landsend checkpoint backup for Param

Private backup of checkpoint files found in Param's folders on Landsend before the rental cluster shutdown at midnight America/Los_Angeles, October 31, 2026 (2026-11-01 07:00 UTC).

## Layout and restoration

`hosts/<hostname>/<path relative to /home/param>/` preserves each original node-local checkpoint layout. Download the corresponding host subtree to `/home/param/` on that host, or use the saved model/export metadata to reshard for a different cluster. Preserve all rank shards and replicated state for an optimizer resume. `resume/COMPLETE`, `train_state.json`, export metadata, RNG state, and other leaf sidecars are included whenever they existed locally.

The upload workers prioritize the latest optimizer/resume saves, then model exports. Existing source checkpoints that were already available on Hugging Face are recorded with their original repositories and pinned revisions in `inventory/existing-*.json`; they need not be downloaded from this backup repository. The active basic OPD run is separately backed up to `InternalCan/pose-basic-opd-b336-20261031`, including its final saved optimizer state.

The Electron pose encoder is included here because the original private source repository was not accessible to the currently authenticated account during verification.

## Inventory and verification

`inventory/landsend-<n>-plan.json` lists the checkpoint leaf directories selected on each node. `inventory/landsend-<n>-uploaded.jsonl` records each completed upload, its immutable HF revision, file count, and byte count. Every uploaded leaf is checked against remote file sizes at its committed revision. `inventory/manifest-hash-verification.json` additionally compares remote LFS SHA256 values to native checkpoint manifests where those manifests provide file hashes. Historical checkpoints without saved SHA256 manifests receive the upload client/server content checks and explicit remote file-size checks.

`inventory/progress.json` records the most recently collected worker state. This repository is complete only when all listed workers are complete and the final verification inventory has been published. The preservation inventory includes multiple assembled copies and distributed replicas; Hugging Face/Xet content deduplication avoids retransmitting identical content where possible.

Source code, miscellaneous files, and restoration notes are handled by the companion personal GitHub repository `paramluh/landsend-backup` and backup branches in the original source repositories.
