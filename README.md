# Landsend rental-cluster backup

Private backup for **paramluh**, initiated October 31, 2026 before the cluster rental ends at midnight America/Los_Angeles.

This repository contains otherwise-unversioned source code, configuration, recovery scripts, and backup receipts. Files under `files/` retain their original absolute paths with the leading slash removed. Credentials and private keys are excluded; recognized credential values in text are redacted.

Large experiment artifacts, logs, conversation samples, dataset derivatives, and archived working copies are stored in the private Hugging Face dataset **[InternalCan/landsend-artifacts-backup-param](https://huggingface.co/datasets/InternalCan/landsend-artifacts-backup-param)**. Personal Hugging Face storage rejected the first archive because its private-storage quota was exhausted; the GitHub repository remains in the requested personal account. Archive receipts include SHA256 hashes. The dataset contains deduplicated Zstandard-compressed tar archives and manifests describing original paths and hardlink aliases.

Existing Git repositories are backed up to branch **`tmp/landsend-backup-param` in their original repositories**. Separate Git backup receipts list those repositories and commits. Existing checkpoint backups and any newly uploaded checkpoint files are tracked by the checkpoint inventory; they are not duplicated into GitHub.

## Restore

1. Clone this private repository.
2. Install `huggingface_hub` and the `zstd` command, then authenticate to Hugging Face with an account that can read the private dataset and referenced existing repositories.
3. Download the private dataset with `hf download InternalCan/landsend-artifacts-backup-param --repo-type dataset --local-dir hf-backup`.
4. Run `python3 restore_archives.py hf-backup --destination /path/to/recovery`. This verifies archive hashes, restores each host into its own subdirectory, then restores hardlinks, empty directories and internal user-file symlinks. To restore one host directly under the destination, add `--host landsend-0`. External/system symlink targets remain in `metadata/symlinks.json` for review. Verified parts from partial archive sets are recovered too, with warnings; use `--complete-only` to skip them. It does not overwrite original cluster paths unless you explicitly select one host and choose `/` as destination.
5. Restore the 25,011 dataset files proven identical to existing HF revisions with `python3 restore_existing_datasets.py hf-backup/landsend-0/datasets/metadata/verified-existing-datasets.json --destination /path/to/recovery/landsend-0`. These 290.08 GiB are referenced by pinned revision and verified hash rather than duplicated into the bulk backup. Unique dataset derivatives are included in the archived `datasets` set. Remote-node dataset proofs and the additional restore helper, when present, are under `reports/cluster-backup/git/`.
6. Overlay any newer files from this repository's `files/` directory, inspect the manifests, and restore original repository working copies using [RESTORE_GIT.md](RESTORE_GIT.md).

Backups are being completed incrementally. An archive set is complete only when its remote `COMPLETE.json` exists and all listed archives pass verification. Final receipts distinguish completed uploads, already-remote datasets/checkpoints, exclusions, and any remaining gaps. Regenerable package/compiler/browser caches and authentication stores are intentionally not copied.

Archive copies are inspected recursively, including compressed source tarballs and nested Git bundles. Credential-bearing copied files are redacted; original source files and home-repository history are not modified by the artifact sanitizer. For rewritten bundle copies, `reports/cluster-backup/git/sanitized-bundles/` contains source/output fingerprints and old-to-new commit maps. Restore uses the sanitized artifact hashes recorded by the final receipts.
