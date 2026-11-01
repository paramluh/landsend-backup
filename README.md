# Landsend rental-cluster backup

Private backup for **paramluh**, initiated October 31, 2026 before the cluster rental ends at midnight America/Los_Angeles.

This repository contains otherwise-unversioned source code, configuration, recovery scripts, and backup receipts. Files under `files/` retain their original absolute paths with the leading slash removed. Credentials and private keys are excluded; recognized credential values in text are redacted.

Large experiment artifacts, logs, conversation samples, dataset derivatives, and archived working copies are stored in the private Hugging Face dataset **[paramluhadiya/landsend-backup-param-20261031](https://huggingface.co/datasets/paramluhadiya/landsend-backup-param-20261031)**. Archive receipts include SHA256 hashes. The dataset contains deduplicated Zstandard-compressed tar archives and manifests describing original paths and hardlink aliases.

Existing Git repositories are backed up to branch **`tmp/landsend-backup-param` in their original repositories**. Separate Git backup receipts list those repositories and commits. Existing checkpoint backups and any newly uploaded checkpoint files are tracked by the checkpoint inventory; they are not duplicated into GitHub.

## Restore

1. Clone this private repository.
2. Install `huggingface_hub` and the `zstd` command, then authenticate to Hugging Face with an account that can read the private dataset and referenced existing repositories.
3. Download the private dataset with `hf download paramluhadiya/landsend-backup-param-20261031 --repo-type dataset --local-dir hf-backup`.
4. Run `python3 restore_archives.py hf-backup --destination /path/to/recovery`. This verifies archive hashes, extracts into the recovery root, then restores hardlink aliases. It does not overwrite the original cluster paths unless you explicitly choose `/` as destination.
5. Overlay any newer files from this repository's `files/` directory, inspect the manifests, and clone the original repositories' backup branches.

Backups are being completed incrementally. An archive set is complete only when its remote `COMPLETE.json` exists and all listed archives pass verification. Final receipts distinguish completed uploads, already-remote datasets/checkpoints, exclusions, and any remaining gaps. Regenerable package/compiler/browser caches and authentication stores are intentionally not copied.
