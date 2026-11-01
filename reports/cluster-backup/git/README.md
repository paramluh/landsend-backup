# Landsend Git backup

Backup branch in every home repository: `tmp/landsend-backup-param`.

Verified 28 worktree snapshots across seven cluster nodes in ten home repositories, with eleven verified remote destinations. The additional local-only pose-ar-diagnostics repository is preserved as a standalone Git bundle. Original checkouts and indexes were left intact.

## Restore

Clone the appropriate home repository and check out `tmp/landsend-backup-param`. Read `LANDSEND_BACKUP_MANIFEST.json`: each worktree has a `working_commit` identifying its complete working-tree snapshot, an optional `staged_commit` preserving its original index, and its original branch/head. All those snapshots and local branch tips are ancestors of the backup branch.

For the Hugging Face pose-repo mirror, an 11 MiB tokenizer JSON had to be converted to Git LFS. Use `LANDSEND_HF_COMMIT_MAP.csv` to translate original commit IDs when restoring from HF; GitHub param-pose preserves the original Git IDs exactly.

For the repository without a home remote:

```sh
git clone landsend-0-pose-ar-diagnostics-7b661f26.bundle pose-ar-diagnostics
git -C pose-ar-diagnostics checkout tmp/landsend-backup-param
```

## Verified remotes

| Repository | Commit |
|---|---|
| git@github.com:canopy-labs-internal/aloe-pose-duplex.git | `7533c61f0b4a06786b7ad86c1ff7156f60d923e9` |
| git@github.com:canopy-labs-internal/aloe-pose-monoplex-infer.git | `06e1ee6c75ec25251bdfb5d880bc08e4e8ead7d1` |
| git@github.com:canopy-labs-internal/page-templates.git | `3cf4d801e360903f251142ef8a73c46be131f42b` |
| git@github.com:canopy-labs-internal/param-pose.git | `cdcf3d6db5facb01f5ceff270320d9225fdf9154` |
| https://huggingface.co/InternalCan/pose-repo | `84b99b91076a177b177095e0319781009690e6cb` |
| git@github.com:canopy-labs-internal/pose-demo-website.git | `a1c5ead06455835274ff4395f9b9571c9965af03` |
| git@github.com:canopy-labs-internal/pose-tokenizer.git | `d74dfea4cd98613f009fd380b13c5b14d6e4e163` |
| git@github.com:canopy-labs-internal/ppm-training.git | `4dadc58a62c781352590e89559282191f67880bf` |
| git@github.com:canopy-labs-internal/vansh-video-models.git | `989b34ac8ff1fa4aab5d9792e7d5d0647727908e` |
| https://huggingface.co/InternalCan/opd-posttrain-speedups | `a54f9eba1aee2dea77678fb1380e8ecab80e6fdb` |
| https://huggingface.co/InternalCan/training-registry | `8f7dcf0cfed76c8143740e660461840df9420722` |

## Scope

Tracked files, nonignored untracked project files, staged versions, every local branch tip, and detached worktree heads were preserved. Dependency environments and generated caches were excluded from Git commits. The separate orphan/artifact backup handles non-cache ignored assets. Credential patterns were checked before new files were committed; no credential exclusions were needed. Checkpoints are handled by the separate Hugging Face checkpoint backup.

Full machine-readable receipt: `VERIFIED_GIT_BACKUPS.json`.
