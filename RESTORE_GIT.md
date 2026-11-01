# Restore Git working copies

All listed home repositories contain branch `tmp/landsend-backup-param`. Backup commits preserve each original branch tip and all captured worktrees without changing the live checkout.

| Home repository | Verified backup commit | Captured worktrees |
| --- | --- | --- |
| [aloe-pose-duplex](https://github.com/canopy-labs-internal/aloe-pose-duplex) | `7533c61f0b4a06786b7ad86c1ff7156f60d923e9` | 1 |
| [aloe-pose-monoplex-infer](https://github.com/canopy-labs-internal/aloe-pose-monoplex-infer) | `06e1ee6c75ec25251bdfb5d880bc08e4e8ead7d1` | 1 |
| [page-templates](https://github.com/canopy-labs-internal/page-templates) | `3cf4d801e360903f251142ef8a73c46be131f42b` | 1 |
| [param-pose](https://github.com/canopy-labs-internal/param-pose) | `cdcf3d6db5facb01f5ceff270320d9225fdf9154` | 13 |
| [pose-repo](https://huggingface.co/InternalCan/pose-repo) | `84b99b91076a177b177095e0319781009690e6cb` | 13 |
| [pose-demo-website](https://github.com/canopy-labs-internal/pose-demo-website) | `a1c5ead06455835274ff4395f9b9571c9965af03` | 1 |
| [pose-tokenizer](https://github.com/canopy-labs-internal/pose-tokenizer) | `d74dfea4cd98613f009fd380b13c5b14d6e4e163` | 2 |
| [ppm-training](https://github.com/canopy-labs-internal/ppm-training) | `4dadc58a62c781352590e89559282191f67880bf` | 2 |
| [vansh-video-models](https://github.com/canopy-labs-internal/vansh-video-models) | `989b34ac8ff1fa4aab5d9792e7d5d0647727908e` | 5 |
| [opd-posttrain-speedups](https://huggingface.co/InternalCan/opd-posttrain-speedups) | `a54f9eba1aee2dea77678fb1380e8ecab80e6fdb` | 1 |
| [training-registry](https://huggingface.co/InternalCan/training-registry) | `8f7dcf0cfed76c8143740e660461840df9420722` | 1 |

After cloning a home repository, fetch/check out `tmp/landsend-backup-param` and read `LANDSEND_BACKUP_MANIFEST.json`. Each `source_snapshots[].worktrees[].working_commit` names an exact captured working tree; check out that SHA to recover it. `staged_commit` separately preserves the original index. The HF mirror of pose-repo has its own commit mapping because large binaries use HF-compatible LFS; consult `LANDSEND_HF_COMMIT_MAP.csv` there.

The formerly local-only pose-ar-diagnostics repository is saved as `git-bundles/landsend-0-pose-ar-diagnostics-7b661f26.bundle`. Restore with:

```bash
git clone git-bundles/landsend-0-pose-ar-diagnostics-7b661f26.bundle pose-ar-diagnostics
```

Full remote-ref and reachability verification is recorded in `reports/cluster-backup/git/VERIFIED_GIT_BACKUPS.json`.
