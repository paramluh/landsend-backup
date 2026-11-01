# Final Git catch-up

Run after the final training save and stop, before publishing the personal backup's final receipts:

```bash
python3 /home/param/outputs/landsend-backup-20261031/git/incremental_backup.py
```

The wrapper discovers owned repositories, audits 29 currently known worktrees on all seven nodes, uses temporary indexes, and compares working trees, staged trees, checked-out heads, branch names, local branches, and tags with the last successful backup. An unchanged audit creates no remote commits. Audit errors stop all remote pushes and return nonzero. It never changes active working branches or their real staging indexes.

Changed snapshots and prior backup heads become parents of a new `tmp/landsend-backup-param` commit, followed by non-forced pushes and remote-ref verification. Every working/staged snapshot named in the new manifest must be reachable before pushing. The pose Hugging Face mirror converts ordinary blobs larger than 10 MB into LFS pointers, appends to its previous history, and records original/mirror commit identities in `LANDSEND_HF_INCREMENTAL_MAP.json`.

`INCREMENTAL_LATEST.json` records the latest audit. `INCREMENTAL_REACHABILITY_VERIFIED.json` records the first refreshed pose manifest's independent ancestry proof. Historical run reports, logs, and any new bundles are under `incremental/<UTC timestamp>/`. `incremental-state.json` is the last fully successful audit baseline.

Original home histories remain intact. Existing historical credentials are not removed by this backup procedure. Do not publish unsanitized full-history home-repository bundles into the new personal backup. `sanitize_git_bundle.py` produces sanitized copies, preserves ref names, and emits SHA256 receipts plus old-to-new commit maps under `sanitized-bundles/`. It never changes source bundles or home repositories.

Validated behavior: a live 29-worktree audit; an unchanged repeat with zero pushes; an isolated appended mirror with exact LFS bytes and multiple worktrees; an isolated staged/worktree divergence test showing the real index unchanged and a newly staged credential excluded.
