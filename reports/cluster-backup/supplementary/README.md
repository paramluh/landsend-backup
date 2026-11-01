# Supplementary backup coverage audit

Collected read-only metadata from landsend-0, landsend-1, landsend-2, landsend-3, landsend-4, landsend-5, and landsend-12.

## Additional recovery material

- Codex state, memory, and goal databases exported as sanitized SQL and JSON from consistent read-only snapshots. Each SQL dump was restored into an in-memory SQLite database and passed integrity_check. Authentication/enrollment data is excluded; logs/auth databases and raw SQLite files are not copied.
- Thread-history metadata exported separately; conversation items remain in the existing session JSONL backup.
- 30 Python environment inventories across seven nodes, including installed package versions, Python/platform information, source-wheel metadata, available pip freeze output, and generated requirements files.
- OS package, GPU driver, CUDA compiler, user crontab, and systemd-unit inventories.
- Small configuration and service files, including /etc/systemd/system/param-gpu-reservation.service and sanitized shell history.
- Per-host symlink inventories preserve targets for later reconstruction.

## Coverage findings

- No additional Git repositories were found in hidden directories.
- No custom Codex skills were present: only bundled .system skills (hash inventory retained). Codex visualizations contained no files.
- Across all seven hosts, no user-owned project directories were found under /opt, /mnt, /workspace, /data, or /var/tmp. User-owned /dev/shm files were NCCL buffers and multiprocessing semaphores only; these are transient process state. User-owned /tmp content is covered by the orphan scanner.
- Remote legacy /home/param/pose-runs directories are real directories on nodes 1,2,3,4,5,12. Remote /home/param/pose-demo-website is a real directory on nodes 1–4. Additional non-Git copies include ppm-training on node5 and vansh-video-models-handover on node12. The updated orphan scanner now discovers all nonhidden home project roots, covering these gaps.
- The /tmp/pose-comparison-browser directory is a Python venv; its package inventory is retained so it need not be treated as unique application source.

## Restore notes

SQL dumps can be restored into fresh databases using sqlite3; avoid replacing the database of a running Codex process. Session JSONL and attachment archives are separate. Requirements lists preserve package versions; adjacent environment metadata retains build/source URLs, and the system inventory captures CUDA/driver context. Reconstruct symlinks only after restoring their target directories. Do not restore ephemeral IPC or old authentication state.
