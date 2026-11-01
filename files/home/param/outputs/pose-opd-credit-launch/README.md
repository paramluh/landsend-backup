# Production launch helpers

These helpers are outside the training repository and do not change its code.
The five fixed hosts are `landsend-2`, `landsend-3`, `landsend-4`, `landsend-5`
and `landsend-12`, with all eight GPUs on each host. Hosts 0 and 1 are excluded.
The run directory is `/home/param/runs/pose-runs/pose-opd-credit-2min-b320`.

Only run preparation after the selected recipe and all source changes are
committed and the source worktree is clean. It packages exactly that HEAD with
`git archive`, records the complete commit plus archive/file/config/helper
SHA256 hashes, and stages a read-only checkout whose name includes the complete
commit. Preparation never launches a model or requires GPUs to be free.

```bash
cd /home/param/outputs/pose-opd-credit-launch
COMMIT=$(git -C /home/param/pose-worktrees/pose-opd-credit rev-parse HEAD)
python3 prepare.py --commit "$COMMIT"
SPEC="/home/param/outputs/pose-opd-credit-launch/$COMMIT/production-launch.json"
python3 start.py "$SPEC" --dry-command
```

`prepare.py --local-only --commit "$COMMIT"` produces local review artifacts
without SSH. A dirty worktree or mismatched two-minute/global320 recipe fails;
the committed recipe must explicitly exclude source rows shorter than1500 frames.
The selected student physical batch is taken from the committed recipe; it is
not overridden by these helpers.

Starting is a separate, explicit action. Do not run the final command until the
production launch is authorized and the benchmark jobs have exited.

```bash
python3 start.py "$SPEC" --check-only   # read-only, all five hosts
python3 start.py "$SPEC"                # ACTUAL PRODUCTION START
```

Every host must pass the source/config/helper identity check, a no-existing-GPU
compute-process check, an eight-GPU check, zero-utilization/<2048MiB idle-memory
check, and an available exclusive launch lock check before any supervisor is
created. Each supervisor then acquires the existing shared GPU lock, repeats
the GPU check, and waits for all five supervisors to become ready before GO.
There is a 120-second readiness timeout. A prior start marker is always rejected,
including after a failed start; these helpers do not silently retry or resume.
If orchestration fails before GO, the waiting workers time out without training.
If GO delivery partially fails, cleanup signals only this invocation's recorded
supervisors after matching their PID start ticks; unreachable hosts are recorded
in `PARTIAL-GO-FAILURE.json` for explicit recovery.
Inspect the receipts before any manual recovery; do not delete start markers
while their recorded processes are live.

The fixed child command uses the original Python environment and activation
script, InfiniBand on `ibs14`, IPv6 sockets and all eight GPUs. Training receives
only `--config` and `--run-dir`: there is **no** `--max-updates`, `--no-save`,
benchmark argument, optimizer warmup or implicit resume. Adam starts normally;
checkpointing remains enabled. All recipe stopping/checkpoint settings apply.

Each host writes `supervisor/worker.log`, `supervisor/training.log`,
`supervisor/command.json`, `supervisor/detached-supervisor.json`,
`supervisor/supervisor.json`, `supervisor/torchrun.json` and, on completion,
`supervisor/finished.json`. Process receipts contain PID and Linux process start
ticks, so later tooling can distinguish PID reuse. Signals received by the
supervisor target only its own live child process group. No broad kill command
is present.

CPU-only helper checks:

```bash
python3 -m unittest -v test_launch.py
python3 -m py_compile common.py prepare.py start.py worker.py
```
