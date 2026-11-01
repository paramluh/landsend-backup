#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
HOSTS=(landsend-1 landsend-2 landsend-3 landsend-4)
MASTER_ADDR=fdf9:c316:93ee:0:5f4:1564:2f9a:6448
SSH=(ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o ConnectTimeout=10)

run_node() {
    local host=$1 command
    shift
    if [[ "$host" == landsend-0 ]]; then "$@"; else
        printf -v command '%q ' "$@"
        "${SSH[@]}" "$host" "$command"
    fi
}

record_pid() {
    /usr/bin/python3 - "$1" "$2" <<'PY'
import json, pathlib, sys
pid, filename = int(sys.argv[1]), pathlib.Path(sys.argv[2])
start = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
filename.write_text(json.dumps({"pid": pid, "start_ticks": start}) + "\n")
PY
}

if [[ ${1:-} == --_supervise ]]; then
    shift
    run_dir=$1 launch_id=$2 node_rank=$3 port=$4
    shift 4
    [[ "$node_rank" =~ ^[0-3]$ && "$(hostname -s)" == "${HOSTS[$node_rank]}" ]] || { echo 'GPU training is restricted to the assigned rank on landsend-1 through 4.' >&2; exit 2; }
    job_dir="$run_dir/cluster/$launch_id"
    child_pid=
    finish() { local rc=$?; printf '%s\n' "$rc" > "$job_dir/exit_code"; date -u +%FT%TZ > "$job_dir/finished_utc"; }
    terminate() { if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; exit 143; }
    trap finish EXIT
    trap terminate TERM INT
    record_pid "$$" "$job_dir/supervisor.json"
    source "$SCRIPT_DIR/environment/activate.sh"
    export NCCL_SOCKET_IFNAME='=ibs14' NCCL_SOCKET_FAMILY=AF_INET6
    export GLOO_SOCKET_IFNAME=ibs14 NCCL_IB_DISABLE=0 NCCL_NET=IB
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
    export WANDB_ENTITY=${WANDB_ENTITY:-canopy-labs}
    cd "$REPO_ROOT"
    date -u +%FT%TZ > "$job_dir/started_utc"
    "$REPO_ROOT/.venv/bin/torchrun" \
        --nnodes="${#HOSTS[@]}" --nproc_per_node=8 --node_rank="$node_rank" \
        --master_addr="$MASTER_ADDR" --master_port="$port" \
        --max_restarts=0 "$SCRIPT_DIR/train.py" "$@" &
    child_pid=$!
    record_pid "$child_pid" "$job_dir/torchrun.json"
    if wait "$child_pid"; then exit 0; else exit "$?"; fi
fi

if [[ ${1:-} == --_start ]]; then
    shift
    run_dir=$1 launch_id=$2
    job_dir="$run_dir/cluster/$launch_id"
    mkdir -p "$job_dir"
    [[ ! -e "$job_dir/supervisor.json" ]] || { echo "Launch already exists: $job_dir" >&2; exit 1; }
    ln -sfn -- "$launch_id" "$run_dir/cluster/latest"
    nohup setsid "$SCRIPT_DIR/launch_cluster.sh" --_supervise "$@" \
        > "$job_dir/node.log" 2>&1 < /dev/null 9>&- &
    supervisor_pid=$!
    sleep 0.3
    if ! kill -0 "$supervisor_pid" 2>/dev/null; then
        tail -30 "$job_dir/node.log" >&2
        exit 1
    fi
    printf '%s supervisor_pid=%s log=%s\n' "$(hostname -s)" "$supervisor_pid" "$job_dir/node.log"
    exit 0
fi

if [[ ${1:-} == --_rollback ]]; then
    /usr/bin/python3 - "$2/cluster/$3/supervisor.json" <<'PY'
import json, os, pathlib, signal, sys
p = pathlib.Path(sys.argv[1])
if p.exists():
    item = json.loads(p.read_text())
    try:
        stat = pathlib.Path(f"/proc/{item['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[19] == item['start_ticks'] and os.getpgid(item['pid']) == item['pid']:
            os.killpg(item['pid'], signal.SIGTERM)
    except ProcessLookupError:
        pass
    except FileNotFoundError:
        pass
PY
    exit 0
fi

usage() {
    cat <<'EOF'
Usage: launch_cluster.sh --run-dir ABS [--mode train|benchmark-old|benchmark-new]
       [--recipe aligned-media|compact-all|no-text]
       [--micro-batch-rows 2] [--gradient-accumulation 1]
       [--resume never|auto] [--benchmark-warmup 1] [--benchmark-steps 2]
       [--port 29462] [--check-only] [other train.py arguments]

Run this CPU-only controller on landsend-0. Training uses only landsend-1 through
4 (32 GPUs), with node1 as rank0. The explicit run directory is used unchanged
on all four workers. Resume defaults to never; auto must be explicitly requested for an
existing run directory. No controls or checkpoints are deleted. --check-only
performs cluster preflight and source synchronization without launching GPUs.
EOF
}

run_dir= mode=train resume=never port=29462 check_only=0
mode_set=0 resume_set=0 warmup_set=0 steps_set=0
trainer_args=()
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --run-dir) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; run_dir=$2; trainer_args+=("$1" "$2"); shift 2 ;;
        --run-dir=*) run_dir=${1#*=}; trainer_args+=("$1"); shift ;;
        --mode) mode=$2; mode_set=1; trainer_args+=("$1" "$2"); shift 2 ;;
        --mode=*) mode=${1#*=}; mode_set=1; trainer_args+=("$1"); shift ;;
        --resume) resume=$2; resume_set=1; trainer_args+=("$1" "$2"); shift 2 ;;
        --resume=*) resume=${1#*=}; resume_set=1; trainer_args+=("$1"); shift ;;
        --benchmark-warmup) warmup_set=1; trainer_args+=("$1" "$2"); shift 2 ;;
        --benchmark-warmup=*) warmup_set=1; trainer_args+=("$1"); shift ;;
        --benchmark-steps) steps_set=1; trainer_args+=("$1" "$2"); shift 2 ;;
        --benchmark-steps=*) steps_set=1; trainer_args+=("$1"); shift ;;
        --port) port=$2; shift 2 ;;
        --port=*) port=${1#*=}; shift ;;
        --check-only) check_only=1; shift ;;
        *) trainer_args+=("$1"); shift ;;
    esac
done
[[ $(hostname -s) == landsend-0 ]] || { echo 'Run this launcher on landsend-0.' >&2; exit 2; }
[[ "$run_dir" == /* && "$run_dir" != / ]] || { echo '--run-dir must be an explicit absolute directory.' >&2; exit 2; }
[[ "$mode" == train || "$mode" == benchmark-old || "$mode" == benchmark-new ]] || { echo 'Invalid --mode.' >&2; exit 2; }
[[ "$resume" == never || "$resume" == auto ]] || { echo 'Invalid --resume.' >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ && "$port" -ge 1024 && "$port" -le 65535 ]] || { echo 'Invalid --port.' >&2; exit 2; }
[[ "$resume" != auto || -d "$run_dir" ]] || { echo '--resume auto requires the existing, explicit run directory.' >&2; exit 2; }
[[ -f "$SCRIPT_DIR/train.py" ]] || { echo 'train.py is not present.' >&2; exit 2; }
((mode_set)) || trainer_args+=(--mode "$mode")
((resume_set)) || trainer_args+=(--resume "$resume")
((warmup_set)) || trainer_args+=(--benchmark-warmup 1)
((steps_set)) || trainer_args+=(--benchmark-steps 2)
mkdir -p "$run_dir/cluster"
exec 9> "$run_dir/cluster/launch.lock"
flock -n 9 || { echo 'Another launcher holds this run directory lock.' >&2; exit 1; }
launch_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
job_dir="$run_dir/cluster/$launch_id"
mkdir -p "$job_dir"
printf '%q ' "$SCRIPT_DIR/train.py" "${trainer_args[@]}" > "$job_dir/command.txt"
printf '\n' >> "$job_dir/command.txt"

read -r -d '' preflight <<'PY' || true
import os, pathlib, socket, subprocess, sys
root, run_dir, port, master = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert socket.gethostname().split('.')[0] in ('landsend-1','landsend-2','landsend-3','landsend-4'), 'Forbidden GPU training host'
assert (root/'.venv/bin/torchrun').is_file(), 'Missing training venv'
assert (root/'.cache/kernels/vllm-flash-attn3/build/torch28-cxx11-cu128-x86_64-linux/__init__.py').is_file(), 'Missing pinned FA3'
trainer = str(root/'finetune/transcript_pose/train.py')
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        args = (proc/'cmdline').read_bytes().split(b'\0')
        if trainer.encode() in args:
            raise RuntimeError(f'Existing transcript-pose process PID {proc.name}; do not launch over a live run')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
gpu_pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip()
assert not gpu_pids, f'GPU compute processes already active: {gpu_pids}'
gpus = subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'], text=True).splitlines()
assert len(gpus) == 8, f'Expected 8 GPUs, got {len(gpus)}'
if socket.gethostname().split('.')[0] == 'landsend-1':
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
        sock.bind((master, port))
print(socket.gethostname(), 'preflight passed')
PY
pids=()
for host in "${HOSTS[@]}"; do
    run_node "$host" /usr/bin/python3 -c "$preflight" "$REPO_ROOT" "$run_dir" "$port" "$MASTER_ADDR" > "$job_dir/preflight-$host.log" 2>&1 &
    pids+=("$!")
done
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then failed=1; fi
    cat "$job_dir/preflight-${HOSTS[$i]}.log"
done
((failed == 0)) || exit 1

# Source only: no environments, model weights, run output, or credentials.
sync_filter=(--exclude=.git/ --exclude=.venv/ --exclude=.cache/ --exclude=__pycache/ --exclude=__pycache__/ --exclude=wandb/ --exclude=runs/ --exclude=outputs/ --exclude=checkpoints/ --exclude=validation/ --exclude='.env*' --exclude='.netrc' --include='*/' --include='*.py' --include='*.sh' --include='*.json' --include='*.md' --include='*.lock' --include='*.yaml' --include='*.yml' --exclude='*')
pids=()
for host in "${HOSTS[@]}"; do
    (
        rsync -a --no-owner --no-group "${sync_filter[@]}" -e 'ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o ConnectTimeout=10' "$SCRIPT_DIR/" "$host:$SCRIPT_DIR/"
        rsync -a --no-owner --no-group "${sync_filter[@]}" -e 'ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o ConnectTimeout=10' "$REPO_ROOT/finetune/xabi-main-17850/code/" "$host:$REPO_ROOT/finetune/xabi-main-17850/code/"
    ) > "$job_dir/sync-$host.log" 2>&1 &
    pids+=("$!")
done
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then cat "$job_dir/sync-${HOSTS[$i]}.log" >&2; failed=1; fi
done
((failed == 0)) || exit 1
if ((check_only)); then echo "Preflight and source sync passed; no GPU launch. Records: $job_dir"; exit 0; fi

rollback() {
    local rc=$?
    trap - EXIT INT TERM
    if ((rc != 0)); then
        echo 'Launch failed; stopping only supervisors from this launch ID.' >&2
        for host in "${HOSTS[@]}"; do run_node "$host" "$SCRIPT_DIR/launch_cluster.sh" --_rollback "$run_dir" "$launch_id" || true; done
    fi
    exit "$rc"
}
trap rollback EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
pids=()
for i in "${!HOSTS[@]}"; do
    run_node "${HOSTS[$i]}" "$SCRIPT_DIR/launch_cluster.sh" --_start "$run_dir" "$launch_id" "$i" "$port" "${trainer_args[@]}" > "$job_dir/start-${HOSTS[$i]}.log" 2>&1 &
    pids+=("$!")
done
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then failed=1; fi
    cat "$job_dir/start-${HOSTS[$i]}.log"
done
((failed == 0)) || exit 1
sleep 3
"$SCRIPT_DIR/status_cluster.sh" --run-dir "$run_dir" --tail 3
printf 'Detached launch %s started on 4 nodes / 32 GPUs (landsend-1 through 4).\n' "$launch_id"
printf 'Status: %q --run-dir %q\n' "$SCRIPT_DIR/status_cluster.sh" "$run_dir"
printf 'Graceful save-and-stop control: %s/control/STOP_AFTER_SAVE on any worker (landsend-1 through 4).\n' "$run_dir"
trap - EXIT INT TERM
