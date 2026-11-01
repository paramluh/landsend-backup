#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --_node ]]; then
    exec /usr/bin/python3 - "$2" "$3" <<'PY'
import json, os, pathlib, socket, subprocess, sys
root, tail_lines = pathlib.Path(sys.argv[1]), int(sys.argv[2])
host = socket.gethostname().split('.')[0]
job = root/'cluster/latest'
if not job.exists():
    print(f'{host}: no launched run recorded under {root}')
    raise SystemExit(1)
job = job.resolve()
def pid_status(name):
    path = job/f'{name}.json'
    if not path.exists():
        return 'pending', False
    item = json.loads(path.read_text())
    try:
        fields = pathlib.Path(f"/proc/{item['pid']}/stat").read_text().rsplit(')',1)[1].split()
        live = fields[19] == item['start_ticks'] and fields[0] != 'Z'
    except FileNotFoundError:
        live = False
    return f"{item['pid']}:{'live' if live else 'exited'}", live
parent, parent_live = pid_status('supervisor')
trainer, trainer_live = pid_status('torchrun')
exit_path = job/'exit_code'
exit_code = exit_path.read_text().strip() if exit_path.exists() else '-'
stop = (root/'control/STOP_AFTER_SAVE').exists()
print(f'{host}: launch={job.name} torchrun={trainer} supervisor={parent} exit={exit_code} STOP_AFTER_SAVE={stop}')
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'], text=True)
rows = [line.split(',') for line in gpu.splitlines()]
print('  GPU util%=[' + ','.join(r[0].strip() for r in rows) + '] memoryGiB=[' + ','.join(f'{float(r[1])/1024:.1f}' for r in rows) + ']')
log = job/'node.log'
if tail_lines and log.exists():
    with log.open('rb') as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell()-32768))
        lines = f.read().decode(errors='replace').splitlines()[-tail_lines:]
    for line in lines:
        print('  ' + line)
if exit_code not in ('-', '0') or (exit_code == '-' and not parent_live):
    raise SystemExit(1)
PY
fi

run_dir= tail_lines=5
while (($#)); do
    case "$1" in
        --run-dir) run_dir=$2; shift 2 ;;
        --run-dir=*) run_dir=${1#*=}; shift ;;
        --tail) tail_lines=$2; shift 2 ;;
        -h|--help) echo 'Usage: status_cluster.sh --run-dir ABS [--tail 5]'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$run_dir" == /* && "$tail_lines" =~ ^[0-9]+$ ]] || { echo 'Require --run-dir ABS and a nonnegative --tail.' >&2; exit 2; }
tmpdir=$(mktemp -d /tmp/pose-cluster-status.XXXXXXXX)
trap 'rm -rf -- "$tmpdir"' EXIT
hosts=(landsend-1 landsend-2 landsend-3 landsend-4)
pids=()
for host in "${hosts[@]}"; do
    if [[ "$host" == "$(hostname -s)" ]]; then
        "$SCRIPT_DIR/status_cluster.sh" --_node "$run_dir" "$tail_lines" > "$tmpdir/$host" 2>&1 &
    else
        printf -v command '%q ' "$SCRIPT_DIR/status_cluster.sh" --_node "$run_dir" "$tail_lines"
        timeout 20s ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes -o ConnectTimeout=10 "$host" "$command" > "$tmpdir/$host" 2>&1 &
    fi
    pids+=("$!")
done
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then failed=1; fi
    cat "$tmpdir/${hosts[$i]}"
done
exit "$failed"
