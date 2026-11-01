#!/usr/bin/env bash
# usage: bash peek.sh [r|c|p] [NODE...]
# e.g.   bash peek.sh              (every cluster, every node)
#        bash peek.sh p 1 2 3

source "$(dirname "$0")/nodes.sh"

if [ -n "$1" ]; then
    cluster=$(resolve_cluster "$1") || exit 1
    CLUSTERS=("$cluster")
fi
NODES_TO_USE=("${@:2}")

SSH_OPTS="-i $HOME/.ssh/id_ed25519 -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

for cluster in "${CLUSTERS[@]}"; do
    declare -n NODES="NODES_${cluster}"
    nodes=("${NODES_TO_USE[@]}")
    if [ "${#nodes[@]}" -eq 0 ]; then
        mapfile -t nodes < <(printf '%s\n' "${!NODES[@]}" | sort -n)
    fi
    for node in "${nodes[@]}"; do
        ip="${NODES[$node]}"
        [ -z "$ip" ] && continue
        echo "=== $cluster $node ($ip)"
        ssh $SSH_OPTS "vansh@$ip" bash -s <<'REMOTE' || echo "  (unreachable)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' '{u = u (NR>1 ? " " : "") $1; used+=$2; tot+=$3}
                END {printf "  util %%: %s   mem %d/%d GiB\n", u, used/1024, tot/1024}'
pid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
if [ -n "$pid" ]; then
    echo "  cmd: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-100)"
else
    echo "  cmd: (idle)"
fi
REMOTE
        echo
    done
done
