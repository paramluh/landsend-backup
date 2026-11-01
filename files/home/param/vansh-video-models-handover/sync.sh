#!/usr/bin/env bash
# usage: bash sync.sh {r|p|l|m} NODE...   (nodes go to this checkout's current branch)
# e.g.   bash sync.sh m 7 8 9 10 11

source "$(dirname "$0")/nodes.sh"

CLUSTER=$(resolve_cluster "$1") || exit 1
NODES_TO_USE=("${@:2}")
declare -n NODES="NODES_${CLUSTER}"
if [ "${#NODES_TO_USE[@]}" -eq 0 ]; then
    echo "usage: bash sync.sh {r|p|l|m} NODE..." >&2
    exit 1
fi

SSH_USER=vansh
SSH_KEY="$HOME/.ssh/id_ed25519"
REPO=$(cd "$(dirname "$0")" && pwd)  # the nodes hold their checkout at this same path
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)

GIT_SSH="GIT_SSH_COMMAND='ssh -i /home/vansh/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new' git -c url.'git@github.com:'.insteadOf='https://github.com/'"
SSH_EXTRA=()
# landsend: the nodes' own keys aren't on GitHub, so fetch with this machine's key via agent forwarding.
if [[ $CLUSTER == LANDSEND ]]; then
    GIT_SSH="GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=accept-new' git -c url.'git@github.com:'.insteadOf='https://github.com/'"
    SSH_EXTRA=(-A -J "$LANDSEND_JUMP")
    if ! ssh-add -l >/dev/null 2>&1; then
        [ -S "$SSH_AUTH_SOCK" ] || { eval "$(ssh-agent -s)" >/dev/null; trap 'ssh-agent -k >/dev/null' EXIT; }
        ssh-add -q "$SSH_KEY"
    fi
fi
REMOTE_CMD="cd $REPO && $GIT_SSH fetch origin $BRANCH && git checkout $BRANCH && $GIT_SSH pull --ff-only origin $BRANCH"

fail=0
for n in "${NODES_TO_USE[@]}"; do
    ip="${NODES[$n]}"
    if [ -z "$ip" ]; then
        echo "=== [node $n] not in $CLUSTER ===" >&2
        fail=1
        continue
    fi
    # richmond: the public IPs are unreachable from inside the cluster.
    [[ $CLUSTER == RICHMOND ]] && ip=richmond-$n
    echo "=== [node $n] $ip -> $BRANCH ==="
    if ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${SSH_EXTRA[@]}" \
        "$SSH_USER@$ip" "$REMOTE_CMD"; then
        echo "=== [node $n] $ip OK ==="
    else
        echo "=== [node $n] $ip FAILED ===" >&2
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "All selected nodes synced to $BRANCH."
else
    echo "Some nodes failed to sync; see above." >&2
fi
exit $fail
