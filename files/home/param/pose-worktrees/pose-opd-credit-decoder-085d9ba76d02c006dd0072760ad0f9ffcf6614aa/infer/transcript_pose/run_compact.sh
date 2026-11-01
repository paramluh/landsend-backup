#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python "$recipe_dir/infer.py" --recipe compact-all "$@"
