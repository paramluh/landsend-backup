#!/bin/bash
# D35 checks on ONE GPU of a spare node (no checkpoints, nothing touches the paused run). usage (inside srun on the node): bash tests/test_d35.sh
# (a) tests/test_d35.py: the gradient-scaling unit test (synthetic) + the trainer's ramp schedule under prod_ppm.env
# (b) resolved config diff: pre-D35 code + pre-D35 env (= job 19127's resume config) vs new code + new env; and vs job 19127's logged cfg
set -u
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
W=/scratch/amartyavarma/ppm/d35test; rm -rf $W; mkdir -p $W/oldcode
for f in $C/*; do ln -s "$f" "$W/oldcode/$(basename "$f")"; done
ln -sf $C/train_ppm.py.bak-pre-d35 $W/oldcode/train_ppm.py; ln -sf $C/model_ppm.py.bak-pre-d35 $W/oldcode/model_ppm.py
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=$W/triton TORCHINDUCTOR_CACHE_DIR=$W/inductor PYTHONDONTWRITEBYTECODE=1
echo "== (a) unit test"
( set -a; source $C/prod_ppm.env; set +a; $PY $C/tests/test_d35.py $C ); echo "unit rc $?"
echo "== (b) env diff (PPM_* after sourcing; pre-D35 env vs new env)"
diff <(env -i bash -c "set -a; source $C/prod_ppm.env.bak-pre-d35; env | grep '^PPM_' | sort") \
     <(env -i bash -c "set -a; source $C/prod_ppm.env; env | grep '^PPM_' | sort")
cfg() {   # $1 code dir, $2 env file -> resolved CFG json (the trainer's own module-level resolution; as ppm.sbatch sets PPM_RUN for job 19127)
  bash -c "unset \$(compgen -e | grep '^PPM_'); set -a; source $2; set +a; export PPM_RUN=ppm-v1-j19127 PPM_RUN_BASE=ppm-v1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    PYTHONDONTWRITEBYTECODE=1; cd /; $PY -c 'import sys, json; sys.path.insert(0, \"$1\"); import train_ppm as T; print(json.dumps(T.CFG, sort_keys=True))'" 2>/dev/null | tail -1
}
cfg $W/oldcode $C/prod_ppm.env.bak-pre-d35 > $W/cfg_old.json
cfg $C $C/prod_ppm.env > $W/cfg_new.json
grep -m1 -o 'cfg {.*' /data/home/amartyavarma/ppm/logs/ppm-19127.log | sed 's/^cfg //' > $W/cfg_19127.json
$PY - $W <<'EOF'
import json, sys
W = sys.argv[1]
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
    return o
old, new, j = (flat(json.load(open(f"{W}/cfg_{n}.json"))) for n in ("old", "new", "19127"))
for nm, a in (("pre-D35 code+env", old), ("job 19127 logged cfg", j)):
    d = {k: (a.get(k, "<absent>"), new.get(k, "<absent>")) for k in sorted(set(a) | set(new)) if a.get(k, "<absent>") != new.get(k, "<absent>")}
    print(f"== resolved CFG diff vs {nm} ({len(a)} keys vs {len(new)}): {d}")
EOF
rm -rf $W
