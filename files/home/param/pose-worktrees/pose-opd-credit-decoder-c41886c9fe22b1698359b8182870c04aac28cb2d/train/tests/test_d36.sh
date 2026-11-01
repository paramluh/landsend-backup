#!/bin/bash
# D36 checks on ONE spare node (inside srun on slinky-7: it carries pose / speech / text stores). Nothing touches the live run; every
# checkpoint is node-local /scratch (deleted at the end); logs / small JSON records -> $O on /data.
# usage: [D36_PARTS="1 2 3 4"] bash tests/test_d36.sh [out dir]
#  (1) tests/test_d36.py         unit: load_local drop_opt (+ negatives), CFG / stats layout under the knob (1 GPU)
#  (2) resolved config diffs      old code + prod env vs new code + prod env (must be identical); + the D36 line uncommented (only the 2 keys)
#  (3) tests/test_d36_ckpt.py     tiny-width resume / equivalence phases a, on, on2, old, off, off2, revive (8 GPUs) + tests/cmp_d36.py
#  (4) the REAL trainer main()    1 layer each, full width: A head on 2 steps -> save; B PPM_POSEHEAD_OFF=1 resume, 2 steps, evals
set -u
O=${1:-/data/home/amartyavarma/ppm/teststore/d36}
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
W=/scratch/amartyavarma/ppm/d36test
rm -rf $W; mkdir -p $W/oldcode $W/ck $O
for f in $C/*; do ln -s "$f" "$W/oldcode/$(basename "$f")"; done
for f in model_ppm.py train_ppm.py safety_ppm.py; do ln -sf $C/$f.bak-pre-d36 $W/oldcode/$f; done
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm TORCHINDUCTOR_CACHE_DIR=$W/inductor PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=" ${D36_PARTS:-1 2 3 4} "
if [[ $P == *" 1 "* ]]; then
echo "== (1) unit test $(date +%T)"
( set -a; source $C/prod_ppm.env; set +a; CUDA_VISIBLE_DEVICES=0 $PY $C/tests/test_d36.py $C $W ) > $O/unit.log 2>&1; echo "unit rc $?"; grep -E "PASS|FAIL" $O/unit.log | tail -30

fi
if [[ $P == *" 2 "* ]]; then
echo "== (2) resolved config diffs"
sed 's/^# *PPM_POSEHEAD_OFF=1/PPM_POSEHEAD_OFF=1/' $C/prod_ppm.env > $W/prod_d36_uncommented.env
grep -n "POSEHEAD_OFF" $W/prod_d36_uncommented.env
cfg() {   # $1 code dir, $2 env file -> resolved CFG json (the trainer's own module-level resolution, as ppm.sbatch sets PPM_RUN for job 20492)
  bash -c "unset \$(compgen -e | grep '^PPM_'); set -a; source $2; set +a; export PPM_RUN=ppm-v1-j20492 PPM_RUN_BASE=ppm-v1 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    PYTHONDONTWRITEBYTECODE=1; cd /; $PY -c 'import sys, json; sys.path.insert(0, \"$1\"); import train_ppm as T; print(json.dumps(T.CFG, sort_keys=True))'" 2>/dev/null | tail -1
}
cfg $W/oldcode $C/prod_ppm.env > $W/cfg_old.json
cfg $C $C/prod_ppm.env > $W/cfg_new.json
cfg $C $W/prod_d36_uncommented.env > $W/cfg_d36.json
grep -m1 -o 'cfg {.*' /data/home/amartyavarma/ppm/logs/ppm-20492.log | sed 's/^cfg //' > $W/cfg_20492.json
$PY - $W <<'EOF' | tee $O/cfgdiff.log
import json, sys
W = sys.argv[1]
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
    return o
old, new, d36, j = (flat(json.load(open(f"{W}/cfg_{n}.json"))) for n in ("old", "new", "d36", "20492"))
def dd(a, b):
    return {k: (a.get(k, "<absent>"), b.get(k, "<absent>")) for k in sorted(set(a) | set(b)) if a.get(k, "<absent>") != b.get(k, "<absent>")}
print(f"== old code + prod env vs new code + prod env ({len(old)} vs {len(new)} keys): {dd(old, new)}")
print(f"== job 20492 logged cfg vs new code + prod env: {dd(j, new)}")
print(f"== new code: prod env vs prod env with the D36 line uncommented: {dd(new, d36)}")
EOF

fi
if [[ $P == *" 3 "* ]]; then
echo "== (3) tiny-width resume / equivalence phases $(date +%T)"
TR="$PY -m torch.distributed.run --standalone --nproc_per_node 8"
(
  set -a; source $C/prod_ppm.env; set +a
  export PPM_RUN=d36ck PPM_RUN_BASE=d36ck PPM_LAYERS=1 PPM_TOWER_LAYERS=1 PPM_DEC_LAYERS=1 PPM_DEC_FAST=0 PPM_DL_WORKERS=0 PPM_EXTRA_PLAN= PPM_EXTRA_FROM_STEP=-1
  export PPM_TRAIN_EXCLUDE_SALTS= PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1 PPM_LOG_ROOT=$W/runs PPM_SAVE_ROOT=$W/ckpt PPM_POSEHEAD_DETACH_FROM=0 PPM_POSEHEAD_DETACH_RAMP=0
  unset PPM_POSEHEAD_OFF
  $TR $C/tests/test_d36_ckpt.py $C a $W/ck > $O/ck_a.log 2>&1 && echo "ck_a ok" &&
  $TR $C/tests/test_d36_ckpt.py $C on $W/ck > $O/ck_on.log 2>&1 && echo "ck_on ok" &&
  $TR $C/tests/test_d36_ckpt.py $C on2 $W/ck > $O/ck_on2.log 2>&1 && echo "ck_on2 ok" &&
  $TR $C/tests/test_d36_ckpt.py $W/oldcode old $W/ck > $O/ck_old.log 2>&1 && echo "ck_old ok" &&
  PPM_POSEHEAD_OFF=1 $TR $C/tests/test_d36_ckpt.py $C off $W/ck > $O/ck_off.log 2>&1 && echo "ck_off ok" &&
  PPM_POSEHEAD_OFF=1 $TR $C/tests/test_d36_ckpt.py $C off2 $W/ck > $O/ck_off2.log 2>&1 && echo "ck_off2 ok" &&
  $TR $C/tests/test_d36_ckpt.py $C revive $W/ck > $O/ck_revive.log 2>&1 && echo "ck_revive ok"
  echo "phases rc $?"
)
for p in a on on2 old off off2 revive; do grep -h "RESULT\|Error\|assert" $O/ck_$p.log | head -3 | cut -c1-600; done
echo "== cmp"
$PY $C/tests/cmp_d36.py $W/ck on on2 > $O/cmp_on_on2.log 2>&1; echo "on vs on2 (noise floor) rc $?"; tail -4 $O/cmp_on_on2.log
$PY $C/tests/cmp_d36.py $W/ck old on > $O/cmp_old_on.log 2>&1; echo "(c) old code vs new code, knob unset, rc $?"; tail -4 $O/cmp_old_on.log
$PY $C/tests/cmp_d36.py $W/ck on off --head-removed > $O/cmp_on_off.log 2>&1; echo "(b) knob unset s=0 vs knob on rc $?"; tail -12 $O/cmp_on_off.log
cp $W/ck/*.json $O/ 2>/dev/null
du -sh $W/ck; rm -rf $W/ck $W/ckpt

fi
if [[ $P == *" 4 "* ]]; then
echo "== (4) real trainer main(), 1 layer each, full width $(date +%T)"
base() {
  set -a; source $C/prod_ppm.env; set +a
  export PPM_DL_WORKERS=0 PPM_EXTRA_PLAN= PPM_EXTRA_FROM_STEP=-1 PPM_TRAIN_EXCLUDE_SALTS= PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1
  export PPM_LOG_ROOT=$W/runs PPM_CONTROL_DIR=$W/control PPM_SAVE_ROOT=$W/ckpt PPM_EXPORT_ROOT=$W/export PPM_EXPORT_MIN_FREE_GB=1
  export PPM_EVAL_EVERY_MIN=0 PPM_SAVE_EVERY_MIN=0 PPM_EXPORT_EVERY_MIN=0 PPM_PROFILE=0
  export PPM_RUN=d36rt PPM_RUN_BASE=d36rt PPM_LAYERS=1 PPM_TOWER_LAYERS=1 PPM_DEC_LAYERS=1 PPM_KEEP_CKPT=1
  export PPM_CODEP=0 PPM_XDEP=0 PPM_EVAL_S3_DIR= PPM_EVAL_GEMMA_S3_ROOT= PPM_EVAL_PER_TASK=1 PPM_POSEHEAD_DETACH_FROM=0 PPM_POSEHEAD_DETACH_RAMP=0
  unset PPM_POSEHEAD_OFF
  mkdir -p $PPM_LOG_ROOT $PPM_CONTROL_DIR
}
( base; export PPM_MAX_STEPS=2 PPM_RESUME_FROM= PPM_FINAL_SAVE=1; bash $C/run_local.sh train_ppm.py ) > $O/rt_a.log 2>&1; echo "rt_a rc $?"
( base; export PPM_POSEHEAD_OFF=1 PPM_MAX_STEPS=4 PPM_RESUME_FROM=auto PPM_FINAL_SAVE=0; bash $C/run_local.sh train_ppm.py ) > $O/rt_b.log 2>&1; echo "rt_b rc $?"
grep -h "D36\|\[resume\]\|Error\|assert\|STOPPED\|^\[.*done" $O/rt_b.log | cut -c1-400
$PY - $W/runs $O <<'EOF' | tee $O/rt_metrics.log
import json, sys, glob
for f in sorted(glob.glob(f"{sys.argv[1]}/*/metrics.jsonl")):
    for l in open(f):
        r = json.loads(l)
        ks = sorted(k for k in r if "pose_head" in k)
        print(f"{f.split('/')[-2]} step {r['step']} {'eval' if any(k.startswith('eval') for k in r) else 'train'}: pose_head keys {ks}; "
              f"groups lr {[k for k in r if k.startswith('train/lr_')]}")
EOF
du -sh $W/ckpt 2>/dev/null
fi
rm -rf $W
echo "== done $(date +%T)"
