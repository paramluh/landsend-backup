#!/bin/bash
# gesturemetric (2) HANDS-ONLY judge retrain (tag hands): exactly full.sbatch's train2 + score2 settings, reusing judge/feat_full
# (no re-prep), with --pose-keep 5-10,13,14 = shoulders, elbows, wrists + both hand centroids (+ the 2 hand spreads); all other points'
# x, y, velocity and conf features zeroed in train AND eval. Run INSIDE an existing 2-GPU allocation (no sbatch / srun here):
#   bash judge/hands_run.sh      (uses the allocation's CUDA_VISIBLE_DEVICES; trains + scores on its first GPU, as full.sbatch did)
# -> ckpt_hands.pt, eval2_hands.json, score2_hands.json; progress lines appended to judge/STATUS_FULL.md; log judge/logs/hands-<job>.out
set -uo pipefail
J=/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
ST=$J/STATUS_FULL.md
FD=feat_full
KEEP=5-10,13,14
JID=${SLURM_JOB_ID:-nojob}
LOG=$J/logs/hands-${JID}.out
exec > >(tee -a "$LOG") 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GM_DEVICE=cuda PYTHONUNBUFFERED=1
st() { echo "- $(date -u '+%m-%d %H:%M') UTC [hands job ${JID}]: $*" | tee -a "$ST"; }
die() { st "FAILED: $* (log judge/logs/hands-${JID}.out)"; exit 1; }

IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#GPUS[@]}" -ge 1 ] || die "no GPU in CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "node $(hostname) gpus ${CUDA_VISIBLE_DEVICES}"; nvidia-smi -L
( while sleep 60; do echo "GPU $(date -u +%H:%M:%S) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; done ) &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

NT=$(ls $J/$FD/train/*.npz 2>/dev/null | grep -vc tmp); NE=$(ls $J/$FD/eval/*.npz 2>/dev/null | grep -vc tmp)
[ "$NT" -ge 1900 ] && [ "$NE" -ge 110 ] || die "feat_full incomplete ($NT train, $NE eval)"
st "training HANDS judge (tag hands, --pose-keep $KEEP, reusing judge/$FD: $NT train + $NE eval tracks; d256 L4 ff1024, batch 64x16, eval every 500 steps, patience 6, cap 60 min) on $(hostname) GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES=${GPUS[0]} OMP_NUM_THREADS=16 $PY $J/train2.py --tag hands --feat-dir $FD --d 256 --layers 4 --ff 1024 \
  --batch 64 --steps 20000 --eval-every 500 --patience 6 --max-min 60 --pose-keep $KEEP || die "train2"
$PY - <<'EOF' | tee -a "$ST" || die "eval summary"
import json
r = json.load(open("/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/eval2_hands.json"))
f = r["final"]
print(f"  - HANDS eval gates (top-1@16 p2s, chance 6.25; pose_keep {r.get('pose_keep')}): full {f['full']['p2s']}, words-only {f['words']['p2s']}, "
      f"audio-only {f['audio']['p2s']}, shuffled {f['shuffled']['p2s']}, pose +2 s {f['pose_shift_2s']['p2s']}, tight {f['tight_full']['p2s']} "
      f"(params {r['params_M']}M, {r['train_sides']} train sides, best of {len(r['log'])} evals, {r['wall_s']} s) -> eval2_hands.json, ckpt_hands.pt")
EOF

st "scoring generated runs (score2 hands, pose_keep from ckpt) on GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES=${GPUS[0]} OMP_NUM_THREADS=16 $PY $J/score2.py hands --feat-dir $FD || die "score2"
$PY - <<'EOF' | tee -a "$ST" || die "score summary"
import json
r = json.load(open("/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/score2_hands.json"))
print(f"  - HANDS REAL (20 s protocol, pose_keep {r.get('pose_keep')}): top-1@16 {r['real']['s'][0]} (words {r['real']['sw'][0]}, audio {r['real']['sa'][0]}), shuffled {r['real_shuffled'][0]}")
for g, v in r["gen"].items():
    if g.startswith("ALL"):
        print(f"  - HANDS {g}: acc {v['acc']} (n {v['windows']}), shuffled {v['shuffled']}, words {v['words_only']}, audio {v['audio_only']}, meaning_norm {v['meaning_norm']}")
EOF
st "DONE: ckpt_hands.pt, eval2_hands.json, score2_hands.json"
