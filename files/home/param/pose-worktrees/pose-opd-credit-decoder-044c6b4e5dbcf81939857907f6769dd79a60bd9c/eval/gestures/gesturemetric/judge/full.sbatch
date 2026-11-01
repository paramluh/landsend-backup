#!/bin/bash
#SBATCH -J gm-judge-full
#SBATCH -w slinky-10
#SBATCH -x slinky-9
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 6:00:00
#SBATCH -o /data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/logs/full-%j.out
# gesturemetric (2) FULL GPU judge run (tag "full", features judge/feat_full; CPU-mini outputs untouched):
#   prep eval 60 convs + train 1,000 convs x both sides x first 7500 tokens (2 shards, one per GPU) -> train2 (d256 L4 ff1024, batch 64x16,
#   eval every 500 steps, patience 6, cap 60 min) -> score2 on out/gen. Progress lines -> judge/STATUS_FULL.md. Fails fast on any error.
set -uo pipefail
J=/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
ST=$J/STATUS_FULL.md
FD=feat_full
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GM_DEVICE=cuda PYTHONUNBUFFERED=1
[ -f "$ST" ] || echo "# gesturemetric (2) FULL GPU judge (tag full, judge/feat_full) — STATUS" > "$ST"
st() { echo "- $(date -u '+%m-%d %H:%M') UTC [job ${SLURM_JOB_ID}]: $*" | tee -a "$ST"; }
die() { st "FAILED: $* (log judge/logs/full-${SLURM_JOB_ID}.out)"; exit 1; }

IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#GPUS[@]}" -ge 2 ] || die "need 2 GPUs, CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'"
echo "node $(hostname) gpus ${CUDA_VISIBLE_DEVICES}"; nvidia-smi -L
# GPU usage line every 60 s into this log
( while sleep 60; do echo "GPU $(date -u +%H:%M:%S) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"; done ) &

st "started on $(hostname) GPUs ${CUDA_VISIBLE_DEVICES}: counting data"
$PY - <<'EOF' | tee -a "$ST" || die "data count"
import sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge")
import prep2
for split, n in (("train", 1000), ("eval", 60)):
    cs = prep2.convs(split, n)
    tok = sum(min(v["sides"][x]["T"], 7500) for v in cs for x in ("assistant", "user"))
    print(f"  - data {split}: {len(cs)} convs x 2 sides = {2 * len(cs)} tracks, {tok} tokens = {tok * 0.08 / 3600:.1f} side-h")
EOF

st "prep (2 shards, one per GPU; eval then train) -> judge/$FD"
pids=()
for i in 0 1; do
  ( export CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=8
    $PY $J/prep2.py eval 60 7500 $i/2 --feat-dir $FD && $PY $J/prep2.py train 1000 7500 $i/2 --feat-dir $FD
  ) > >(sed -u "s/^/[prep$i] /") 2>&1 &
  pids+=($!)
done
for _ in 0 1; do wait -n "${pids[@]}" || die "prep shard exited non-zero"; done
NT=$(ls $J/$FD/train/*.npz 2>/dev/null | grep -vc tmp); NE=$(ls $J/$FD/eval/*.npz 2>/dev/null | grep -vc tmp)
st "prep done: $NT train + $NE eval tracks in judge/$FD"
[ "$NT" -ge 1900 ] && [ "$NE" -ge 110 ] || die "too few prepped tracks ($NT train, $NE eval)"

st "training judge (tag full: d256 L4 ff1024, batch 64x16, eval every 500 steps, patience 6, cap 60 min) on GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES=${GPUS[0]} OMP_NUM_THREADS=16 $PY $J/train2.py --tag full --feat-dir $FD --d 256 --layers 4 --ff 1024 \
  --batch 64 --steps 20000 --eval-every 500 --patience 6 --max-min 60 || die "train2"
$PY - <<'EOF' | tee -a "$ST" || die "eval summary"
import json
r = json.load(open("/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/eval2_full.json"))
f = r["final"]
print(f"  - eval gates (top-1@16 p2s, chance 6.25): full {f['full']['p2s']}, words-only {f['words']['p2s']}, audio-only {f['audio']['p2s']}, "
      f"shuffled {f['shuffled']['p2s']}, pose +2 s {f['pose_shift_2s']['p2s']}, tight {f['tight_full']['p2s']} "
      f"(params {r['params_M']}M, {r['train_sides']} train sides, best of {len(r['log'])} evals, {r['wall_s']} s) -> eval2_full.json, ckpt_full.pt")
EOF

st "scoring generated runs (score2 full) on GPU ${GPUS[0]}"
CUDA_VISIBLE_DEVICES=${GPUS[0]} OMP_NUM_THREADS=16 $PY $J/score2.py full --feat-dir $FD || die "score2"
$PY - <<'EOF' | tee -a "$ST" || die "score summary"
import json
r = json.load(open("/data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/score2_full.json"))
print(f"  - REAL (20 s protocol): top-1@16 {r['real']['s'][0]} (words {r['real']['sw'][0]}, audio {r['real']['sa'][0]}), shuffled {r['real_shuffled'][0]}")
for g, v in r["gen"].items():
    if g.startswith("ALL"):
        print(f"  - {g}: acc {v['acc']} (n {v['windows']}), shuffled {v['shuffled']}, words {v['words_only']}, audio {v['audio_only']}, meaning_norm {v['meaning_norm']}")
EOF
st "DONE: ckpt_full.pt, eval2_full.json, score2_full.json"
