#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 12 --gres=gpu:4 --gres-flags=disable-binding --mem=200G -t 45 -J pmpo-gennp
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/gennp-%j.out
# NO-PERSONA pose-only generation of all 20 candidates (rows-np: R-000..R-019, voice clip on, persona off, 60 s windows), same settings as run_gen.sh (gen_poseonly.py: export-357,
# text + audio forced, pose generated, pose CFG 3 / top-k 30 / temp 1). slinky-4 GPUs must be 2-5 (0-1 demo, 6-7 throttled; guarded).
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
CK=/data/home/amartyavarma/ppm/aloe-pose-mono-8020/export-357; G=$W/work/gen_poseonly.py
for b in 0 1 6 7; do case ",$CUDA_VISIBLE_DEVICES," in *,$b,*) echo "ABORT: allocation holds GPU $b ($CUDA_VISIBLE_DEVICES)"; exit 3;; esac; done
# rows (CPU, the run's env + code): rows-np/R-000..R-019, then the generation
( export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
  source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh; set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env; set +a
  [ -s $W/rows-np/rows.json ] || /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $W/work/build_rows_np.py ) > $W/logs/rows-np.log 2>&1
[ -s $W/rows-np/rows.json ] || { echo "ABORT: rows-np failed"; tail -5 $W/logs/rows-np.log; exit 4; }
echo "$(TZ=America/Los_Angeles date '+%F %T PT') rows-np ready"
source $FI/env_fast.sh
IFS=, read -ra ALL <<< "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES
echo "$(TZ=America/Los_Angeles date '+%F %T PT') start"
( export CUDA_VISIBLE_DEVICES=${ALL[0]},${ALL[1]}; exec nice -n 5 $PY -u $G --ckpt $CK --rows-dir $W/rows-np --out $W --rows R-000,R-001,R-002,R-003,R-004,R-005,R-006,R-007,R-008,R-009 --seed 20261023 ) > $W/logs/gennp-worker-0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=${ALL[2]},${ALL[3]}; exec nice -n 5 $PY -u $G --ckpt $CK --rows-dir $W/rows-np --out $W --rows R-010,R-011,R-012,R-013,R-014,R-015,R-016,R-017,R-018,R-019 --seed 20261024 ) > $W/logs/gennp-worker-1.log 2>&1 &
wait
echo "$(TZ=America/Los_Angeles date '+%F %T PT') DONE: $(ls $W/runs-poseonly/*/DONE 2>/dev/null | wc -l)/20"
