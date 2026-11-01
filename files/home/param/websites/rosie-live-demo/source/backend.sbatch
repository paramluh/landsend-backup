#!/bin/bash
#SBATCH -N1 -n1 -c 12
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH -t 7-00:00:00
# 10-29 livecall (OPD A/B): ONE PPM live speech backend = the 10-24 clone engine (aloe-m1vox-live-1024 server.py + live_ppm.py, unchanged),
# env from clone_backend.sbatch; CK / BACK / NAME from the environment (sbatch --export). GPUs pinned to this job's own IDX.
# sbatch -w slinky-10 -J lc-base --export=ALL,CK=<export dir>,BACK=8918,NAME=base1 backend.sbatch
J=$SLURM_JOB_ID
IDX=$(scontrol show job -d $J | grep -oP 'IDX:\K[0-9,-]+' | head -1)
GL=$(python3 -c "
s='$IDX'; out=[]
for p in s.split(','):
    a,_,b=p.partition('-'); out+=list(range(int(a),int(b or a)+1))
print(','.join(map(str,out)))")
export CUDA_VISIBLE_DEVICES=$GL
echo "job $J host $(hostname) IDX=$IDX CVD=$CUDA_VISIBLE_DEVICES CK=$CK BACK=$BACK NAME=$NAME $(date)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i $GL
free -g | head -2
if [ -n "${CK_FROM:-}" ]; then   # OPD: hold the GPUs, wait for the converter's READY file (one line = the export dir), then load it
  echo "waiting for $CK_FROM $(date)"
  while true; do if [ -s "$CK_FROM" ]; then CK=$(head -1 "$CK_FROM" | awk '{print $1}'); [ -f "$CK/COMPLETE" ] && [ -f "$CK/meta.json" ] && break; fi; sleep 15; done
  echo "CK from $CK_FROM: $CK $(date)"; free -g | head -2
fi
bash /data/home/amartyavarma/ppm/aloe-posedup-live/tools/gpu_guard.sh "livecall $NAME" || { echo "GPU guard refused; exiting"; exit 3; }
S=/data/home/amartyavarma/ppm/opd/livecall/srv; TOK=$HOME/.s4live_token
source $S/env_live.sh
export PPMLIVE_SWAP_TO=$CK PPMLIVE_DEFAULTS=unbiased PPMLIVE_STATE_DIR=/data/home/amartyavarma/ppm/opd/livecall/state/$NAME PPMLIVE_BIND=0.0.0.0
mkdir -p $PPMLIVE_STATE_DIR/sessions
while true; do $PY -u $S/server.py --ck "$CK" --port $BACK --token-file $TOK; echo "server exited rc=$? $(date)"; sleep 10; done
