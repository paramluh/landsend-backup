#!/bin/bash
#SBATCH -J aloezoomco -p all --ntasks-per-node=8 --gres=gpu:8 -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-%j.log
#SBATCH --signal=B:USR1@300
# aloe-zoom-cotrain (10-21): Aloe (26232's recipe / data / init) + zoom-strangers-5k co-trained 1:1 per step on the 16 PPM nodes
#   (code/train_cotrain.py doc; README.md; LAUNCH.txt). = aloe-v2vox-asr-w15973/aloe_v2vox_raw.sbatch (job 26232) with: run dir, the
#   cotrain trainer + env, 16 nodes (8 aloe hosts = COTRAIN_ALOE_HOSTS = 26232's 8 hosts, 8 zoom hosts), per-group pre-flight, resume pass-through.
# usage (only while the PPM main job is stopped, user's call): sbatch -N16 -w <the 16 PPM hosts> --cpus-per-task=8 --mem=540G aloe_zoom_cotrain.sbatch
#   resume: COTRAIN_RESUME=/scratch/amartyavarma/aloe-zoom-cotrain/resume/<run>/dcp-<step> sbatch ... (same 16 hosts)
# Stop = save-and-exit: touch aloe-zoom-cotrain/control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>): node-local resume state + weights-only
#   export at the next step boundary, then exit 0.
set -uo pipefail
[ "${SLURM_CPUS_PER_TASK:-0}" -ge 4 ] || { echo "ABORT: --cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} < 4"; exit 1; }
Q=/data/home/amartyavarma/ppm/aloe-zoom-cotrain; QT=/data/home/amartyavarma/ppm/aloe-v2vox-raw/tools; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source ${QQA_ENV:-$Q/aloe_zoom_cotrain.env}; set +a
export QQA_INIT MASTER_PORT=29873 COTRAIN_RESUME=${COTRAIN_RESUME:-}
[ "${GEMM_PRECISION:-bf16}" = bf16 ] || { echo "ABORT: bf16 only"; exit 1; }
[ "$VOX_A_GAIN" = raw ] || { echo "ABORT: this run is VOX_A_GAIN=raw"; exit 1; }
[ -f $QQA_INIT/COMPLETE ] || { echo "ABORT: init not COMPLETE"; exit 1; }
[ -f $VOX_ADAPTER/COMPLETE ] || { echo "ABORT: ASR adapter export not COMPLETE"; exit 1; }
[ -f $Q/plan/zoom_table.npz ] && [ -f $Q/plan/ref_cfg_26232.json ] || { echo "ABORT: plan/zoom_table.npz or plan/ref_cfg_26232.json missing"; exit 1; }
for f in $C/model_ppm.py $C/data_ppm.py $C/data_qqa.py $C/data_aloe.py $C/data_aloe_v2.py $C/data_aloe_v2vox.py $C/data_zoom.py $C/chat_ppm.py \
         $C/safety_ppm.py $C/train_cotrain.py $C/rowcache_vox.py $C/lib/audio_s4q.py $C/lib/voxtral_front.py $C/lib/vadapter_s4q.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
NN=$SLURM_NNODES
[ $NN -ge 2 ] || { echo "ABORT: $NN node (needs >= 1 aloe host + >= 1 zoom host)"; exit 1; }
[ $NN = 16 ] || echo "NOTE: $NN nodes (the planned split is 16 = 8 aloe + 8 zoom; another split runs the same maths with micro slots / dummies)"
for h in $(echo $COTRAIN_ALOE_HOSTS | tr , ' '); do echo ",$PPM_HOSTS," | grep -q ",$h," || { echo "ABORT: aloe host $h not in the allocation"; exit 1; }; done
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR; rm -f $QQA_CONTROL_DIR/STOP_AFTER_SAVE
trap 'echo "== USR1 $(TZ=America/Los_Angeles date "+%T PT"): touch STOP_AFTER_SAVE"; touch "$QQA_CONTROL_DIR/STOP_AFTER_SAVE"' USR1
echo "== aloe-zoom-cotrain job $SLURM_JOB_ID on $PPM_HOSTS (world $((8 * NN)); aloe hosts $COTRAIN_ALOE_HOSTS; cpus/task $SLURM_CPUS_PER_TASK) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT run $QQA_RUN resume '${COTRAIN_RESUME}'"
env | grep -E '^(QQA_|ALOE_|VOX_|COTRAIN_|ZOOM_|S4_VOXTRAL|PPM_CD|PPM_DEC_ATTN|S3_MIN)' | sort
FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$FD" -ge $((300 + 80)) ] && echo "/data free ${FD} GB: final consolidated /data export will be attempted (floor 300)" \
  || echo "NOTE: /data free ${FD} GB < 300 floor + 80: the final consolidated /data export will be REFUSED (node-local export-<last> + HF backup = final copy)"
# pre-flight on every node (one task per node): aloe hosts = 26232's (stores node-local + the shared A FLAC symlink + all 8 ledgers);
# every host: /scratch >= 100 GB free (16-node exports ~4.9 GB / node each + a resume state ~36 GB / node), stale /dev/shm NCCL files
srun -N$NN -n$NN --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
H=$(hostname -s)
cd /tmp
if echo ",$COTRAIN_ALOE_HOSTS," | grep -q ",$H,"; then
  OMP_NUM_THREADS=4 '"$PY"' -I '"$QT"'/stage_raw.py --lnodes "" --root "$ALOE_SHARED_FLAC" --stores-only > '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log 2>&1 \
    || { echo "ABORT $H: store staging failed"; tail -3 '"$Q"'/logs/preflight-stores-$SLURM_JOB_ID-$H.log; exit 1; }
  FR=$ALOE_FLAC_ROOT
  if [ -L "$FR" ]; then [ "$(readlink "$FR")" = "$ALOE_SHARED_FLAC" ] || { echo "ABORT $H: $FR -> $(readlink "$FR") != $ALOE_SHARED_FLAC"; exit 1; }
  elif [ -e "$FR" ]; then echo "ABORT $H: $FR exists as a real path (expected the symlink to the shared copy)"; exit 1
  else mkdir -p "$(dirname "$FR")" && ln -s "$ALOE_SHARED_FLAC" "$FR" || { echo "ABORT $H: cannot link $FR"; exit 1; }; fi
  for l in 0 1 2 3 4 5 6 7; do [ -f "$FR/_ledger/DONE_v2vox_node$l.json" ] || { echo "ABORT $H: logical node $l A tracks not staged / verified"; exit 1; }; done
  G="aloe: stores ok, A tracks $(readlink $FR) ledgers ok"
else
  G="zoom"
fi
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ "$F" -ge 100 ] || { echo "ABORT $H: /scratch free ${F} GB < 100"; exit 1; }
mkdir -p "$ALOE_LOCAL_EXPORT_ROOT" "$COTRAIN_RESUME_ROOT"
if [ -n "$COTRAIN_RESUME" ]; then [ -f "$COTRAIN_RESUME/COMPLETE" ] || { echo "ABORT $H: resume state $COTRAIN_RESUME not COMPLETE on this node"; exit 1; }; fi
find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" \) -mmin +5 -delete 2>/dev/null
echo "$H $G; /scratch free ${F} GB; GPUs $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | sort -n | head -1) min SM clock"' || exit 1
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_cotrain.py &
SRUN=$!
while :; do wait $SRUN; rc=$?; kill -0 $SRUN 2>/dev/null || break; done
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc
