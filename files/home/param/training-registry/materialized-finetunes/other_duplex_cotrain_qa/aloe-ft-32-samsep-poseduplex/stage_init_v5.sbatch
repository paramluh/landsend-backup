#!/bin/bash
#SBATCH -J pdx-stage -p all --ntasks-per-node=1 -c 32 --mem=64G --gres=none -t 01:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/stage-%j.log
# poseduplex init staging (CPU only, one task per training host, max CPU): pull ONLY the v5 export files each host's 8 ranks read at world 64
# (data/init_subset_v5.json from tools/init_subset_v5.py, +-1 source margin; + rep.pt / meta.json / COMPLETE) from OUR private HF repo
# InternalCan/ppmvox-v5-17850 into /scratch/amartyavarma/poseduplex-init/<tag> (sha256 == HF LFS per file; the dir appears only when complete).
# Never touches anything else on the host. TAG: an explicit w-<N> (preferred: the export the HF watcher verified) or 'auto' (= the newest w-N with
# all 128 rank files + rep.pt / meta.json / COMPLETE on HF, resolved ONCE here so every host stages the same tag).
# usage (login node: sbatch only):
#   POSEDUP_HOSTS=h0,h1,h2,h3,h4,h5,h6,h7 sbatch -N8 -w $POSEDUP_HOSTS --export=ALL /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/stage_init_v5.sbatch [w-<N>|auto]
#   (the host order = the training job's PPM_HOSTS order: host index h holds ranks 8h..8h+7). Writes control/INIT_TAG on success.
set -uo pipefail
P=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex; PY=/data/home/amartyavarma/venvs/duplex/bin/python
: "${POSEDUP_HOSTS:?set POSEDUP_HOSTS (8 hosts, training order)}"
[ "$(echo $POSEDUP_HOSTS | tr , '\n' | wc -l)" = 8 ] || { echo "ABORT: POSEDUP_HOSTS needs 8 hosts"; exit 1; }
echo ",$POSEDUP_HOSTS," | grep -qE ',slinky-(9|10),' && { echo "ABORT: never slinky-9 / slinky-10"; exit 1; }
ALLOC=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort | paste -sd,)
[ "$ALLOC" = "$(echo $POSEDUP_HOSTS | tr , '\n' | sort | paste -sd,)" ] || { echo "ABORT: allocation $ALLOC != POSEDUP_HOSTS $POSEDUP_HOSTS"; exit 1; }
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 STAGE_THREADS=${STAGE_THREADS:-16} PYTHONUNBUFFERED=1
TAG=${1:-auto}
[ "$TAG" = auto ] && TAG=$($PY -I $P/tools/stage_v5.py resolve) && echo "resolved auto -> $TAG"
[[ "$TAG" =~ ^w-[0-9]+$ ]] || { echo "ABORT: bad tag '$TAG'"; exit 1; }
mkdir -p $P/control
echo "== stage v5 $TAG on $POSEDUP_HOSTS $(TZ=America/Los_Angeles date '+%F %T PT') job $SLURM_JOB_ID"
srun -N8 -n8 --ntasks-per-node=1 -c 32 --kill-on-bad-exit=0 bash -c '
set -uo pipefail
H=$(hostname -s); h=$(( $(echo '"$POSEDUP_HOSTS"' | tr , "\n" | grep -nx $H | cut -d: -f1) - 1 ))
[ $h -ge 0 ] || { echo "ABORT $H not in POSEDUP_HOSTS"; exit 1; }
DEST=/scratch/amartyavarma/poseduplex-init
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ -f $DEST/'"$TAG"'/COMPLETE ] || [ $F -ge 50 ] || { echo "ABORT $H: /scratch free ${F} GB < 50 (~17 GB init subset + 30 floor)"; exit 1; }
mkdir -p $DEST; cd /tmp
t0=$(date +%s)
nice -n 5 '"$PY"' -I '"$P"'/tools/stage_v5.py '"$TAG"' '"$P"'/data/init_subset_v5.json $h $DEST || { echo "ABORT $H: staging failed"; exit 1; }
echo "$H (host $h): $(du -sh $DEST/'"$TAG"' | cut -f1) in $(( $(date +%s) - t0 )) s; /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d " ")"'
rc=$?
[ $rc = 0 ] && echo "$TAG" > $P/control/INIT_TAG && echo "== staged $TAG on all 8 hosts; control/INIT_TAG = $TAG" || echo "== staging FAILED rc $rc"
exit $rc
