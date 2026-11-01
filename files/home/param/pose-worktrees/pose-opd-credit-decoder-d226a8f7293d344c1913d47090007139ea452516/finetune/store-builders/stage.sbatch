#!/bin/bash
#SBATCH -J aloepose-stage -p all -N8 -w slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19 --ntasks-per-node=1 -c 32 --mem=64G -t 0:45:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-ft/logs/stage-%j.log
# the 8 aloe-pose-ft stores (Weka stores/p0..p7) -> every training node's /scratch/amartyavarma/aloe-pose-ft/stores/pK, verified (node digest,
# every file's size); STORES.json (digests the loader asserts); then a CPU check of the plan + real micros on the first node (tests/stage_check.py).
set -uo pipefail
S=/data/home/amartyavarma/ppm/aloe-pose-ft/stores; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export S3_MIN_FRAMES=125 PPM_CD=2
set -a; source /data/home/amartyavarma/ppm/aloe-pose-ft/aloe_pose.env; set +a
for k in 0 1 2 3 4 5 6 7; do [ -f $S/p$k/BUILD_DONE ] || { echo "ABORT: p$k not built"; exit 1; }; done
$PY -I - <<PYEOF || exit 1
import sys, json
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-ft/code/lib")
import s3rows as F
out = {}
for k in range(8):
    n = F.load_node("$S/p%d" % k); pj = json.load(open("$S/p%d/POSE.json" % k))
    out["p%d" % k] = {"node_digest": n["digest"], "rows": int(len(n["length"])), "pose_digest": pj["pose_digest"]}
json.dump(out, open("$S/STORES.json", "w"), indent=1); print("STORES.json", out)
PYEOF
srun bash -c '
S=/data/home/amartyavarma/ppm/aloe-pose-ft/stores; D=/scratch/amartyavarma/aloe-pose-ft/stores; H=$(hostname -s)
mkdir -p $D; df -h /scratch | tail -1
for k in 0 1 2 3 4 5 6 7; do ( rm -rf $D/p$k.tmp; cp -r $S/p$k $D/p$k.tmp && rm -rf $D/p$k && mv $D/p$k.tmp $D/p$k ) & done; wait
for k in 0 1 2 3 4 5 6 7; do
  a=$(cd $S/p$k && find . -type f -printf "%p %s\n" | sort | md5sum); b=$(cd $D/p$k && find . -type f -printf "%p %s\n" | sort | md5sum)
  [ "$a" = "$b" ] || { echo "$H p$k MISMATCH"; exit 1; }
done
S3_MIN_FRAMES=125 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I -c "
import sys, json; sys.path.insert(0, \"/data/home/amartyavarma/ppm/aloe-pose-ft/code/lib\"); import s3rows as F
ref = json.load(open(\"$S/STORES.json\"))
for k in range(8):
    assert F.load_node(\"$D/p%d\" % k)[\"digest\"] == ref[\"p%d\" % k][\"node_digest\"], k
print(\"$H all 8 stores staged + verified\")"
df -h /scratch | tail -1' || exit 1
srun -N1 -n1 -w slinky-10 bash -c "cd /tmp; S3_MIN_FRAMES=125 PPM_CD=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -I /data/home/amartyavarma/ppm/aloe-pose-ft/tests/stage_check.py" || exit 1
echo "== stage done $(TZ=America/Los_Angeles date '+%T PT')"
