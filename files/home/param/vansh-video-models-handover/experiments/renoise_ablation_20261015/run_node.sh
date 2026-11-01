set -uo pipefail
arm=$1
first=$2
root=/mnt/paramfs/param/outputs/renoise-ablation-20261015
cd "$root/code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    shard=$((first + gpu))
    (
        CUDA_VISIBLE_DEVICES="$gpu" /home/param/vansh-video-models/.venv/bin/python bimodal/sampler.py "$root/configs/$arm.yaml" --shard "$shard" --num-shards 16 2>&1 | tee "$root/logs/$arm-$shard.log"
        status=${PIPESTATUS[0]}
        echo "$status" > "$root/logs/$arm-$shard.exit"
        exit "$status"
    ) &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
