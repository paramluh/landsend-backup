#!/bin/bash
# Copy the 'mimidec' A FLACs (Mimi decodes of purged conversations, staged node-local by tools/stage_m1vox.py) that the m1vox / v2vox train
# items read, from the training nodes' /scratch to loudness_audit/_tmp_mimidec (deleted after the audit). Light reads only: job 22956 overlap,
# 2 CPUs, no CPU binding, nice 19 + idle IO class. Node i = the m1vox staging host i (logs/stage-node<i>-<host>.log).
D=/data/home/amartyavarma/ppm/aloe-m1vox-ft/loudness_audit
ROOT=/scratch/amartyavarma/aloe-m1vox-ft/flac
HOSTS=(slinky-0 slinky-2 slinky-3 slinky-5 slinky-6 slinky-8 slinky-9 slinky-12)
mkdir -p $D/_tmp_mimidec
srun --overlap --jobid 22958 -w slinky-4 -N1 -n1 -c 1 --cpu-bind=none /data/home/amartyavarma/venvs/qwen-s2/bin/python -I - <<'EOF'
import json, collections
D = "/data/home/amartyavarma/ppm/aloe-m1vox-ft/loudness_audit"
need = json.load(open("/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/NEEDS.json"))
want = sorted({json.loads(x)["flac"] for x in open(f"{D}/items.jsonl") if json.loads(x)["kind"] == "mimidec"})
where = collections.defaultdict(list)
for node, ts in need.items():
    for t in ts:
        if t["kind"] == "mimidec":
            where[t["rel"]].append(int(node))
lists = collections.defaultdict(list)
for rel in want:
    assert where[rel], rel
    lists[min(where[rel])].append(rel)
for n in range(8):
    open(f"{D}/_tmp_mimidec/list_node{n}.txt", "w").write("".join(r + "\n" for r in lists.get(n, [])))
print("mimidec files wanted", len(want), {n: len(v) for n, v in sorted(lists.items())})
EOF
for i in 0 1 2 3 4 5 6 7; do
  L=$D/_tmp_mimidec/list_node$i.txt
  [ -s "$L" ] || continue
  srun --overlap --jobid 22956 -w ${HOSTS[$i]} -N1 -n1 -c 2 --cpu-bind=none nice -n 19 ionice -c 3 \
    tar -C $ROOT -cf - -T $L | tar -C $D/_tmp_mimidec -xf - &
done
wait
echo "copied: $(find $D/_tmp_mimidec/_mimidec -name '*.flac' | wc -l) files, $(du -sh $D/_tmp_mimidec/_mimidec | cut -f1)"
