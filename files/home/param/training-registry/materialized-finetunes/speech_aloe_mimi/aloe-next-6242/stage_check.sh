#!/bin/bash
# per node: the 4 Model-1 stores (m1n0..m1n3, s3-duplex-rows-v1) at /scratch/amartyavarma/aloe-ft/stores/m1nK (21980's staged copies);
# a store whose digest / lengths / file sizes differ from the /data source (or is missing) is re-copied (= aloe-ft/stage.sbatch).
S=/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores; D=/scratch/amartyavarma/aloe-ft/stores; H=$(hostname -s)
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $D; echo "$H scratch: $(df -h /scratch | tail -1)"
chk() {
$PY -I - "$1" <<'PYEOF'
import os, sys; sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2-ft/code/lib")
import s3rows as F
k = int(sys.argv[1]); S = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores"; D = "/scratch/amartyavarma/aloe-ft/stores"
try:
    a, b = F.load_node(f"{S}/m1n{k}"), F.load_node(f"{D}/m1n{k}")
    assert a["digest"] == b["digest"] and (a["length"] == b["length"]).all()
    for f in sorted(os.listdir(f"{S}/m1n{k}")):
        assert os.path.getsize(f"{S}/m1n{k}/{f}") == os.path.getsize(f"{D}/m1n{k}/{f}"), f
    print(f"m1n{k} ok rows {len(b['length'])} digest {b['digest']}")
except Exception as e:
    print(f"m1n{k} BAD {type(e).__name__} {e}"); sys.exit(1)
PYEOF
}
rc=0
for k in 0 1 2 3; do
  if out=$(chk $k); then echo "$H $out (already staged)"; continue; fi
  echo "$H $out -> restaging"
  rm -rf $D/m1n$k.tmp; cp -r $S/m1n$k $D/m1n$k.tmp && rm -rf $D/m1n$k && mv $D/m1n$k.tmp $D/m1n$k
  if out=$(chk $k); then echo "$H $out (restaged)"; else echo "$H $out STILL BAD"; rc=1; fi
done
exit $rc
