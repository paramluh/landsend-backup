#!/bin/bash
# full v3 store build: 4 stores in parallel (one per fast host, CPU only, niced, unbound), output /data/home/amartyavarma/ppm/aloe-v3-ft/stores
V=/data/home/amartyavarma/ppm/aloe-v3-ft; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $V/stores $V/logs; rm -f $V/logs/build.status
H=(slinky-1 slinky-10 slinky-11 slinky-13)
for k in 0 1 2 3; do
  ( srun -p all -w ${H[$k]} -N1 -n1 -c 40 --mem=120G --gres=none -t 120 -J v3build$k bash -c "cd /tmp && nice -n 10 $PY -I $V/tools/build_v3_rows.py --out $V/stores --only-store $k --workers 38" > $V/logs/build-v3n$k.log 2>&1; echo "v3n$k rc $?" >> $V/logs/build.status ) &
done
wait; touch $V/logs/BUILD_DONE
