#!/bin/bash
#SBATCH -J hf-memtest -p all -w slinky-4 -x slinky-9 -N1 -n1 -c 1 --mem=2G --gres=none -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025/memtest-%j.out
# Memory test of hf_state_host_v2's upload settings (1 worker, xet fixed upload concurrency 2, 1 file ingested at a time) under the SAME 2G cap
# as the MoT overlap steps: 2 x 4 GB random files -> a throwaway PRIVATE test repo; then the repo and the files are deleted (test data only).
set -u
MT=${MT_TAG:-a}; NW=${HFS_WORKERS:-1}; W=/scratch/amartyavarma/hfb/memtest-1025-$MT; R=InternalCan/hfstate-memtest-1025-$MT; PY=/data/home/amartyavarma/venvs/duplex/bin/python; D=/data/home/amartyavarma/setup-logs/hfb
source /data/home/amartyavarma/ppm/scratch_guard.sh
[ -e $W ] && guard_rm $W
mkdir -p $W/src $W/xc
for i in 0 1; do head -c 6000000000 /dev/urandom > $W/src/f$i.bin; done
echo "{\"dirs\": [[\"$W/src\", \"t\"]]}" > $W/spec.json
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$W/xc HF_XET_CHUNK_CACHE_SIZE_BYTES=0
[ "${MT_XETLIM:-1}" = 1 ] && export HF_XET_FIXED_UPLOAD_CONCURRENCY=${HFS_XET_UPC:-2} HF_XET_MAX_CONCURRENT_FILE_INGESTION=1 HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false
( m=0; while sleep 1; do s=$(for q in $(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cgroup.procs 2>/dev/null); do cat /proc/$q/status 2>/dev/null; done | awk '/^Name:/{n=$2} /^RssAnon:/{if (n ~ /python/) t+=$2} END{print t+0}'); [ "$s" -gt "$m" ] && m=$s && echo $m > $W/anon_peak_kb; done ) & SAMP=$!
t0=$(date +%s)
nice -n 19 ionice -c3 $PY $D/hfb_up.py --repo $R --type model --farm $W/farm --spec $W/spec.json --manifest-name MANIFEST.t.tsv --tag memtest --workers $NW --verify-only --no-verify
[ "${MT_THREADS:-0}" != 0 ] && export TOKIO_WORKER_THREADS=$MT_THREADS RAYON_NUM_THREADS=$MT_THREADS
TS=""; [ -n "${MT_TASKSET:-}" ] && TS="taskset -c $MT_TASKSET"   # emulate the overlap step (--cpu-bind=none: all 64 CPUs in the affinity)
TM=""; [ -x /usr/bin/time ] && TM="/usr/bin/time -v"; $TM $TS nice -n 19 ionice -c3 $PY $D/hfb_up.py --child --repo $R --type model --farm $W/farm --spec $W/spec.json --workers $NW 2> $W/time.txt; rc=$?
kill $SAMP 2>/dev/null
echo "== [$MT workers $NW xetlim ${MT_XETLIM:-1} threads ${MT_THREADS:-0} taskset ${MT_TASKSET:-none}] upload rc $rc in $(( $(date +%s) - t0 )) s; python anon RSS peak (sum, 1 s samples) $(cat $W/anon_peak_kb 2>/dev/null) kB; $(tail -2 $W/time.txt | tr '\n' ' ')"
cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak 2>/dev/null | awk '{print "== cgroup memory.peak " $1/1e9 " GB"}'
$PY -I -c "from huggingface_hub import HfApi; import sys; a=HfApi(); a.delete_repo(sys.argv[1]); print('== test repo deleted', sys.argv[1])" $R
guard_rm $W
exit $rc
