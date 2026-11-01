#!/bin/bash
# 10-26: restart a playground unit idle-only so it loads the current scripts:   roll_flexpair.sh <job id> [vansh|dup|mono|hot]
# NEVER with work in flight (a reshuffle once killed a worker mid-run): 1) drain every worker id of the job (mono<job>, dup<job>, flexmono<job>,
# d<job>: they claim nothing new), 2) wait until no run claimed by them is queued / running, 3) wait until the job's GPUs are idle (< 10 % for
# 3 checks: no render in flight), then scancel + resubmit on the same node; the drain flags are removed after.
J=$1; KIND=${2:-vansh}; G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb; D=$G/flex/drain; mkdir -p $D
N=$(squeue -h -j $J -o %N); [ -n "$N" ] || { echo "$J not running"; exit 1; }
IDX=$(scontrol show job -d $J | grep -oP 'IDX:\K[0-9,-]+' | head -1); GL=$(python3 -c "
s='$IDX'; out=[]
for p in s.split(','):
    a,_,b=p.partition('-'); out+=list(range(int(a),int(b or a)+1))
print(','.join(map(str,out)))")
W="mono$J dup$J flexmono$J d$J"; for w in $W; do touch $D/$w; done; sleep 2
busy(){ local c s; for c in $G/runs/*/CLAIMED.*$J; do [ -e "$c" ] || continue
  s=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('state'))" "$(dirname "$c")/status.json" 2>/dev/null)
  [ "$s" = running ] || [ "$s" = queued ] && return 0; done; return 1; }
for i in $(seq 1200); do busy || break; [ $((i % 20)) = 1 ] && echo "$(date -u +%T) $J: a run is in flight, draining"; sleep 3; done
busy && { echo "$J still busy after 1 h; not restarted"; for w in $W; do rm -f $D/$w; done; exit 1; }
ok=0; for i in $(seq 600); do
  u=$(srun --jobid $J --overlap -w $N -x slinky-9 -n 1 -c 1 --mem=1G nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GL 2>/dev/null | sort -n | tail -1 | tr -d ' ')
  [ -n "$u" ] && [ "$u" -lt 10 ] && ok=$((ok + 1)) || ok=0; [ $ok -ge 3 ] && break; sleep 3; done
[ $ok -ge 3 ] || { echo "$J GPUs never idle; not restarted"; for w in $W; do rm -f $D/$w; done; exit 1; }
busy && { echo "$J: a run appeared; not restarted"; for w in $W; do rm -f $D/$w; done; exit 1; }
scancel $J; for i in $(seq 15); do squeue -h -j $J 2>/dev/null | grep -q . || break; sleep 2; done
for w in $W; do rm -f $D/$w; done; rm -f $G/flex/mono/$J
case $KIND in
  dup)  sbatch -w $N -x slinky-9 --gres=gpu:2 -c 6 --mem=110G -J genweb-dup $G/unit.sbatch dup;;
  mono) sbatch -w $N -x slinky-9 --gres=gpu:2 -c 8 --mem=120G -J genweb-u-mono $G/unit.sbatch mono;;
  hot)  sbatch -w $N -x slinky-9 --gres=gpu:2 -c 8 --mem=120G -t 24:00:00 -J genweb-u-hot -o "$G/logs/%x-%j.out" $G/hot_unit.sh;;
  none) echo "$J drained + cancelled (no resubmit)";;
  vanshonly) sbatch -w $N -x slinky-9 --gres=gpu:2 -c 6 --mem=110G -J genweb-u-vanshonly $G/unit.sbatch vanshonly;;
  paramonly) sbatch -w $N -x slinky-9 --gres=gpu:2 -c 6 --mem=110G -J genweb-u-param $G/unit.sbatch paramonly;;
  *)    sbatch -w $N -x slinky-9 --gres=gpu:2 -c 8 --mem=120G -J genweb-u-vansh $G/unit.sbatch vansh;;
esac
