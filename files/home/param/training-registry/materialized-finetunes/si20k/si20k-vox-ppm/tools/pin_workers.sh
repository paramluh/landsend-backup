#!/bin/bash
# Live CPU split inside each training rank's cpuset (no maths change): the rank's main process (all threads) on its first M CPUs, its
# DataLoader workers + their qqa-corrupt-v2 servers on the remaining CPUs, so the CPU-heavy collator never delays the rank's kernel
# launches (the cause of the 5x slowdown: stragglers -> NCCL waits). usage (per node, inside the job): bash pin_workers.sh <M main CPUs>
M=${1:-3}
TR=$(pgrep -u $USER -f "code/train_si20k_vox.py")
for p in $TR; do
  pp=$(ps -o ppid= -p $p | tr -d ' ')
  if echo "$TR" | grep -qx "$pp"; then continue; fi                 # a DataLoader worker (child of a rank), handled below
  cpus=$(grep Cpus_allowed_list /proc/$p/status | awk '{print $2}')
  all=$(python3 -c "
s='$cpus'; out=[]
for part in s.split(','):
    a,_,b=part.partition('-'); out += list(range(int(a), int(b or a)+1))
print(' '.join(map(str,out)))")
  set -- $all; n=$#
  main=$(echo $all | cut -d' ' -f1-$M | tr ' ' ','); work=$(echo $all | cut -d' ' -f$((M+1))-$n | tr ' ' ',')
  taskset -a -c -p $main $p >/dev/null
  kids=$(pgrep -P $p); desc="$kids"; for k in $kids; do desc="$desc $(pgrep -P $k)"; done
  for d in $desc; do taskset -a -c -p $work $d >/dev/null 2>&1; done
  echo "$(hostname -s) rank pid $p cpus $cpus -> main $main, collator ($(echo $desc | wc -w) procs) $work"
done
