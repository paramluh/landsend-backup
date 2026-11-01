#!/bin/bash
# si20k-vox: retry the partial launch (user GO 10-18 PT) every 2 min until the strict keep spans cover >= 98 % of the final conversations;
# launch_both.sh aborts before submitting anything while coverage is short. Login node: file checks + sbatch only. Log: logs/auto_launch.log
L=/data/home/amartyavarma/ppm/si20k-vox-ppm/logs/auto_launch.log
for i in $(seq 1 90); do
  out=$(SI20K_PARTIAL=1 SPANS=/data/home/amartyavarma/data/si20k-tts/filter/keep_spans.tsv SPANS_STRICT_RE='STRICT.*0\.233' bash /data/home/amartyavarma/ppm/si20k-vox-ppm/tools/launch_both.sh 2>&1)
  echo "$(TZ=America/Los_Angeles date '+%F %T PT') attempt $i: $(echo "$out" | grep -E 'selection:|ABORT|launched' | tr '\n' ' ' | cut -c1-400)" >> $L
  echo "$out" | grep -q "^launched" && { echo "$out" >> $L; exit 0; }
  echo "$out" | grep -q "ABORT: selection / spans coverage\|not the strict rule" || { echo "$out" >> $L; exit 1; }
  sleep 120
done
