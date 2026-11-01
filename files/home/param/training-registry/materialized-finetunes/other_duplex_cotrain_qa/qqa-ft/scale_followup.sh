#!/bin/bash
# when a node's own tag finishes, it joins the slower tags (claim mode; local scratch copies): 7 -> 25126, 15 -> 25126, 18 -> 5913.
QE=/data/home/amartyavarma/ppm/qqa-ft/qa_eval; SC=/scratch/amartyavarma/ppm-qqa-scale; LOG=/data/home/amartyavarma/ppm/qqa-ft/logs/scale_driver.log
R=/data/home/amartyavarma/qwen-1ch/stage4/qa_eval/results
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
follow() {  # job node tag ck
  while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 20; done
  if [ -f $R/$3/score.json ]; then echo "[$(pt)] $2 free, $3 already scored" >> $LOG; return; fi
  J=$(bash $QE/launch_eval.sh $3 $4 $2 "0 1 2 3 4 5 6 7"); echo "[$(pt)] $2 free -> joins $3 (job $J)" >> $LOG
}
follow 21840 slinky-7 ppmqqa-scale-25126 $SC/export-59 &
follow 21864 slinky-15 ppmqqa-scale-25126 $SC/export-59 &
follow 21857 slinky-18 ppmqqa-scale-5913 $SC/export-14 &
wait
