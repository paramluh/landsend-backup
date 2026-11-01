#!/bin/bash
#SBATCH -J ppm811f-savenow -p all -w slinky-15 -N1 -n1 -c1 --mem=512M -t 2:00:00 -o /data/home/amartyavarma/ppm/code-811/fresh/logs/savenow16890-%j.out
# one-shot: at step 16889 of job 27100 touch SAVE_NOW, so the hourly saves land ~27 steps away from the abandoned branch's HF-uploaded
# dcp-16914 / dcp-16967 / dcp-17012 (HF done markers are keyed by step only)
L=/data/home/amartyavarma/ppm/logs/ppm-27100.log; CT=/data/home/amartyavarma/ppm/control/ppm-v1
until grep -aq "\] step 16889/" $L; do squeue -h -j 27100 | grep -q . || { echo "job gone"; exit 1; }; sleep 15; done
[ -e $CT/SAVE_NOW ] || echo "fresh16861: 2nd save-cadence shift away from the old branch's HF dcp-16914/16967/17012" > $CT/SAVE_NOW
echo "touched SAVE_NOW $(TZ=America/Los_Angeles date '+%T PT')"
until grep -aq "SAVE_NOW at step 1689" $L; do sleep 10; done; grep -a "SAVE_NOW at step 1689" $L
