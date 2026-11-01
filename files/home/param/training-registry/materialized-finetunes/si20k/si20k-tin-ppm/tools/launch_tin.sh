#!/bin/bash
# si20k-tin: submit BOTH text-in fine-tunes on the same hosts as their si20k-vox runs (the podcast rows are node-local per host) + one export
# watcher each (they may share the training nodes' spare CPUs). Run from the login node (launch only). Guard: LAUNCH_DONE.
set -u
P=/data/home/amartyavarma/ppm/si20k-tin-ppm; I=/data/home/amartyavarma/ppm/si20k-tin-intl; T=$P/tools
[ -f $P/LAUNCH_DONE ] && { echo "already launched: $(cat $P/LAUNCH_DONE)"; exit 1; }
HP=slinky-0,slinky-1,slinky-3,slinky-5,slinky-8,slinky-9,slinky-16,slinky-17
HI=slinky-2,slinky-6,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-19
jp=$(sbatch --parsable -w $HP $P/si20k_tin_ppm.sbatch) || exit 1
ji=$(sbatch --parsable -w $HI $I/si20k_tin_intl.sbatch) || exit 1
wp=$(sbatch --parsable --exclude=slinky-4,slinky-18 -o $P/logs/watch-%j.out $T/watch_tin.sbatch $P $jp si20ktinppm 80 600 /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12829)
wi=$(sbatch --parsable --exclude=slinky-4,slinky-18 -o $I/logs/watch-%j.out $T/watch_tin.sbatch $I $ji si20ktinintl 36 550 -)
echo $jp > $P/LAUNCHED_JOB; echo $ji > $I/LAUNCHED_JOB
echo "$(TZ=America/Los_Angeles date '+%F %T PT') PPM $jp (watch $wp) INTL $ji (watch $wi)" | tee $P/LAUNCH_DONE
