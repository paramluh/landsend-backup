#!/bin/bash
# main (86) 10-24 ~08:35 PT: back up the node-local END full saves dcp-249 of jobs 29345 (vox17127) and 29444 (persona) to PRIVATE HF, from
# inside MoT job 29514 (srun --overlap, -c 4, --mem 48G, --cpu-bind=none, nice 19 + idle IO, streaming upload). No deletion. Runs from a tiny CPU job.
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN)_[A-Za-z0-9_]+'); do unset $v; done
U=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/tools/upload_save.sh; S=/scratch/amartyavarma/aloe-ft-32-samsep
for h in slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-8 slinky-11 slinky-12; do
  srun --overlap --jobid=29514 -w $h -N1 -n1 -c 4 --mem=48G --cpu-bind=none bash $U $S/saves/ppm-aloe-samsep-vox17127-j29345/dcp-249 InternalCan/ppm-aloe-samsep-vox17127-saves dcp-249__$h > /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/save-up-vox-$h.log 2>&1 &
done
for h in slinky-6 slinky-7 slinky-13 slinky-14 slinky-15 slinky-16 slinky-17 slinky-19; do
  srun --overlap --jobid=29514 -w $h -N1 -n1 -c 4 --mem=48G --cpu-bind=none bash $U $S/saves-persona/ppm-aloe-samsep-vox17127-persona-j29444/dcp-249 InternalCan/ppm-aloe-samsep-vox17127-persona-saves dcp-249__$h > /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/save-up-persona-$h.log 2>&1 &
done
wait
grep -h "VERIFY PASS\|FAILED" /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/save-up-*.log
