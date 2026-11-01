#!/bin/bash
#SBATCH -J aloevox-stage -p all --ntasks-per-node=1 -c 8 --mem=16G -t 00:30:00
#SBATCH -o aloevox-stage-%j.log
# VOX run: stage the PPMVOX v3 stop export w-17127 from private HF InternalCan/ppmvox-v3-16861 (sha256 == HF LFS) onto each of the 8 hosts:
# only the rank files that host's 8 ranks read at world 64 (w64/init_subset.json, host index = position in PPM_HOSTS) + rep/meta/COMPLETE.
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 bash -c '
H=$(hostname -s); HOSTS=slinky-1,slinky-2,slinky-5,slinky-11,slinky-3,slinky-0,slinky-8,slinky-12
h=$(( $(echo $HOSTS | tr , "\n" | grep -nx $H | cut -d: -f1) - 1 ))
mkdir -p ${ALOE_SCRATCH}/aloevox-init; cd /tmp
EVALVOX_HF_REPO=InternalCan/ppmvox-v3-16861 nice -n 10 ${ALOE_ROOT}/venvs/duplex/bin/python -I ${ALOE_ROOT}/ppm/aloe-ft-32-samsep/evalvox/stage_export.py w-17127 ${ALOE_ROOT}/ppm/aloe-ft-32-samsep/w64/init_subset.json $h ${ALOE_SCRATCH}/aloevox-init/w-17127 2>&1 | grep -v Warn | sed "s/^/$H (host $h): /"
echo "$H free after: $(df -BG --output=avail /scratch | tail -1 | tr -d " ")"; [ -f ${ALOE_SCRATCH}/aloevox-init/w-17127/COMPLETE ]'
