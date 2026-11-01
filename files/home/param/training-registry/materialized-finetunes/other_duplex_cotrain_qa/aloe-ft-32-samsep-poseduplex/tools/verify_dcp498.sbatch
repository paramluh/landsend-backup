#!/bin/bash
#SBATCH -p all -N1 -n1 -c 8 --mem=64G --gres=none -t 01:00:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498/verify-%x-%j.out
# READ-ONLY verify of job 30074's end save dcp-498 on ONE host (sbatch -w <host> -J vdcp498-<host> verify_dcp498.sbatch <host index>).
# CPU only, nice 10, 4 digest workers. Also prints /scratch free + sizes of dcp-498 and export-498 (task-3 planning).
IDX=$1; H=$(hostname -s)
D=/scratch/amartyavarma/aloe-ft-32-samsep/saves-poseduplex/ppm-aloe-poseduplex-v5-j30074/dcp-498
X=/scratch/amartyavarma/aloe-ft-32-samsep/exports-poseduplex-v5/export-498
O=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
case ",$H," in ,slinky-9,|,slinky-10,) echo "ABORT: never on $H"; exit 1;; esac
echo "== $H idx $IDX job $SLURM_JOB_ID $(TZ=America/Los_Angeles date '+%F %T PT'); /scratch: $(df -h --output=size,used,avail /scratch | tail -1)"
echo "dcp-498: $(du -sb $D | cut -f1) B, $(ls -A $D | wc -l) files; export-498: $(du -sb $X 2>/dev/null | cut -f1) B, $(ls -A $X 2>/dev/null | wc -l) files"
ls -la $D
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
nice -n 10 $PY -I /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools/verify_dcp_host.py $D 498 $IDX 8 $O/verify-dcp498-$H.json 4
rc=$?
echo "== $H verify rc $rc $(TZ=America/Los_Angeles date '+%T PT')"
exit $rc
