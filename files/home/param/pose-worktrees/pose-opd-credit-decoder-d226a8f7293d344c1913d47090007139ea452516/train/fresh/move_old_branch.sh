#!/bin/bash
# 10-21 23:2x PT: the abandoned fresh-from-17012 branch (job 27030): delete dcp-17118 (superseded by dcp-17120 of the same branch), move dcp-17120
# (its final STOP_AFTER_SAVE state; not exported) out of the ckpt dir so PPM_RESUME_FROM=auto / the watchdog can never resume it.
R=/scratch/amartyavarma/ppm; H=$(hostname -s)
mkdir -p $R/ckpt-old811f/ppm-v1
[ -f $R/ckpt/ppm-v1/dcp-17120/COMPLETE ] && mv $R/ckpt/ppm-v1/dcp-17120 $R/ckpt-old811f/ppm-v1/dcp-17120
[ -d $R/ckpt/ppm-v1/dcp-17118 ] && { b=$(du -sb $R/ckpt/ppm-v1/dcp-17118 | cut -f1); rm -rf $R/ckpt/ppm-v1/dcp-17118 && echo "$(date '+%F %T') $H deleted $R/ckpt/ppm-v1/dcp-17118 ($b B; superseded by dcp-17120 of the abandoned fresh-from-17012 branch, job 27030)" >> /data/home/amartyavarma/DELETED-1021-hfb-hold.log; }
echo "$H ckpt=$(ls $R/ckpt/ppm-v1 | tr '\n' ' ') old=$(ls $R/ckpt-old811f/ppm-v1 | tr '\n' ' ') free=$(df -BG --output=avail /scratch | tail -1)"
