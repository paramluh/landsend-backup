#!/bin/bash
# read-only: per-host view of the ttconvo-ft final state on node-local /scratch
H=$(hostname -s); D=/scratch/amartyavarma/ttconvo-ft
echo "== $H $(df -BG /scratch | tail -1 | awk '{print "scratch_total="$2" used="$3" free="$4}')"
echo "$H awk=$(readlink -f $(which awk)) nproc=$(nproc) gpus_visible=${CUDA_VISIBLE_DEVICES:-none}"
ls -la $D/ckpt/ttconvo-ft/ 2>&1 | sed "s/^/$H ckpt: /"
ls -la $D/ckpt/ttconvo-ft/dcp-285/ 2>&1 | sed "s/^/$H dcp285: /"
du -sb $D/ckpt/ttconvo-ft/dcp-285 2>&1 | sed "s/^/$H du285: /"
cat $D/ckpt/ttconvo-ft/dcp-285/COMPLETE 2>&1 | sed "s/^/$H COMPLETE: /"; echo
ls -la $D/exports/ 2>&1 | sed "s/^/$H exports: /"
du -sh /scratch/amartyavarma/hfb/ttconvo-ft-* /scratch/amartyavarma/hfb/xet-cache 2>&1 | sed "s/^/$H hfb: /"
du -sh /scratch/amartyavarma/* 2>/dev/null | sort -h | tail -8 | sed "s/^/$H top: /"
