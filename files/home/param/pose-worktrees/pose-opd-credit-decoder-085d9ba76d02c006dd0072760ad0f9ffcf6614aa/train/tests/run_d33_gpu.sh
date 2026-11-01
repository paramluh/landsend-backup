#!/bin/bash
# D33 GPU test chain (one node, inside srun): grads x4 (old code / new off / new on / new on alpha 0), align, learn off/on (prod decoder path).
O=/data/home/amartyavarma/ppm/teststore/d33; C=/data/home/amartyavarma/ppm/code; B=${D33_BACKUP:?backup code dir}
S="bash $C/tests/test_d33.sh"
cd $O
$S $B grads $O/g_old.json > $O/g_old.log 2>&1; tail -1 $O/g_old.log
PPM_DECP_BB=0 $S $C grads $O/g_off.json > $O/g_off.log 2>&1; tail -1 $O/g_off.log
PPM_DECP_BB=1 $S $C grads $O/g_on.json > $O/g_on.log 2>&1; tail -1 $O/g_on.log
PPM_DECP_BB=1 PPM_DECP_BB_ALPHA=0 $S $C grads $O/g_a0.json > $O/g_a0.log 2>&1; tail -1 $O/g_a0.log
PPM_DECP_BB=1 $S $C align $O/align.json > $O/align.log 2>&1; tail -1 $O/align.log
for k in off on; do
  PPM_DECP_BB=$([ $k = on ] && echo 1 || echo 0) D33_FAST=1 D33_NSTEP=16 PPM_DEC_LAYERS_T=28 $S $C learn $O/learn_$k.json > $O/learn_$k.log 2>&1; tail -1 $O/learn_$k.log
done
rm -rf /scratch/amartyavarma/ppm/d33test/runs /scratch/amartyavarma/ppm/d33test/ckpt
echo CHAIN_DONE
