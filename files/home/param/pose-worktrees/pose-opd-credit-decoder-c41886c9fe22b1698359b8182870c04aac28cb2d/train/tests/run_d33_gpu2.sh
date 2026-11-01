#!/bin/bash
# D33 GPU tests part 2 (one node, inside srun): align, then the checkpoint chain.
O=/data/home/amartyavarma/ppm/teststore/d33; C=/data/home/amartyavarma/ppm/code
PPM_DECP_BB=1 bash $C/tests/test_d33.sh $C align $O/align.json > $O/align.log 2>&1; tail -1 $O/align.log
bash $C/tests/test_d33_ckpt.sh $O
rm -rf /scratch/amartyavarma/ppm/d33test
echo CHAIN2_DONE
