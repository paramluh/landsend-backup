#!/bin/bash
# one 2-CPU slot: (c) resume phases A..L (stage2.sbatch body), then the production-CE-chunk (1024) tiny rerun of (a2)/(b-i)
S2P_TEST_CE_CHUNK=128 bash /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge/stage2.sbatch
unset S2P_TEST_CE_CHUNK
TAGSFX=_c1024 STEP=tiny bash /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge/stage1.sbatch
