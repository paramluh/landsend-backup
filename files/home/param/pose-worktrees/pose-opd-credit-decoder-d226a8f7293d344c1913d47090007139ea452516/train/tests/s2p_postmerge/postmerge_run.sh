#!/bin/bash
# pose5 D: A's s2p suite on the MERGED code (G0COND merged + own groups), PPM_G0COND unset (off). Arg: which part.
P=/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge
case "$1" in
  cfg_c128) STEP=cfg bash $P/stage1.sbatch; S2P_TEST_CE_CHUNK=128 TAGSFX=_c128 STEP=tiny bash $P/stage1.sbatch ;;
  c1024) unset S2P_TEST_CE_CHUNK; TAGSFX=_c1024 STEP=tiny bash $P/stage1.sbatch ;;
  resume) S2P_TEST_CE_CHUNK=128 bash $P/stage2.sbatch; S2P_TEST_CE_CHUNK=128 S2P_ON=cb0 S2P_RESUME_TAG=_cb0 bash $P/stage2.sbatch ;;
esac
echo "== postmerge part $1 done $(TZ=America/Los_Angeles date '+%T PT')"
