#!/bin/bash
# (b-ii) step 3: compare ORIG flag off vs COPY full / cb0 (zero s_emb)
source /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge/stage3_common.sh
$PY $TD/compare.py var $OUT/real_orig_off.json $OUT/real_copy_full.json s2p_zero $OUT/CMP_real_orig_vs_full0.json
$PY $TD/compare.py var $OUT/real_orig_off.json $OUT/real_copy_cb0.json s2p_zero $OUT/CMP_real_orig_vs_cb00.json
