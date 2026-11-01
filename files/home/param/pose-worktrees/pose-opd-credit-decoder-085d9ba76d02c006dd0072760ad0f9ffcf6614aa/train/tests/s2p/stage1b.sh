#!/bin/bash
# stage 1 minus the (done) cfg step: micros then tiny variants (wrapper so one job runs both in order)
STEP=micros bash /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p/stage1.sbatch || exit 1
STEP=tiny bash /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p/stage1.sbatch
