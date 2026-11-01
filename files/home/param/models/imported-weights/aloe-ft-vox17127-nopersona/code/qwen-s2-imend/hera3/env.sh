# Sourced by every hera3 launcher / test runner. Triton (fused kernels) compiles a tiny C launcher per kernel signature that includes
# Python.h: it needs the Python 3.10 development headers (apt install python3.10-dev). RELEASE: a machine without them (e.g. the
# training cluster's compute nodes) sets ALOE_PY_INCLUDE in aloe.env to an unpacked libpython3.10-dev include dir list
# (<dir>/usr/include/python3.10:<dir>/usr/include). The compiled launchers are cached in TRITON_CACHE_DIR.
if [ -n "${ALOE_PY_INCLUDE:-}" ]; then export C_INCLUDE_PATH=${ALOE_PY_INCLUDE}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}; fi
export RAW_AUDIO_CACHE=${RAW_AUDIO_CACHE:-${ALOE_SCRATCH}/hera2/raw}
