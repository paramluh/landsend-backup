"""PERSONA CPU dry resolution (no GPU, no launch): the persona trainer's resolved CFG built from run_persona.env exactly as rank 0 of the
world-64 job builds it; its startup diff vs 26232 (abort rule) and the report-only diffs vs 26467 / 28511 / 29345.
usage (compute node, env sourced from persona/run_persona.env): python -I -B cfg_dry.py <code dir> <out dir>"""
import os
import sys

sys.path.insert(0, sys.argv[1])
OUT = sys.argv[2]
os.environ.update(RANK="0", WORLD_SIZE="64", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", COTRAIN_CFG_DRY="1")
os.environ["ALOE_FLAC_ROOT"] = os.environ["ALOE_SHARED_FLAC"]
import train_samsep_persona as T  # noqa: E402

T.LOGDIR = OUT
print("PROD", T.PROD, "ALOE_ONLY", T.ALOE_ONLY, "aloe ranks", T.NA, "slots", T.GMAX, "offload", T.OFFLOAD, "INITVOX", T.INITVOX)
n, bad = T.cfg_diff_26232()
for tag in ("26467", "28511", "29345"):
    rows = T.cfg_diff_report(os.environ[f"ALOE_REF_CFG_{tag}"], tag)
    print(f"vs {tag}: {len(rows)} keys differ")
print("vs 26232:", n, "bad", bad)
print("persona cfg", T.CFG["persona"])
