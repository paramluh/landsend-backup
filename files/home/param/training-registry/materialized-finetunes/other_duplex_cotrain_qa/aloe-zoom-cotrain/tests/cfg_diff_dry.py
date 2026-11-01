"""resolved-config diff of the PRODUCTION cotrain launch vs job 26232, without launching: the trainer module's CFG built from the production
env (aloe_zoom_cotrain.env [+ an override env for the zoom_epoch block]) as rank 0 of the 16-node job would build it, then its own
cfg_diff_26232() (same allowed-key rule the run asserts at start). usage (compute node): python -I cfg_diff_dry.py <code dir> <out txt>"""
import os, sys
sys.path.insert(0, sys.argv[1])
os.environ.update(RANK="0", WORLD_SIZE="128", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", COTRAIN_CFG_DRY="1",
                  PPM_HOSTS=",".join(f"slinky-{i}" for i in (0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19)))
os.environ["QQA_RUN"] = os.environ["QQA_RUN"] + "-jDRY"
import train_cotrain as T
T.LOGDIR = os.path.dirname(sys.argv[2])
print("PROD", T.PROD, "groups aloe", T.NA, "zoom", T.NZ, "slots", T.GMAX, "rank 0 group", T.GROUP)
n, bad = T.cfg_diff_26232()
os.replace(f"{T.LOGDIR}/cfg_diff_26232.txt", sys.argv[2])
print("written", sys.argv[2], "bad", bad)
