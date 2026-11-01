import pickle, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live")
import live_ppm_pose as LP
ci = LP.ckpt_info(sys.argv[1])
pickle.dump(ci, open(sys.argv[2], "wb"))
print({k: v for k, v in ci.items() if k not in ("shapes", "side")}, len(ci["shapes"]))
