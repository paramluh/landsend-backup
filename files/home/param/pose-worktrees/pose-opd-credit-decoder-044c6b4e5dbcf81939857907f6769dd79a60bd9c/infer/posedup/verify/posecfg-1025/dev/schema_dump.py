import json, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live")
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/stage-posecfg")
import server as S
assert S.__file__.startswith("/data/home/amartyavarma/ppm/aloe-posedup-live/stage-posecfg"), S.__file__
S.POSE_CFG_LAT.update(p50=95.0, p95=99.0, src="test", lag_growth=None)
rows = S.ready_schema()
json.dump(rows, open(sys.argv[1], "w"))
print([ (r["name"], r.get("group"), r.get("type"), r.get("default")) for r in rows if "pose" in r["name"] or r["name"].startswith("cfg")])
print([r.get("hint") for r in rows if r["name"] == "pose_cfg_on"])
print(S.canon_in({"pose_cfg_on": True, "pose_cfg_s": 2.5, "pose_cfg_light": "false"}), S.canon_out({"pose_cfg_on": 1, "pose_cfg_s": 3.0}, {}))
