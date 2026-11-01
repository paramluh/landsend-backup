import json, sys
d = json.load(open(sys.argv[1]))
A = d["agg"]
ks = ["wrist_speed","wrist_p50","wrist_p90","wrist_p99","path_45s","wrist_pos_std","still_frac","gestures_45s","hand_height_range","band_05_3","band_05_3_frac","hf_frac","elbow_speed","finger_speed","finger_artic_speed","hand_visible_frac","wrist_conf","speak_frac","speed_speaking","speed_silent","speak_silent_ratio","r_speed_energy","r_max","r_max_lag_s","onset_speed_ratio","code_cb0_distinct","code_cb0_entropy_bits","code_cb0_repeat","code_cb0_mean_run","code_cb_change_frac","code_tuple_distinct"]
srcs = [s for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(A)
print("metric".ljust(22) + "".join(s[:10].rjust(11) for s in ["GT"] + srcs))
for k in ks:
    row = k.ljust(22) + ("%11.4f" % A[srcs[0]]["gen"]["gt_mean"].get(k, float("nan")))
    for s in srcs:
        v = A[s]["gen"]["mean"].get(k)
        row += ("%11.4f" % v) if v is not None else "        nan"
    print(row)
