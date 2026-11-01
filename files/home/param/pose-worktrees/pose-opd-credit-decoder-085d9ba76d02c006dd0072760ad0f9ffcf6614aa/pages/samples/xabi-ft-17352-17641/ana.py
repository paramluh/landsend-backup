import json, os, statistics as st
E = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"; R = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
for t in ("xs16610_fs", "xs17352_fs", "xs17641_fs"):
    m = json.load(open(f"{E}/{t}.json")); W = []; U = []; A = []; V = []
    print(f"===== {t}")
    for row, (rid, k) in m.items():
        sd = f"{R}/{rid}/s{k}"
        if not os.path.exists(f"{sd}/meta.json"):
            print(row, "no meta"); continue
        me = json.load(open(f"{sd}/meta.json"))
        W.append(me["gen_words"]); U.append(me["pose_gen_cb0_unique"]); A.append(me.get("audio", {}))
        vs = json.load(open(f"{sd}/video_status.json")) if os.path.exists(f"{sd}/video_status.json") else {}
        V.append(vs.get("refs"))
        print(f"{row} words {me['gen_words']:3d} pose_cb0_uniq {me['pose_gen_cb0_unique']:3d} audio {json.dumps(me.get('audio'))[:120]} refs {vs.get('refs')} vfail {os.path.exists(sd + '/VIDEO_FAIL')}")
        print("   ", me["gen_text"][:400])
    if W:
        print(f"MEAN words {st.mean(W):.1f}  pose_cb0_uniq {st.mean(U):.1f}")
