#!/bin/bash
# page build test on the fake runs (B-000 with POSE_DONE, B-001 = same run without pose videos); writes into page/ (removed after by the caller)
S=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples; T=$S/test-pose; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
NODE=/data/home/amartyavarma/.vscode-server/cli/servers/Stable-2d0b7506a36c1d6db9a466440131a8e01eab2289/server/node
FF=/data/home/amartyavarma/venvs/qwen-s2/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
rm -rf $T/B-001; mkdir -p $T/B-001; cp $T/B-000/{*.npy,*.wav,meta.json,words.json,DONE} $T/B-001/
time $PY -I $S/page/build_page.py $T 2 || exit 1
$PY -I - <<'PY'
import json, os
P = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/page"
d = json.load(open(f"{P}/data.json")); f = json.load(open(f"{P}/files.json"))
for c in d["convs"]:
    for tag, k in c["ck"].items():
        for v in k["versions"]:
            print(c["name"], tag, v["label"], "dur", round(v["dur"], 3), "video", v.get("video"), "sampling.pose_topk", k["sampling"].get("pose_topk"))
miss = [x for x in f if not os.path.isfile(f"{P}/{x}")]
print("files.json", len(f), "entries; mp4", sorted(x for x in f if x.endswith(".mp4")), "missing", miss)
PY
$PY -I -c "
import re;s=open('$S/page/index.html').read();m=re.findall(r'<script>(.*?)</script>',s,re.S);open('$T/page_script.js','w').write(m[0]);print('script chars',len(m[0]),'scripts',len(m))"
$NODE --check $T/page_script.js && echo "node --check OK"
$FF -hide_banner -loglevel error -y -ss 5 -i /data/home/amartyavarma/ppm/samples/aloe-pose-ft-430/runs-pose/B-000/gt_pose_s1.mp4 -frames:v 1 $T/precedent-gt_s1-5s.png && echo precedent still ok
