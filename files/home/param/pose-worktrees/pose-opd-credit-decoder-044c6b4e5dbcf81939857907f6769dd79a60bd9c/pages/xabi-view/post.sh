#!/bin/bash
# page build + a contact sheet of frame 2 s of every mp4 (visual check) + ffprobe durations
O=/data/home/amartyavarma/ppm/xabi-view/out
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
$PY -u /data/home/amartyavarma/ppm/xabi-view/code/build_page.py $O
FF=$($PY -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p $O/check
for f in $O/media/*.mp4; do b=$(basename $f .mp4); $FF -y -hide_banner -loglevel error -ss 2 -i $f -frames:v 1 $O/check/$b.png; $FF -hide_banner -i $f 2>&1 | grep -e Duration -e Stream | tr -s ' ' | sed "s|^|$b |"; done
$PY - <<'P'
import glob, cv2, numpy as np
ims=[cv2.imread(f) for f in sorted(glob.glob("/data/home/amartyavarma/ppm/xabi-view/out/check/c*.png"))]
h=max(i.shape[0] for i in ims); ims=[cv2.copyMakeBorder(i,0,h-i.shape[0],0,4,cv2.BORDER_CONSTANT,value=(255,255,255)) for i in ims]
cv2.imwrite("/data/home/amartyavarma/ppm/xabi-view/out/check/sheet.png", np.concatenate(ims,1)); print("sheet ok")
P
