#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=none --mem=24G -t 60 -J pmpo-poststpf
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/poststpf-%j.out
# eval-set page with the card's studio people (CPU): waits for the 4 studioprefill rows, moves the app-footage page out of klweb, builds
# klweb/ppm_aloe_posemono_poseonly (build_page_studiopf.py), JS + data.json + jsdom checks, and a frame sheet (0.5 s, 4 s) per video.
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly; N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node
R=/data/home/amartyavarma/video/studio-posemono/studioprefill; F=/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg
source $FI/env_fast.sh
until [ $(ls $R/*/DONE 2>/dev/null | wc -l) -ge 4 ]; do sleep 5; done
set -e
[ -d $K ] && mv $K $W/old/page-realprefill-appfootage-$(date +%s)
$PY -u $W/work/build_page_studiopf.py $W $K $W/eval5/ana/metrics.json
rm -rf $W/eval5/page_js; mkdir -p $W/eval5/page_js
$PY -c "import re; s=open('$K/index.html').read(); [open(f'$W/eval5/page_js/s{i}.js','w').write(m) for i,m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S))]"
for f in $W/eval5/page_js/*.js; do $N --check $f && echo "JS OK $f"; done
$N -e "JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')"
NODE_PATH=/data/home/amartyavarma/tools/jsdomenv/node_modules $N $W/work/test_page_studiopf.js $K
mkdir -p $W/eval5/frames
for r in R-004:05 R-012:13 R-014:15 R-019:20; do row=${r%%:*}; c=${r#*:}
  for k in gen real; do for t in 0.5 4; do $F -v error -y -ss $t -i $K/rv/$row-$k.mp4 -frames:v 1 $W/eval5/frames/$row-$k-$t.png; done; done
  $F -v error -y -i /data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_candidates/r/c$c.jpg $W/eval5/frames/$row-card.png
done
$PY -c "
from PIL import Image
rows=['R-004','R-012','R-014','R-019']; cols=['card','gen-0.5','gen-4','real-0.5','real-4']
sheet=Image.new('RGB',(256*5,480*4))
for i,r in enumerate(rows):
    for j,c in enumerate(cols):
        sheet.paste(Image.open(f'$W/eval5/frames/{r}-{c}.png').convert('RGB').resize((256,480)),(256*j,480*i))
sheet.resize((640,960)).save('/tmp/claude-243004069/-data-home-amartyavarma/23826bf0-1ae4-487f-94fa-3b391ad97da8/scratchpad/eval4_frames_sheet.png')"
echo "$(TZ=America/Los_Angeles date '+%F %T PT') POSTSTPF DONE"
