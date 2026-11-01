#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=none --mem=24G -t 60 -J pmpo-post4
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/post4-%j.out
# eval-set page (CPU): waits for the 8 renders, then metrics (analyze_eval5.py) -> klweb/ppm_aloe_posemono_poseonly (build_page_eval4.py)
# -> JS parse + data.json parse + jsdom page test.
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly; N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node
D=/data/home/amartyavarma/video/studio-posemono/eval4/done
source $FI/env_fast.sh
$PY -u $W/work/analyze_eval5.py $W $W/eval5/ana || exit 1
until [ $(ls $D 2>/dev/null | grep -vc tmp) -ge 8 ]; do sleep 5; done
set -e
$PY -u $W/work/build_page_eval4.py $W $K $W/eval5/ana/metrics.json
rm -rf $W/eval5/page_js; mkdir -p $W/eval5/page_js
$PY -c "import re; s=open('$K/index.html').read(); [open(f'$W/eval5/page_js/s{i}.js','w').write(m) for i,m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S))]"
for f in $W/eval5/page_js/*.js; do $N --check $f && echo "JS OK $f"; done
$N -e "JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')"
NODE_PATH=/data/home/amartyavarma/tools/jsdomenv/node_modules $N $W/work/test_page_eval4.js $K
echo "$(TZ=America/Los_Angeles date '+%F %T PT') POST4 DONE"
