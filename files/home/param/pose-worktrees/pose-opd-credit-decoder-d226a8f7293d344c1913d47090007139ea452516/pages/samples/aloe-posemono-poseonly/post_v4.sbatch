#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=none --mem=24G -t 40 -J pmpo-postv4
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/postv4-%j.out
# v4 eval page (CPU), after the finals job: moves the previous page out of klweb, builds klweb/ppm_aloe_posemono_poseonly (build_page_v4.py),
# JS + data.json + jsdom checks, proofs (prove_v3.py).
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly; N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node
R=/data/home/amartyavarma/video/studio-posemono/v3
source $FI/env_fast.sh
set -e
[ $(ls $R/*/splice.json | wc -l) -ge 4 ] && [ $(ls $R/*/real_recording.mp4 | wc -l) -ge 4 ]
[ -d $K ] && mv $K $W/old/page-allgen-v2-$(date +%s)
$PY -u $W/work/build_page_v4.py $W $K $W/eval5/ana-allgen/metrics.json
rm -rf $W/eval5/page_js; mkdir -p $W/eval5/page_js
$PY -c "import re; s=open('$K/index.html').read(); [open(f'$W/eval5/page_js/s{i}.js','w').write(m) for i,m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S))]"
for f in $W/eval5/page_js/*.js; do $N --check $f && echo "JS OK $f"; done
$N -e "JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')"
NODE_PATH=/data/home/amartyavarma/tools/jsdomenv/node_modules $N $W/work/test_page_v4.js $K
mkdir -p $W/eval5/proof-v4
$PY -u $W/work/prove_v3.py $K $W/eval5/proof-v4
echo "$(TZ=America/Los_Angeles date '+%F %T PT') POSTV4 DONE"
