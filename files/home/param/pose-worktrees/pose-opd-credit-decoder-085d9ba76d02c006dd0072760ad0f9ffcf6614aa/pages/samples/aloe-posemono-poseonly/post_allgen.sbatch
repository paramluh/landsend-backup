#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=none --mem=24G -t 60 -J pmpo-postag
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/postag-%j.out
# all-generated eval page (CPU): waits for the 4 allgen renders, metrics (analyze_allgen.py), moves the pose-only page out of klweb, builds
# klweb/ppm_aloe_posemono_poseonly (build_page_allgen.py), JS + data.json + jsdom checks, frame sheet + audio check (check_allgen.py).
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly; N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node
R=/data/home/amartyavarma/video/studio-posemono/allgen
source $FI/env_fast.sh
until [ $(ls $W/runs-allgen/*/DONE 2>/dev/null | wc -l) -ge 4 ]; do sleep 5; done
$PY -u $W/work/analyze_allgen.py $W $W/eval5/ana-allgen || exit 1
until [ $(ls $R/*/DONE 2>/dev/null | wc -l) -ge 4 ]; do sleep 5; done
set -e
[ -d $K ] && mv $K $W/old/page-poseonly-studio-$(date +%s)
$PY -u $W/work/build_page_allgen.py $W $K $W/eval5/ana-allgen/metrics.json
rm -rf $W/eval5/page_js; mkdir -p $W/eval5/page_js
$PY -c "import re; s=open('$K/index.html').read(); [open(f'$W/eval5/page_js/s{i}.js','w').write(m) for i,m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S))]"
for f in $W/eval5/page_js/*.js; do $N --check $f && echo "JS OK $f"; done
$N -e "JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')"
NODE_PATH=/data/home/amartyavarma/tools/jsdomenv/node_modules $N $W/work/test_page_studiopf.js $K
mkdir -p $W/eval5/frames-allgen
$PY -u $W/work/check_allgen.py $K $W/eval5/frames-allgen
echo "$(TZ=America/Los_Angeles date '+%F %T PT') POSTAG DONE"
