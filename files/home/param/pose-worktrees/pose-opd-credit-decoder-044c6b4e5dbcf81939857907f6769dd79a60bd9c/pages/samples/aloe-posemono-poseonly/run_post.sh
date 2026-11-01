#!/bin/bash
# post (10-19, compute node CPU): hand-motion metrics (analyze.py) -> page klweb/ppm_aloe_posemono_poseonly (build_page.py, after the studio
# renders) -> JS parse + data.json parse + jsdom page test. arg: metrics | page
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_poseonly; N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node
if [ "$1" = metrics ]; then
  srun -p all -w slinky-4 -N1 -n1 -c 2 --cpu-bind=none --gres=none --mem=24G -t 40 -J pmpo-post \
    bash -c "source $FI/env_fast.sh; \$PY -u $W/work/analyze.py $W $W/ana"
else
  srun -p all -w slinky-4 -N1 -n1 -c 2 --cpu-bind=none --gres=none --mem=24G -t 40 -J pmpo-post \
    bash -c "source $FI/env_fast.sh; set -e; \$PY -u $W/work/build_page.py $W $K $W/ana/metrics.json; \
      rm -rf $W/ana/page_js; mkdir -p $W/ana/page_js; \
      \$PY -c \"import re,sys; s=open('$K/index.html').read(); [open(f'$W/ana/page_js/s{i}.js','w').write(m) for i,m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S))]\"; \
      for f in $W/ana/page_js/*.js; do $N --check \$f && echo \"JS OK \$f\"; done; \
      $N -e \"JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')\"; \
      NODE_PATH=/data/home/amartyavarma/tools/jsdomenv/node_modules $N $W/work/test_page.js $K"
fi
echo "$(TZ=America/Los_Angeles date '+%F %T PT') post $1 rc $?"
