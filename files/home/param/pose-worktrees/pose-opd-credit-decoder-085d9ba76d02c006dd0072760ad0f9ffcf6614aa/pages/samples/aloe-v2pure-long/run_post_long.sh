#!/bin/bash
# long post (10-18): on a compute node (CPU): metrics (analyze_long.py: whole generated window + first / last 30 s) -> NEW page
# klweb/ppm_aloe_v2pure_long (build_page_long.py) -> PAGE_LONG_READY marker (rv updater go) -> JS parse + data.json parse + jsdom page test.
W=/data/home/amartyavarma/ppm/samples/aloe-v2pure-long; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_v2pure_long; T=/data/home/amartyavarma/ppm/samples/aloe-v2pure-200/work-k30/test_page.js
NODE=${1:-slinky-4}
until [ $(ls $W/runs-gen-k140/*/DONE 2>/dev/null | wc -l) -ge 8 ]; do sleep 3; done
srun -p all -w $NODE -N1 -n1 -c 8 --cpu-bind=none --gres=none --mem=40G -t 30 -J v2plongpost \
  bash -c "source $FI/env_fast.sh; set -e; \$PY -u $W/work/analyze_long.py $W $W/ana; \$PY -u $W/work/build_page_long.py $W $K $W/ana/metrics.json; \
    echo ok > /data/home/amartyavarma/video/distilled-k30/PAGE_LONG_READY; \
    N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node; rm -rf $W/ana/page_js; mkdir -p $W/ana/page_js; \
    \$PY -c \"import re,sys; s=open('$K/index.html').read(); [open(f'$W/ana/page_js/s{i}.js','w').write(m) for i,m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S))]\"; \
    for f in $W/ana/page_js/*.js; do \$N --check \$f && echo \"JS OK \$f\"; done; \
    \$N -e \"JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')\"; \
    NODE_PATH=/data/home/amartyavarma/tools/jsdomenv/node_modules \$N $T $K"
echo "$(date -Is) post done rc $?"
