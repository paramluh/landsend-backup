#!/bin/bash
# k140 post (10-18): metrics for all 5 sources (analyze_v2pure.py) -> in-place page edit (add_k140.py) -> JS parse (node --check) + data.json
# parse + jsdom page test. Compute node (slinky-4 by default, CPU only). Launch-only on the login node.
W=/data/home/amartyavarma/ppm/samples/aloe-v2pure-200; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_v2pure_samples
NODE=${1:-slinky-4}
srun -p all -w $NODE -N1 -n1 -c 16 --cpu-bind=none --gres=none --mem=48G -t 30 -J v2ppost \
  bash -c "source $FI/env_fast.sh; set -e; \$PY -u $W/work/analyze_v2pure.py $W $W/ana; \$PY -u $W/work/add_k140.py $W $K $W/ana/metrics.json; \
    N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node; rm -rf $W/ana/page_js; mkdir -p $W/ana/page_js; \
    \$PY -c \"import re,sys; s=open('$K/index.html').read(); [open(f'$W/ana/page_js/s{i}.js','w').write(m) for i,m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S))]\"; \
    for f in $W/ana/page_js/*.js; do \$N --check \$f && echo \"JS OK \$f\"; done; \
    \$N -e \"JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')\"; \
    NODE_PATH=/data/home/amartyavarma/tools/jsdomenv/node_modules \$N $W/work/test_page.js $K"
