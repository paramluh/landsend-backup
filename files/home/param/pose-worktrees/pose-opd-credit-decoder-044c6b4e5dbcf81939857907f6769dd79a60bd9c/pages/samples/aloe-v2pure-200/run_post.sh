#!/bin/bash
# aloe-v2pure-200 post: hand-motion metrics (analyze_v2pure.py = analyze.py's maths) -> samples page (build_page.py) on a compute node, then
# the JS parse check (node) of the page's scripts. Launch-only on the login node.
W=/data/home/amartyavarma/ppm/samples/aloe-v2pure-200; FI=/data/home/amartyavarma/ppm/fastinfer
K=/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_v2pure_samples
NODE=${1:-slinky-15}
set -e
srun -p all -w $NODE -N1 -n1 -c 16 --cpu-bind=none --gres=none --mem=48G -t 30 -J v2ppost \
  bash -c "source $FI/env_fast.sh; set -e; \$PY -u $W/work/analyze_v2pure.py $W $W/ana; \$PY -u $W/work/build_page.py $W $K $W/ana/metrics.json; \
    N=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node; \$PY - $K/index.html $W/ana/page_js <<'EOF'
import re, sys, os
s = open(sys.argv[1]).read(); os.makedirs(sys.argv[2], exist_ok=True)
for i, m in enumerate(re.findall(r'<script>(.*?)</script>', s, re.S)):
    open(f'{sys.argv[2]}/s{i}.js', 'w').write(m)
print('scripts', i + 1)
EOF
    for f in $W/ana/page_js/*.js; do \$N --check \$f && echo \"JS OK \$f\"; done; \
    \$N -e \"JSON.parse(require('fs').readFileSync('$K/data.json','utf8')); console.log('data.json parses')\""
