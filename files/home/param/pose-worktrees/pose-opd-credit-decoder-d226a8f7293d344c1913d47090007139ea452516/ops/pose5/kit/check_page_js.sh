#!/bin/bash
# pose5: node --check every inline <script> of the klweb template before/after the npl5-roll header patch (compute node; read-only on klweb)
NODE=/data/home/amartyavarma/tools/node-v20.18.0-linux-x64/bin/node; T=/data/home/amartyavarma/ppm/pose5/kit/state/jscheck; K=/data/home/amartyavarma/setup-logs/klweb
for v in bak new; do f=$K/ppm.tpl.html; [ $v = bak ] && f=$K/ppm.tpl.html.bak-pre-pose5roll-1027
  python3 -I - "$f" "$T/$v" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
for i, m in enumerate(re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", src, re.S)):
    open(f"{sys.argv[2]}-{i}.js", "w").write(m.group(1))
PY
  for f in $T/$v-*.js; do $NODE --check "$f" >/dev/null 2>&1 && echo "$v $(basename $f) OK" || echo "$v $(basename $f) FAIL"; done
done
