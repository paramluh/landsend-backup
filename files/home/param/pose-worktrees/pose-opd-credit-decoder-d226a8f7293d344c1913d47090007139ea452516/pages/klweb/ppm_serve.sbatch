#!/bin/bash
#SBATCH -J klweb-ppm -p all -w slinky-15 -N1 -n1 -c2 --mem=2G -t 7-00:00:00 -o /data/home/amartyavarma/setup-logs/klweb/serve-ppm-%j.log
# PPM live curves: 60 s updater (copies the live run's metrics.jsonl + meta) + static server + Cloudflare quick tunnel. CPU only.
W=/data/home/amartyavarma/setup-logs/klweb; P=/data/home/amartyavarma/ppm; B=/data/home/amartyavarma/bin
cd $W
(
while true; do
  R=$(ls -td $P/runs/ppm-v1-j* 2>/dev/null | head -1)
  if [ -n "$R" ]; then
    J=${R##*-j}
    cp $R/metrics.jsonl $W/ppm_metrics.jsonl.tmp 2>/dev/null && mv $W/ppm_metrics.jsonl.tmp $W/ppm_metrics.jsonl
    L=$P/logs/ppm-$J.log
    TOT=$(grep -oE 'step [0-9]+/[0-9]+' $L 2>/dev/null | tail -1 | cut -d/ -f2)
    ST=$(squeue -h -j $J -o %T 2>/dev/null); NL=$(squeue -h -j $J -o %N 2>/dev/null)
    printf '{"job":"job %s","total":%s,"state":"%s","nodes":"%s"}\n' "$J" "${TOT:-0}" "${ST:-ENDED}" "$NL" > $W/ppm_meta.json.tmp && mv $W/ppm_meta.json.tmp $W/ppm_meta.json
  fi
  sleep 60
done
) &
python3 $W/range_http.py 8787 $W >/dev/null 2>&1 &   # 10-20: Range-capable (video seeking); was python3 -m http.server
sleep 2
$B/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8787 2>&1 | tee $W/tunnel.log | while read -r l; do echo "$l"; u=$(echo "$l" | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com'); [ -n "$u" ] && echo "$u" > $W/TUNNEL_URL; done
