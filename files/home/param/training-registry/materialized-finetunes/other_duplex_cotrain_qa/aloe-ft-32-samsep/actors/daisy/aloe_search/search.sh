#!/bin/bash
# Case-insensitive "dais" search across every Aloe source on /data (text/json files) -> hits_*.txt
O=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy/aloe_search
find /data/home/amartyavarma/data/aloe-handover-full/DATA -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" \) -print0 | xargs -0 -P 32 -n 400 grep -l -i "dais" > $O/hits_handover_full.txt 2>/dev/null
find /data/home/amartyavarma/data/aloe-handover /data/home/amartyavarma/data/aloe-redl /data/home/amartyavarma/data/aloe-audit /data/home/amartyavarma/data/aloe-sam-sep /data/home/amartyavarma/qwen-1ch-data/aloe-ft-prep /data/home/amartyavarma/qwen-1ch-data/aloe-handover-prep /data/home/amartyavarma/qwen-1ch-data/aloe-tt3-data /data/home/amartyavarma/qwen-1ch-data/s2handover-xbleed-meta /data/home/amartyavarma/ppm/aloe-m1vox-live -type f \( -name "*.json" -o -name "*.jsonl" -o -name "*.csv" -o -name "*.md" -o -name "*.txt" -o -name "*.html" \) -size -2G -print0 | xargs -0 -P 32 -n 200 grep -l -i "dais" > $O/hits_other.txt 2>/dev/null
echo done $(wc -l < $O/hits_handover_full.txt) $(wc -l < $O/hits_other.txt)
