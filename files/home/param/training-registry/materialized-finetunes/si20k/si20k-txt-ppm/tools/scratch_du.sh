#!/bin/bash
H=$(hostname -s); echo "== $H free $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)G total $(df -BG --output=size /scratch | tail -1 | tr -dc 0-9)G"
du -s --block-size=1G /scratch/* 2>/dev/null | sort -rn | head -6 | sed "s/^/$H top /"
du -s --block-size=1G /scratch/amartyavarma/* 2>/dev/null | sort -rn | head -14 | sed "s/^/$H /"
