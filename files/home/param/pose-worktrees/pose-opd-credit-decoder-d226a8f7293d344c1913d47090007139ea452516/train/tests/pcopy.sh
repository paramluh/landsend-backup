#!/bin/bash
# parallel directory copy (flat dirs): pcopy.sh SRC DST [P]
set -e; S=$1; D=$2; P=${3:-32}; mkdir -p $D
ls -1 $S | xargs -P $P -I{} cp -r $S/{} $D/{}
echo "copied $(ls -1 $D | wc -l) entries $S -> $D ($(du -sh $D | cut -f1))"
