#!/bin/bash
H=$(hostname -s)
echo "$H self: $(grep Cpus_allowed_list /proc/self/status)"
for p in $(pgrep -u amartyavarma -f export_from_dcp.py); do echo "$H conv pid $p $(grep Cpus_allowed_list /proc/$p/status) state $(awk '{print $3}' /proc/$p/stat)"; done
for p in $(pgrep -u amartyavarma -f train_ilp.py | head -16); do echo "$H train pid $p $(grep Cpus_allowed_list /proc/$p/status) threads $(ls /proc/$p/task | wc -l)"; done
# per-core busy % over 3 s
a=$(grep '^cpu[0-9]' /proc/stat); sleep 3; b=$(grep '^cpu[0-9]' /proc/stat)
paste <(echo "$a") <(echo "$b") | awk '{n=NF/2; t1=0;t2=0; for(i=2;i<=n;i++){t1+=$i; t2+=$(i+n)}; i1=$5+$6; i2=$(5+n)+$(6+n); busy=100*(1-(i2-i1)/(t2-t1)); printf "%s:%.0f ", $1, busy} END {print ""}' | sed "s/^/$H busy: /"
