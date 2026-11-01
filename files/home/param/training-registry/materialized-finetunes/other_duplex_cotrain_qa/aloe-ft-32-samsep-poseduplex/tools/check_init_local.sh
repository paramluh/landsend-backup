#!/bin/bash
# on THIS pose host: the host-to-host init copy is complete (COMPLETE + meta.json with vgain_theta_fp32 + rep.pt) and every file listed in
# SHA256SUMS (from the source, written by PPMVOX) matches. exit 0 ok / 2 not there yet / 1 mismatch.
D=/scratch/amartyavarma/poseduplex-init/${1:-w-18100}; H=$(hostname -s)
[ -f $D/COMPLETE ] && [ -f $D/meta.json ] && [ -f $D/rep.pt ] || { echo "$H: not yet"; exit 2; }
grep -q vgain_theta_fp32 $D/meta.json || { echo "$H: meta.json has no vgain_theta_fp32"; exit 1; }
if [ -f $D/SHA256SUMS ]; then
  cd $D && bad=$(awk '{print $2}' SHA256SUMS | xargs -P 16 -I{} sh -c 'echo "$(sha256sum {} | cut -c1-64) {}"' | sort -k2 | diff - <(awk '{print $1" "$2}' SHA256SUMS | sort -k2) | grep -c '^[<>]')
  [ "$bad" = 0 ] || { echo "$H: SHA MISMATCH ($bad lines)"; exit 1; }
  echo "$H: OK $(ls $D | wc -l) files, $(du -sh $D | cut -f1), sha256 == SHA256SUMS ($(wc -l < SHA256SUMS) entries)"
else
  echo "$H: OK $(ls $D | wc -l) files, $(du -sh $D | cut -f1) (NO SHA256SUMS: unverified)"
fi
