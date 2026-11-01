# aloe-ft-32-samsep DATA status (PT)
- 10-23 22:20 **READY**
  - stores (/data): /data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores/m1n{0..3} — kept 11,777 rows (first chunks 7,949 train + 158 eval; later chunks 3,670); dropped 163 rows of 122 convs (first-chunk train 118, first-chunk eval 4, later chunks 41) -> dropped_122.txt
  - digests m1n0..3: 7318ffb3ea75cbcc3ff48b63b6cf8b88 31ecb2273542310096936c0baf49efa8 1759200f7818c6017dfb6a46e37fa5df 611b2e7edb1503cc600a6489e49dcf40
  - STAGED node-local on slinky-18 + slinky-4: /scratch/amartyavarma/aloe-ft-samsep/stores/m1n{0..3} (cmp-verified; /scratch free 234 / 163 GB). Reference /scratch/amartyavarma/aloe-ft/stores untouched.
  - channel-A FLAC: ALOE_FLAC_ROOT=/scratch/amartyavarma/aloe-ft-samsep/flac (both nodes) -> ALOE_SHARED_FLAC=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/flac (symlinks into the reference's GCS-verified copy; 16,214 gcs tracks, 0 mimidec; ledgers DONE_v2vox_node0..7 for new plan d1a3e3721328da0a)
  - ALOE_M1VOX_INDEX=data/plan/M1VOX_INDEX_samsep.json, ALOE_RAWGAIN=data/plan/RAWGAIN_samsep.json (plan_digest d1a3e3721328da0a), plan 15,898 items / 249 steps at 64 logical ranks
  - verification: VERIFY.md / VERIFY.json (ALL_OK); spot decodes in spot/
- 10-23 22:05 build done (20 s); verify + spot decode done 22:12.
