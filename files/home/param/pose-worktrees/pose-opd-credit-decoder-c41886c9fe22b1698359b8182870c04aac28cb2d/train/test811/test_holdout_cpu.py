"""CPU check of the 8,1,1 speech epoch-set holdout expression (train_ppm.py, S811 streams) against data_ppm.d2_stream's own holdout mask,
on the stage-3 eval store's index (its rows are the s3eval-v1 holdout episodes)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_ppm as D  # noqa: E402

root = "/data/home/amartyavarma/qwen-1ch-data/s3-evalstore/rows"
sidx = D.DS.load_index(root)
n = len(sidx["row"])
salts, frac = ["s3eval-v1"], 0.0005
h_ = np.arange(n, dtype=np.int64)[::-1]                      # a 'head' of native codes (slot 0 code == node row)
mine = np.array([any(D.DS.is_eval_episode(D.DS.episode_key(sidx["dataset"][r], sidx["podcast_id"][r], sidx["episode_id"][r]), frac, sl) for sl in salts)
                 for r in h_], bool)
hn = np.array([any(D.DS.is_eval_episode(k, frac, sl) for sl in salts) for k in [D.DS.episode_key(a, b, c) for a, b, c in
               zip(sidx["dataset"], sidx["podcast_id"], sidx["episode_id"])]], bool)
ok = np.array_equal(mine, hn[h_])
print(("PASS" if ok else "FAIL"), f"8,1,1 holdout mask == d2_stream's hn[head] on {n} rows ({int(mine.sum())} holdout)")
sys.exit(0 if ok else 1)
