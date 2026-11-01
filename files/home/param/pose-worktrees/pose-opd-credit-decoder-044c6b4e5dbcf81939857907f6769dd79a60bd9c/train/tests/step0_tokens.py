"""shared by hf_ref.py / step0_full.py: the plain-text token sequence of step-0 check 1 = the channel-B data speaker's word pieces of the
text-duplex EVAL pack 0 (data_ppm side rule), concatenated in conversation order, first N tokens. CPU only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_ppm as D  # noqa: E402

TEXT_ROOT = os.environ.get("PPM_TEXT_ROOT", "/data/home/amartyavarma/data/gemma-txtdup-v1")


def pack0(cfg=None):
    cfg = cfg or D.default_cfg()
    st = D.TXD.Store(TEXT_ROOT)
    recs = [st.get(j) for j in st.pack("eval", 0)]
    return recs, cfg


def plain_tokens(n=1024):
    recs, cfg = pack0()
    out = []
    for r in recs:
        sw = D.side_of(f"txtdup/{r['conv_id']}", cfg["seed"])
        side = "a" if sw else "b"                                    # swap 1: data side a ('user') is channel B
        out += [int(x) for x in r[f"pids_{side}"]]
    return np.asarray(out[:n], np.int64)
