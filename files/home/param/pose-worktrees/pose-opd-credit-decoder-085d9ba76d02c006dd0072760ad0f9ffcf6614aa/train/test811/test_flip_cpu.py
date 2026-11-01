"""CPU unit test of the fresh-resume flip plumbing (no data): FLIP_BIT round trip through the speech codes / text pack ids, the trainer's
epoch >= 2 marking on s811.speech_stream, chat_ppm.eob_to_user on a synthetic two-lane conversation."""
import sys

import numpy as np

C = "/data/home/amartyavarma/ppm/code-811"
sys.path.insert(0, f"{C}/lib"); sys.path.insert(0, C)
import data_ppm as D  # noqa: E402
import chat_ppm as CH  # noqa: E402
import s811 as S8  # noqa: E402

n = 0
# 1. FLIP_BIT round trip
for slot, r in ((0, 0), (0, 123456), (3, 2 ** 39 - 1), (15, 77)):
    c = D.code_of(slot, r)
    c2, fl = D.flip_split(c | D.FLIP_BIT)
    assert c2 == c and fl and D.split_code(c2) == (slot, r)
    assert D.flip_split(c) == (c, False)
    n += 1
a = np.array([D.code_of(1, 5), D.code_of(0, 9)], np.int64); a[1:] |= D.FLIP_BIT
assert [D.flip_split(x) for x in a] == [(D.code_of(1, 5), False), (D.code_of(0, 9), True)]; n += 1
# 2. the trainer's marking: rows of epoch >= 2 only (consumed 33722 of a 34740-row epoch: 1018 epoch-1 rows left)
rows = np.arange(34740, dtype=np.int64) * 7 + 1
ss, marks = S8.speech_stream(rows, 33722, rows[:34000], 20261021, 5, 5000)
sf0 = min([i for e, i in marks if e >= 2] + [len(ss)])
assert sf0 == 1018 and marks[0] == (1, 0) and marks[1] == (2, 1018), marks
ss2 = ss.copy(); ss2[sf0:] |= D.FLIP_BIT
fl = np.array([D.flip_split(x)[1] for x in ss2])
assert not fl[:1018].any() and fl[1018:].all() and np.array_equal(ss2 & ~D.FLIP_BIT, ss); n += 1
ss, marks = S8.speech_stream(rows, 33722, rows, 20261021, 5, 500)          # stream ends inside epoch 1: nothing flipped
assert min([i for e, i in marks if e >= 2] + [len(ss)]) == len(ss) == 500; n += 1
# 3. eob_to_user: P 2 system tokens, PREPEND 1, user turn [5 6] at frames 3, 4, gap, assistant [7 8 EOB] at 9..11, user [9] at 15, end
PAD, EOB = CH.PAD, CH.EOB
L = 20
lanes = np.full((2, L), PAD, np.int64)
lanes[1, :2] = [100, 101]
lanes[0, 3:5] = [5, 6]; lanes[1, 9:12] = [7, 8, EOB]; lanes[0, 15] = 9
out, ne, nx = CH.eob_to_user(lanes, 2)
want = lanes.copy(); want[1, 11] = PAD; want[0, 5] = EOB; want[0, 16] = EOB
assert np.array_equal(out, want) and (ne, nx) == (2, 0), (out, ne, nx); n += 1
lanes2 = lanes.copy(); lanes2[0, 19] = 10                                   # 15 and 19 = one user run ending on the last frame: no frame left
out2, ne2, nx2 = CH.eob_to_user(lanes2, 2)
assert (ne2, nx2) == (1, 1) and out2[0, 16] == PAD and out2[0, 20 - 1] == 10, (ne2, nx2); n += 1
print(f"test_flip_cpu: {n}/{n} OK")
