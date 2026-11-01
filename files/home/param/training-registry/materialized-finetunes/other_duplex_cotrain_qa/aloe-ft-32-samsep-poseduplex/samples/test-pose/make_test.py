"""fake run dir samples/test-pose/B-000 from real aloe-handover pose tokens (cropped to 500 frames = 40 s); pose_s* = gt for first 125 tokens
(10 s prefill) then tokens from later in the same cut (stand-in for 'generated'); first 2 columns of pose_s* set to -1."""
import glob, json, os, sys
import numpy as np, soundfile as sf
D = "/data/home/amartyavarma/data/aloe-handover-full/DATA/files/none"
OUT = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/test-pose/B-000"
os.makedirs(OUT, exist_ok=True)
T = 500
for sd in sorted(glob.glob(f"{D}/*/seg00"))[:50]:
    a = np.load(f"{sd}/w00.SPEAKER_00.pose_tokens.npy"); b = np.load(f"{sd}/w00.SPEAKER_01.pose_tokens.npy")
    if a.shape[1] >= 3 * T and b.shape[1] >= 3 * T and (a[:, :2 * T] >= 0).all() and (b[:, :2 * T] >= 0).all():
        break
print(sd, a.shape, b.shape, a.dtype)
for s, x in (("s1", a), ("s2", b)):
    gt = x[:, :T].astype(np.int16)
    gen = np.concatenate([x[:, :125], x[:, 1000:1000 + T - 125]], 1).astype(np.int16)
    gen[:, :2] = -1
    np.save(f"{OUT}/gt_pose_{s}.npy", gt); np.save(f"{OUT}/pose_{s}.npy", gen)
sess = sd.split("/")[-2]
sr = 24000
t = np.arange(T * 1920) / sr
for fn, f0 in (("s1.wav", 220), ("s2.wav", 330), ("gt_s1.wav", 220), ("gt_s2.wav", 330)):
    y = 0.1 * np.sin(2 * np.pi * f0 * t) * (np.sin(2 * np.pi * 0.25 * t) > 0)
    sf.write(f"{OUT}/{fn}", y.astype(np.float32), sr)
words = lambda off: [{"t": round(i * 0.8 + off, 2), "text": f"w{i}", "prompt": i * 0.8 < 10} for i in range(48)]  # noqa: E731
json.dump({"s1": words(0), "s2": words(0.4), "gt": {"s1": words(0.1), "s2": words(0.5)}}, open(f"{OUT}/words.json", "w"))
meta = {"step": 999, "prov": {"chan_B_data_side": "assistant", "row_key": f"ALOE/s2handover-pose/{sess}/seg00/c00"},
        "cards": {"s1": {"text": "", "n_tokens": 0, "clip": None}, "s2": {"text": "", "n_tokens": 0, "clip": None}},
        "sampling": {"text_temp": 1.0, "text_topk": 50, "audio_temp": 1.0, "audio_topk": 30, "pose_temp": 1.0, "pose_topk": 30},
        "prompt_s": 10.0, "gen_s": 30.0, "order": [{"kind": "speech", "spk": "s1"}, {"kind": "speech", "spk": "s2"}],
        "gen_words": {"s1": 36, "s2": 36}, "audio": {"s1_gen": {}, "s2_gen": {}}}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
open(f"{OUT}/DONE", "w").write("ok\n")
print("ok", OUT)
