"""ECAPA cross-check of the 'called Daisy' accounts: mean cosine between every pair of accounts' own-speech segments (diagonal = within one
account, across different conversations)."""
import json, itertools
import numpy as np, soundfile as sf, torch, torchaudio
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy/aloe"
from speechbrain.inference.speaker import EncoderClassifier
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ec = EncoderClassifier.from_hparams(source="/data/home/amartyavarma/data/voicebank-yodas3/models/spkrec-ecapa-voxceleb", savedir=f"{D}/ecapa", run_opts={"device": dev})
C = json.load(open(f"{D}/cands.json"))
E, who, conv = [], [], []
for acc, r in C.items():
    for s in r["segs"]:
        w, sr = sf.read(f"{D}/{s['file']}", dtype="float32")
        with torch.no_grad():
            e = ec.encode_batch(torchaudio.functional.resample(torch.from_numpy(w)[None], sr, 16000).to(dev))[0, 0].float().cpu().numpy()
        E.append(e / np.linalg.norm(e)); who.append(acc[:8]); conv.append(s["key"])
E = np.stack(E); accs = list(dict.fromkeys(who))
print("acc       " + " ".join(f"{a:>9}" for a in accs))
for a in accs:
    row = []
    for b in accs:
        v = [float(E[i] @ E[j]) for i in range(len(E)) for j in range(len(E)) if who[i] == a and who[j] == b and conv[i] != conv[j]]
        row.append(f"{np.mean(v):9.2f}" if v else "      n/a")
    print(f"{a:9} " + " ".join(row))
