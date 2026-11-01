"""ECAPA cosine matrix of Ashley's and her nearest voices' clips: mean cosine over pairs of different clips (diagonal = within one account) -> ecapa.json"""
import json
import numpy as np, soundfile as sf, torch, torchaudio
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/allvoices/ash"
from speechbrain.inference.speaker import EncoderClassifier
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ec = EncoderClassifier.from_hparams(source="/data/home/amartyavarma/data/voicebank-yodas3/models/spkrec-ecapa-voxceleb",
                                    savedir="/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/speakers/ecapa", run_opts={"device": dev})
C = json.load(open(f"{D}/clips.json"))
E = []
for x in C:
    w, sr = sf.read(f"{D}/audio/{x['file'][:-4]}.wav", dtype="float32")
    with torch.no_grad():
        e = ec.encode_batch(torchaudio.functional.resample(torch.from_numpy(w)[None], sr, 16000).to(dev))[0, 0].float().cpu().numpy()
    E.append(e / np.linalg.norm(e))
E = np.stack(E); accs = list(dict.fromkeys(x["acc"] for x in C))
M = [[None] * len(accs) for _ in accs]
for a, A in enumerate(accs):
    for b, B in enumerate(accs):
        v = [float(E[i] @ E[j]) for i in range(len(C)) for j in range(len(C)) if C[i]["acc"] == A and C[j]["acc"] == B and i != j]
        M[a][b] = round(float(np.mean(v)), 2) if v else None
for a, r in zip(accs, M):
    print(a[:8], r)
json.dump({"accs": accs, "M": M}, open(f"{D}/ecapa.json", "w"))
