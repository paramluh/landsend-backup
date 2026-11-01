"""ECAPA check of the 8 Daisy/Ada excerpts: cosine of each to the centroid of the other 7 -> aloe/ada_ecapa.json"""
import json
import numpy as np, soundfile as sf, torch, torchaudio
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy/aloe"
from speechbrain.inference.speaker import EncoderClassifier
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ec = EncoderClassifier.from_hparams(source="/data/home/amartyavarma/data/voicebank-yodas3/models/spkrec-ecapa-voxceleb", savedir=f"{D}/ecapa", run_opts={"device": dev})
A = json.load(open(f"{D}/ada.json"))
E = []
for x in A["excerpts"]:
    w, sr = sf.read(f"{D}/audio/{x['file'][:-4]}.wav", dtype="float32")
    with torch.no_grad():
        e = ec.encode_batch(torchaudio.functional.resample(torch.from_numpy(w)[None], sr, 16000).to(dev))[0, 0].float().cpu().numpy()
    E.append(e / np.linalg.norm(e))
E = np.stack(E)
sims = []
for i in range(len(E)):
    c = np.delete(E, i, 0).mean(0); c /= np.linalg.norm(c)
    sims.append(round(float(E[i] @ c), 3))
print(sims)
json.dump({"loo_cos": sims}, open(f"{D}/ada_ecapa.json", "w"))
