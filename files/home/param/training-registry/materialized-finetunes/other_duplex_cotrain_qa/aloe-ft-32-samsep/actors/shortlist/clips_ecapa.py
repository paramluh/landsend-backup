"""ECAPA cosine between each shortlisted speaker's 2 excerpts (one voice check) -> shortlist/clips_ecapa.json"""
import json
import numpy as np, soundfile as sf, torch, torchaudio
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/shortlist"
from speechbrain.inference.speaker import EncoderClassifier
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
ec = EncoderClassifier.from_hparams(source="/data/home/amartyavarma/data/voicebank-yodas3/models/spkrec-ecapa-voxceleb",
                                    savedir="/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/speakers/ecapa", run_opts={"device": dev})
C = json.load(open(f"{D}/clips.json"))["clips"]
out = {}
for acc, cl in C.items():
    E = []
    for x in cl:
        w, sr = sf.read(f"{D}/audio/{x['file'][:-4]}.wav", dtype="float32")
        with torch.no_grad():
            e = ec.encode_batch(torchaudio.functional.resample(torch.from_numpy(w)[None], sr, 16000).to(dev))[0, 0].float().cpu().numpy()
        E.append(e / np.linalg.norm(e))
    out[acc] = round(float(E[0] @ E[1]), 3) if len(E) == 2 else None
    print(acc[:8], out[acc])
json.dump(out, open(f"{D}/clips_ecapa.json", "w"))
