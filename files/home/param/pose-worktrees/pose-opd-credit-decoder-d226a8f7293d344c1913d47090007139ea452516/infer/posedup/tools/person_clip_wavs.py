"""10-25: the voice-clip preview wavs of the 20 video-person presets (preset:<NN>_<hex>) for the proxy's /clipwav: the Mimi decode of exactly the
clip the prompt holds (gt/mimi_local codes [:, :125]) -> clips/<id>.wav (24 kHz int16). CPU is fine."""
import os, sys, numpy as np, torch, soundfile as sf
from safetensors.numpy import load_file
from transformers import MimiModel
R = "/data/home/amartyavarma/video/demo-data/prompts"; OUT = "/data/home/amartyavarma/ppm/aloe-posedup-live/clips"
m = MimiModel.from_pretrained(os.environ.get("MIMI_DIR", "/data/home/amartyavarma/models/mimi")).eval()
for d in sorted(os.listdir(R)):
    f = f"{R}/{d}/gt/mimi_local.safetensors"
    if not os.path.isfile(f) or os.path.exists(f"{OUT}/{d}.wav"):
        continue
    c = torch.as_tensor(np.asarray(load_file(f)["codes"], np.int64)[:, :125])[None]
    with torch.inference_mode():
        w = m.decode(c).audio_values[0, 0].float().numpy()
    sf.write(f"{OUT}/{d}.wav.tmp.wav", np.clip(w, -1, 1), 24000, subtype="PCM_16"); os.replace(f"{OUT}/{d}.wav.tmp.wav", f"{OUT}/{d}.wav"); print(d, len(w) / 24000, flush=True)
