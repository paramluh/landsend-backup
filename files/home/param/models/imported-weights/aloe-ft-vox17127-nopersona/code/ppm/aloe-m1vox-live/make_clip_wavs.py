"""preset voice-clip previews: the Mimi decode of each held-out preset's B clip codes (= what the model's system prompt holds) -> clips/<id>.wav
10-24: python make_clip_wavs.py [codes] [out dir]: codes = prompts_ppm.CODES key ('handover' default, 'samsep' = the sam-sep presets of a
checkpoint whose CK_LABELS.json entry says preset_codes samsep), out dir default <this dir>/clips (the sam-sep ones go to the instance state
dirs' clips/, which the proxy reads first)."""
_ALOE_ROOT = __import__("os").environ["ALOE_ROOT"]        # release: every path comes from aloe.env (ALOE_ROOT)
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts_ppm as PR  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import MimiModel  # noqa: E402

CODES = sys.argv[1] if len(sys.argv) > 1 else "handover"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{HERE}/clips"
os.makedirs(OUT, exist_ok=True)
mimi = MimiModel.from_pretrained(os.environ.get("ALOE_MIMI_DIR") or f"{_ALOE_ROOT}/models/mimi").eval()   # aloe.env ALOE_MIMI_DIR
pres = PR.Presets(codes=CODES)
for p in pres.items:
    c = p["clip"]
    codes = np.asarray(c["src_row"][f"codes_{c['src_side']}"])[:, int(c["w0"]):int(c["w0"]) + int(c["L"])].astype(np.int64)
    with torch.no_grad():
        w = mimi.decode(torch.as_tensor(codes)[None]).audio_values[0, 0].float().numpy()[:codes.shape[1] * 1920]
    sf.write(f"{OUT}/{p['id']}.wav", np.clip(w, -1, 1), 24000, subtype="PCM_16")
    print(p["id"], p["name"], c["src_key"], codes.shape, f"{len(w) / 24000:.2f}s rms {float(np.sqrt((w ** 2).mean())):.4f}", flush=True)
