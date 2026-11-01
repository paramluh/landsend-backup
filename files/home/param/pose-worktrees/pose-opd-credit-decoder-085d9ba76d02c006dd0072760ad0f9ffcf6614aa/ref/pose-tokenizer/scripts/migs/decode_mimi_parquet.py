import argparse
import wave

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import MimiModel


def write_wav(path, audio, sr):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


parser = argparse.ArgumentParser()
parser.add_argument("parquet")
parser.add_argument("--row", type=int, default=0)
parser.add_argument("--out", default="decoded_mimi.wav")
args = parser.parse_args()

row = pq.read_table(args.parquet).slice(args.row, 1).to_pylist()[0]
codes = torch.tensor(row["mimi_codes"], dtype=torch.long).unsqueeze(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MimiModel.from_pretrained(row.get("tokenizer") or "kyutai/mimi").to(device).eval()
with torch.inference_mode():
    audio = model.decode(codes.to(device))[0].detach().cpu().numpy().squeeze()
write_wav(args.out, audio, 24000)
print(f"wrote {args.out}")
