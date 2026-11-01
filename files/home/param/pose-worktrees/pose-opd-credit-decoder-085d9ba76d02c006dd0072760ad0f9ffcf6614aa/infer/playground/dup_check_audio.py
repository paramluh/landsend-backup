import subprocess, sys, numpy as np, soundfile as sf
mp4, wav = sys.argv[1], sys.argv[2]
FF = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static"
print(subprocess.run([f"{FF}/ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,duration,sample_rate", "-of", "csv", mp4], capture_output=True, text=True).stdout)
raw = subprocess.run([f"{FF}/ffmpeg", "-v", "error", "-i", mp4, "-ac", "1", "-ar", "24000", "-f", "f32le", "-"], capture_output=True).stdout
a = np.frombuffer(raw, np.float32); b, sr = sf.read(wav, dtype="float32")
print("mp4 audio s", len(a) / 24000, "rms", float(np.sqrt((a ** 2).mean())), "b.wav s", len(b) / 24000, "rms", float(np.sqrt((b ** 2).mean())))
n = min(len(a), len(b)); best = (0, 0)
for lag in range(-4800, 4801, 240):
    x = a[max(0, lag):n + min(0, lag)]; y = b[max(0, -lag):n - max(0, lag)]; m = min(len(x), len(y))
    c = float(np.corrcoef(x[:m], y[:m])[0, 1]); best = max(best, (c, lag))
print("best corr with b.wav", best)
def env(x, h=480):
    m = len(x) // h
    return np.sqrt((x[:m * h].reshape(m, h) ** 2).mean(1))
ea, eb = env(a), env(b)
best = max((float(np.corrcoef(ea[max(0, l):len(ea) + min(0, l)][:min(len(ea), len(eb)) - abs(l)], eb[max(0, -l):][:min(len(ea), len(eb)) - abs(l)])[0, 1]), l) for l in range(-10, 11))
print("20 ms envelope corr with b.wav (corr, lag frames)", best)
