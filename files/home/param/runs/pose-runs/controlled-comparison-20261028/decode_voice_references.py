import pathlib,sys,json,hashlib,gc
import numpy as np
import torch
import soundfile as sf
sys.path.insert(0,'/home/param/pose-repo/infer/engine/ppm')
import ppm_io as IP
root=pathlib.Path('/home/param/pose-runs/controlled-comparison-20261028')
torch.cuda.set_device(0)
mimi=IP.Mimi('cuda:0')
reports=[]
for i in range(1,9):
 d=root/f'clip-{i:02d}';cp=d/'voice_clip.npy';codes=np.load(cp,allow_pickle=False)
 assert codes.shape==(32,100)
 wav=mimi.decode(codes)
 assert wav.shape==(192000,) and bool(np.isfinite(wav).all())
 out=d/'voice_reference.wav';IP.write_wav(str(out),wav)
 written,sr=sf.read(out,dtype='float32');assert sr==24000 and written.shape==(192000,)
 info=dict(clip_id=d.name,codes_file=str(cp),codes_file_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),wav=str(out),wav_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),sample_rate=sr,samples=int(len(written)),duration_seconds=len(written)/sr,all_finite=True,decoded_peak=float(np.abs(wav).max()),decoded_clipped_samples=int((np.abs(wav)>1).sum()),decoded_clipped_fraction=float((np.abs(wav)>1).mean()),written_peak=float(np.abs(written).max()),audio_stats=IP.audio_stats(wav),codec_repo=IP.MIMI_REPO,codec_revision=IP.MIMI_REV,codec_local_directory=IP.MIMI,execution_node='landsend-1',visible_physical_gpu=2)
 (d/'voice-wav-validation.json').write_text(json.dumps(info,indent=2));reports.append(info);print(json.dumps(info),flush=True)
(root/'voice-wav-validation.json').write_text(json.dumps(dict(status='complete',references=reports),indent=2))
del mimi;gc.collect();torch.cuda.empty_cache()
