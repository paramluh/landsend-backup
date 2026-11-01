import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import json
from pathlib import Path
import numpy as np,torch,soundfile as sf
from safetensors.numpy import load_file
from transformers import MimiModel
SITE=Path('/home/param/websites/pose-playground');DEST=Path('/home/param/runs/pose-runs/pose-sites-20261030/duplex-live/clips');DEST.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
model=MimiModel.from_pretrained('/home/param/models/mimi',torch_dtype=torch.float32).to('cuda').eval()
with torch.inference_mode():
 for p in json.loads((SITE/'conditioning-manifest.json').read_text())['people']:
  ident=p['id']; opening=SITE/'openings'/ident
  for kind in ('voice','start'):
   src=opening/'voice_clip.safetensors' if kind=='voice' and (opening/'voice_clip.safetensors').is_file() else opening/'mimi_local.safetensors'
   codes=load_file(str(src))['codes'][:,:125].copy();a=model.decode(torch.as_tensor(codes,dtype=torch.long,device='cuda')[None]).audio_values[0,0].float().cpu().numpy()
   dst=DEST/(ident+'.wav' if kind=='voice' else 'start_'+ident+'.wav');sf.write(dst,np.clip(a,-1,1),24000,subtype='PCM_16')
  print('PREVIEW_READY',ident,flush=True)
