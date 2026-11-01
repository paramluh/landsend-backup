"""Verify the real four-sample archive against source prefills and rollout masks."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import ProductionRows
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
root=a.run
cfg=TrainingConfig.from_dict(json.loads((root/'recipe.json').read_text()))
assert cfg.partner_sampling=='website_20261031' and cfg.partner_pad_bias==.125
folder=root/'run/raw-rollouts/update-000001';manifest=json.loads((folder/'manifest.json').read_text())
assert len(manifest['samples'])==4 and not manifest['synthetic']
data=ProductionRows(cfg);results=[]
for item in manifest['samples']:
 path=folder/item['file'];assert path.stat().st_size==item['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 assert item['conversation_mode']=='symmetric_streaming'
 assert item['incoming_audio_delay_samples']==item['output_audio_delay_samples']==3912
 assert item['output_audio_source']=='partner_received_pcm'
 assert item['frames']==1500 and 38<=item['prefill_frames']<=62
 row=data.load(tuple(item['item']),item['rank'],update=0,slot=item['slot'])
 assert int(row['prefill_frames'])==item['prefill_frames']
 n=item['prefill_frames'];delay=3912
 with np.load(path,allow_pickle=False) as arrays:
  for key,prefix in (('input_audio',row['wav']),('output_audio',row['partner']['wav'])):
   value=arrays[key];assert value.dtype==np.float32 and value.shape==(1500*1920,) and np.isfinite(value).all()
   assert np.array_equal(value[:delay],np.zeros(delay,np.float32))
   assert np.array_equal(value[delay:delay+n*1920],prefix)
   assert np.max(np.abs(value[delay+n*1920:]))<=.995001
  for stream in ('text','audio','pose'):
   mask=arrays[f'sampled_{stream}_mask'];assert mask.shape[0]==1500
   assert not mask[:n].any() and mask[n:].all()
   modelmask=arrays[f'sampled_{stream}_model_mask'];lp=arrays[f'behavior_{stream}_logprobs']
   expected_shape=modelmask.shape if stream=='text' else (*modelmask.shape,32)
   assert lp.shape==expected_shape and np.isfinite(lp[modelmask]).all() and (lp[modelmask]<=1e-6).all()
  for stream in ('audio','pose'):
   codes=arrays[f'{stream}_codes'];assert codes.shape==(32,1500) and (codes>=0).all() and (codes<2048).all()
  results.append({'id':item['id'],'frames':item['frames'],'prefill_frames':n,'source_prefills_exact':True,
   'input_audio_sha256':hashlib.sha256(arrays['input_audio'].tobytes()).hexdigest(),
   'output_audio_sha256':hashlib.sha256(arrays['output_audio'].tobytes()).hexdigest(),
   'both_delays_samples':delay,'conversation_mode':item['conversation_mode'],
   'prefill_in_sampled_masks':False,'sampled_masks_cover_all_generated_tokens':True,'finite_behavior_logprobs':True,
   'audio_codes_shape':list(arrays['audio_codes'].shape),'pose_codes_shape':list(arrays['pose_codes'].shape)})
out={'time':time.time(),'passed':True,'samples':results,'manifest':str(folder/'manifest.json')}
(root/'SAMPLES-VERIFIED.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
