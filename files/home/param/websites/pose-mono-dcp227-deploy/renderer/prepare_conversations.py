#!/usr/bin/env python3
"""Export authentic paired evaluation conversations for pose-duplex genweb."""
from pathlib import Path
import hashlib,json,sys,subprocess
import numpy as np
import soundfile as sf
LIB=Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/code/train/code/lib')
sys.path.insert(0,str(LIB))
import s3rows as R, speech_q as SP, sysprompt as SPR
BASE=Path('/home/param/datasets/aloe-duplex')
OUT=Path('/home/param/datasets/pose-duplex/convs')
FRAMES=1500

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 pix=json.loads((BASE/'pose/POSE_INDEX.json').read_text())
 aix=json.loads((BASE/'pose/M1VOX_INDEX_samsep.json').read_text())
 choices=[]; raw_nodes={}; raw_index={}
 for k in sorted(pix['rows'],key=int):
  store=BASE/f'original/stores/m1n{k}'; idx={}
  for line in (store/'rows.jsonl').open():
   row=json.loads(line);idx[row['row_key']]=row
  raw_index[k]=idx
  for r,meta in sorted(pix['rows'][k].items(),key=lambda x:int(x[0])):
   if meta['split']!='eval' or not meta['row_key'].endswith('/c00') or min(meta['T'],meta['n'])<FRAMES:continue
   if not all(meta['sides'][s].get('ok') for s in ('assistant','user')):continue
   audio=aix['rows'][k][int(r)];assert audio['row_key']==meta['row_key']
   if not all(audio['A'][s]['kind']=='gcs' and (BASE/'original/flac'/audio['A'][s]['flac']).is_file() for s in ('assistant','user')):continue
   raw=idx.get(meta['row_key'])
   if not raw or raw['n_frames']<2*FRAMES:continue
   choices.append((k,int(r),meta,audio,raw))
   if len(choices)==8:break
  if len(choices)==8:break
 assert len(choices)==8, len(choices)
 summary=[];offset=np.arange(32)*2048
 for i,(k,r,meta,audio,raw) in enumerate(choices):
  if k not in raw_nodes:raw_nodes[k]=R.Rows(R.load_node(str(BASE/f'original/stores/m1n{k}')))
  row=raw_nodes[k][raw['row']];assert row['row_key']==meta['row_key']
  arrays={};dirs=[]
  for side_n,side in enumerate(('assistant','user')):
   pose_path=BASE/'pose/data'/meta['sides'][side]['path']; assert sha(pose_path)==meta['sides'][side]['sha256']
   pc=np.load(pose_path,mmap_mode='r')[:,:FRAMES];mc=np.asarray(row[f'codes_{side}'][:,:FRAMES])
   assert pc.shape==mc.shape==(32,FRAMES) and pc.min()>=0 and mc.min()>=0 and pc.max()<2048 and mc.max()<2048
   db=np.full((FRAMES+2,32),-1,np.int64);pb=db.copy();db[2:]=mc.T+offset;pb[2:]=pc.T+offset
   wif=np.zeros(FRAMES+2,np.int64);lane=SP.lane(row[f'words_{side}'],row[f'piece_ids_{side}'],row[f'piece_offsets_{side}'],FRAMES+2,wif);SP.insert_new_word(lane,wif)
   # Authentic own-speaker voice reference, disjoint from the first120s replay.
   w0=max(FRAMES,int(row['n_frames'])//2); clip=np.asarray(row[f'codes_{side}'][:,w0:w0+125]).copy();assert clip.shape==(32,125)
   segment={'kind':'speech','track':1,'w0':0,'L':125,'src_row':{'codes_user':clip,'codes_assistant':clip},'src_side':'user','blank_lane':True}
   pre=SPR.build_prefix({},[segment],{'new_word':True})
   a=audio['A'][side];path=BASE/'original/flac'/a['flac']
   decoded=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(int(a['audio_frame0'])/12.5),'-i',str(path),'-t',str(FRAMES/12.5),'-ac','1','-ar','24000','-f','f32le','pipe:1'])
   stem=np.frombuffer(decoded,dtype='<f4').copy()
   assert stem.shape==(FRAMES*1920,) and np.isfinite(stem).all()
   arrays.update({f'stem_{side_n}':stem,f'laneB_{side_n}':lane,f'depB_{side_n}':db,f'pdepB_{side_n}':pb,f'pre_ids_{side_n}':pre['ids'],f'pre_depth_{side_n}':pre['depth'],f'pre_present_{side_n}':pre['present']})
   dirs.append({'chan_B_data_side':side,'speaker':raw[f'{side}_speaker'],'raw_store_row':int(raw['row']),'voice_clip_frames':[w0,w0+125],'voice_clip_source':'same speaker, same source row second half, disjoint from 120s replay','pose_source':str(pose_path),'pose_sha256':meta['sides'][side]['sha256'],'audio_source':str(path),'audio_gain_db':0.0,'audio_gain_mode':'raw, matching export424 VOX_A_GAIN=raw'})
  np.savez_compressed(OUT/f'c{i}.npz',**arrays)
  j={'id':f'c{i}','split':'eval','row_key':meta['row_key'],'conversation':meta['conv'],'seconds':FRAMES/12.5,'raw_store':str(BASE/f'original/stores/m1n{k}'),'sam_pose_index_row':r,'dirs':dirs,'data_rule':'Both original speakers of one authentic heldout conversation. Raw Mimi, aligned words, verified pose, raw-gain WLT FLAC. No synthetic conditioning.'}
  (OUT/f'c{i}.json').write_text(json.dumps(j,indent=2)+'\n')
  summary.append({'id':f'c{i}','row_key':meta['row_key'],'sha256':sha(OUT/f'c{i}.npz')});print('READY',f'c{i}',meta['row_key'],flush=True)
 (OUT/'MANIFEST.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
