#!/usr/bin/env python3
"""Exhaustively select exact held-code runs from historical original357 lanes.

No GPU, retokenization, pose modification, or target masking is performed.
"""
import concurrent.futures,collections,csv,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.ipc as ipc

BASE=Path('/home/param/datasets/blackband-repro-20261031')
HIST=Path('/home/param/datasets/aloe-duplex/pose/data')
RUN=Path('/home/param/runs/pose-runs/aloe-blackband-only-repro-20261031')
DATA=RUN/'data'
REV='bab8985c697f0bbba7a23a138586cb9f436c248b'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def split(session):return 'eval' if int.from_bytes(hashlib.blake2b(('aloe-pose-ft-v1|'+session).encode(),digest_size=8).digest(),'little')%10000<200 else 'train'

MAN={r['relative_dir']:r for r in csv.DictReader((HIST/'manifest.csv').open())}
POSE=collections.defaultdict(dict)
for r in csv.DictReader((HIST/'pose_manifest.csv').open()):POSE[r['relative_dir']][r['speaker']]=r
HF={(r['session_id'],r['seg']):r for r in pq.read_table(BASE/'source/index/conversations.parquet').to_pylist()}
SHARDS={(r['session_id'],r['seg']):r for r in pq.read_table(BASE/'source/tools/shard_index.parquet').to_pylist()}
FILES={f['path']:f for f in json.loads((BASE/'source-files.json').read_text())['files']}
ELIG=collections.defaultdict(list)
for rel,r in MAN.items():
    for speaker,p in POSE[rel].items():
        if p['pose_status']!='ok' or r['combined_crossbleed_side'] in [speaker,'both','unknown']:continue
        s=SHARDS[(r['session_id'],r['seg'])];T=int(r['mimi_T']);tail=T%7500;used=T-(tail if 0<tail<125 else 0)
        ELIG[s['shard']].append({'relative_dir':rel,'session_id':r['session_id'],'seg':r['seg'],'speaker_id':speaker,'b_side':'assistant' if speaker=='SPEAKER_00' else 'user','split':split(r['session_id']),'n_frames':T,'training_n_frames':used,'original_training_rows':math.ceil(used/7500),'row_index':s['row_index'],'original_pose_status':p['pose_status'],'crossbleed_side':r['combined_crossbleed_side'],'source_annotation_intervals_seconds':s[f'speaker_{int(speaker[-1]):02d}_no_detection_intervals'] or []})

def arrayscalar(table,name,row,T):
    a=table[name][row]
    if not a.is_valid:return None
    out=a.values.flatten().to_numpy(zero_copy_only=False).reshape(32,T)
    assert out.dtype==np.int16 and np.all((out>=0)&(out<2048))
    return out

def inspect_shard(shard):
    file=BASE/'source'/shard;digest=sha(file)
    assert digest==FILES[shard]['sha256'],(shard,'source shard checksum mismatch')
    cols=['session_id','seg','T']+[f'speaker_{i:02d}_{k}' for i in [0,1] for k in ['pose_tokens','mimi_tokens','words']]
    tab=pq.read_table(file,columns=cols)
    stats=collections.Counter();sources={};items=[];track_stats=[]
    for meta in ELIG[shard]:
        ri=meta['row_index'];T=meta['n_frames'];used=meta['training_n_frames'];j=int(meta['speaker_id'][-1]);pref=f'speaker_{j:02d}_'
        assert tab['session_id'][ri].as_py()==meta['session_id'] and tab['seg'][ri].as_py()==meta['seg'] and tab['T'][ri].as_py()==T
        local=HIST/meta['relative_dir']/f'w00.{meta["speaker_id"]}.pose_tokens.npy'
        hfpose=arrayscalar(tab,pref+'pose_tokens',ri,T)
        if local.exists():
            sidecar=local.with_name(local.name.replace('.pose_tokens.npy','.pose.json'));sc=json.loads(sidecar.read_text())
            assert sha(local)==sc['pose_tokens_sha256']
            pose=np.load(local,mmap_mode='r');assert pose.shape==(32,T) and pose.dtype==np.int16
            if hfpose is not None:assert np.array_equal(pose,hfpose),(str(local),'HF differs from historical pose')
            else:stats['historical_pose_restored_from_local']+=1
            pose_evidence={'kind':'historical_local_pose_tokens','path':str(local),'sha256':sc['pose_tokens_sha256'],'hf_pose_same':hfpose is not None}
        else:
            assert hfpose is not None;pose=hfpose
            pose_evidence={'kind':'pinned_hf_original_pose_tokens','source_shard':shard,'row_index':ri,'source_parquet_sha256':digest}
        assert np.all((pose>=0)&(pose<2048))
        equal=np.all(pose[:,1:used]==pose[:,:used-1],axis=0)
        boundaries=np.r_[0,np.flatnonzero(~equal)+1,used];lengths=np.diff(boundaries)
        allruns=[(int(s),int(e)) for s,e,n in zip(boundaries[:-1],boundaries[1:],lengths) if n>=7]
        label=meta['split'];stats[label+'_tracks']+=1;stats[label+'_valid_frames']+=used;stats[label+'_rows']+=meta['original_training_rows']
        for minimum in [7,13]:
            sizes=lengths[lengths>=minimum];stats[f'{label}_runs_ge{minimum}']+=len(sizes);stats[f'{label}_held_frames_inclusive_ge{minimum}']+=int(sizes.sum());stats[f'{label}_equal_transition_frames_ge{minimum}']+=int(np.maximum(sizes-1,0).sum())
        if allruns:stats[label+'_tracks_with_runs']+=1
        track_stats.append({**meta,'source_shard':shard,'pose_source':pose_evidence,'runs_ge7_frames':allruns,'valid_frames':used})
        selected={}
        for start,end in allruns:
            if used<750:stats[label+'_run_skipped_source_shorter60s']+=1;continue
            crop=min(max(end-375,0),used-750)
            if end+25>crop+750:stats[label+'_run_skipped_no2s_postexit']+=1;continue
            if min(end,crop+750)-max(start,crop)<7:stats[label+'_run_skipped_band_overlap_lt7']+=1;continue
            selected.setdefault(crop,[]).append((start,end))
        if not selected:continue
        sid=f'{meta["session_id"]}__{meta["seg"]}__{meta["speaker_id"]}'
        target=DATA/'sources'/sid;target.mkdir(parents=True,exist_ok=True)
        mimi=arrayscalar(tab,pref+'mimi_tokens',ri,T);assert mimi is not None
        npz=target/'tokens.npz';tmp=target/'tokens.tmp.npz';np.savez(tmp,pose=np.asarray(pose.T,dtype=np.int16),mimi=np.asarray(mimi.T,dtype=np.int16));tmp.replace(npz)
        words=tab[pref+'words'][ri].as_py() or [];write(target/'words.json',words)
        source={**meta,'id':sid,'npz':str(npz),'words_json':str(target/'words.json'),'npz_sha256':sha(npz),'words_sha256':sha(target/'words.json'),'pose_array_sha256':hashlib.sha256(pose.T.tobytes()).hexdigest(),'mimi_array_sha256':hashlib.sha256(mimi.T.tobytes()).hexdigest(),'row_key_prefix':f'ALOE/s2handover-pose/{meta["session_id"]}/{meta["seg"]}','source_shard':shard,'source_parquet_sha256':digest,'source_revision':REV,'pose_source':pose_evidence,'all_constant_pose_runs_frames':allruns,'band_definition':'maximal run with all32 unoffset int16 pose codes identical at every token; minimum7 tokens; no retokenization'}
        write(target/'source.json',source);sources[sid]=source
        for crop,bands in sorted(selected.items()):
            overlaps=[(s,e) for s,e in allruns if min(e,crop+750)-max(s,crop)>=7]
            item={'id':f'{sid}__f{crop:07d}','source_id':sid,'split':label,'row_key':source['row_key_prefix']+f'/c{crop//7500:02d}','b_side':meta['b_side'],'recording_id':meta['session_id']+':'+str(int(meta['seg'][3:])),'chunk_index':crop//7500,'crop_start_frame':crop,'crop_end_frame':crop+750,'band_intervals_frames':[list(b) for b in overlaps],'band_intervals_local_frames':[[max(s,crop)-crop,min(e,crop+750)-crop] for s,e in overlaps],'anchor_band_intervals_frames':[list(b) for b in bands],'selection_evidence':'exact_all32_constant_pose_codes','target_masking':False}
            items.append(item);stats[label+'_selected_crops']+=1
    return {'shard':shard,'stats':dict(stats),'sources':sources,'items':items,'track_stats':track_stats}

def main():
    DATA.mkdir(parents=True,exist_ok=True);start=time.time();pending=set(ELIG);futures={};results=[];stats=collections.Counter();source_map={};items=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        while pending or futures:
            ready=[s for s in sorted(pending) if (BASE/'source'/s).exists() and (BASE/'source'/s).stat().st_size==FILES[s]['size']]
            for s in ready[:max(0,24-len(futures))]:pending.remove(s);futures[pool.submit(inspect_shard,s)]=s
            done,_=concurrent.futures.wait(futures,timeout=.5,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result=future.result();results.append(result);stats.update(result['stats']);source_map.update(result['sources']);items.extend(result['items']);del futures[future]
                write(DATA/'preparation-status.json',{'state':'scanning_and_materializing','pid':os.getpid(),'completed_shards':len(results),'total_shards':len(ELIG),'stats':dict(stats),'elapsed_seconds':time.time()-start,'updated':time.time()})
                if len(results)%40==0:print(json.dumps({'completed_shards':len(results),'stats':dict(stats)}),flush=True)
    assert stats['train_tracks']==17742 and stats['train_rows']==27517 and stats['train_valid_frames']==136497507
    assert stats['eval_tracks']==351 and stats['eval_rows']==531
    # Preserve available original persona sections; absence is explicit, never fabricated.
    persona={}
    for path in Path('/home/param/datasets/aloe-duplex/original/stores').glob('m1n*/*.persona.arrow'):
        reader=ipc.open_file(str(path))
        for b in range(reader.num_record_batches):
            for row in reader.get_batch(b).to_pylist():
                parts=row['row_key'].split('/');key=(parts[-3],parts[-2],int(parts[-1][1:]))
                persona[key]={'source':str(path),'row_key':row['row_key'],'sides':{side:{k[len(side)+1:]:v for k,v in row.items() if k.startswith(side+'_')} for side in ['user','assistant']}}
    for sid,source in source_map.items():
        source['persona_by_chunk']={str(chunk):persona[(source['session_id'],source['seg'],chunk)] for chunk in range(math.ceil(source['n_frames']/7500)) if (source['session_id'],source['seg'],chunk) in persona}
        write(DATA/'sources'/sid/'source.json',source)
    items.sort(key=lambda x:(x['split'],x['id']))
    report={'state':'passed','source_revision':REV,'definition':'all32 pose codebooks identical for>=7 consecutive12.5Hz token frames; user-authorized operational blackband definition; not a pixel-blackness claim','denominator':'All true original357-eligible training pose frames; pose-statusok, no crossbleed into targetlane, originalsessionhashsplit; exclude padded/absentlanes and finalsourcechunk<125frames. Continuous source tracks scanned independently.','counts':dict(stats),'fractions':{f'ge{n}':{'inclusive_held_frame_fraction':stats[f'train_held_frames_inclusive_ge{n}']/stats['train_valid_frames'],'equal_transition_frame_fraction':stats[f'train_equal_transition_frames_ge{n}']/stats['train_valid_frames']} for n in [7,13]},'source_manifest_sha256':sha(HIST/'manifest.csv'),'source_pose_manifest_sha256':sha(HIST/'pose_manifest.csv'),'hf_index_sha256':sha(BASE/'source/index/conversations.parquet'),'script_sha256':sha(__file__),'source_rows_scanned':len(results),'tracks':[t for r in results for t in r['track_stats']]}
    write(DATA/'static-band-scan.json',report)
    manifest={'schema':'aloe-blackband-training-crops-v1','created':time.time(),'source_repo':'InternalCan/stage2-handover-pose-mimi-sam-v1','source_revision':REV,'selection':'exact_all32_static_runs_ge7','scan_report':str(DATA/'static-band-scan.json'),'scan_report_sha256':sha(DATA/'static-band-scan.json'),'fps':12.5,'clip_frames':750,'clip_seconds':60,'clips_per_sequence':10,'pose_loss_masking':False,'word_loss_masking':False,'audio_loss_masking':False,'crop_policy':'One60s window per constant run, normally30s before/after exit; clamp withinvalidsource; require>=7heldframes overlap and>=25tokenframes ofpostexit context; exactduplicatewindows deduplicated. No frozen-input changes.','sources':source_map,'items':items,'counts':{'train_crops':sum(i['split']=='train' for i in items),'eval_crops':sum(i['split']=='eval' for i in items),'selected_sources':len(source_map)},'conditioning':{'voice_probability':.8,'persona_probability':.2,'voice_reference':'original357 later-chunk, second-half same-source speech sampler; fullsource rawMimi+words retained','persona':'original sidecar provided where available; missing source persona explicitly absent'},'provenance':{'historical_manifest':str(HIST/'manifest.csv'),'historical_pose_manifest':str(HIST/'pose_manifest.csv'),'source_files_inventory':str(BASE/'source-files.json')},'no_targets_masked_for_bands':True}
    write(DATA/'manifest.json',manifest)
    write(DATA/'preparation-status.json',{'state':'complete','pid':os.getpid(),'completed_shards':len(results),'total_shards':len(ELIG),'stats':dict(stats),'manifest':str(DATA/'manifest.json'),'manifest_sha256':sha(DATA/'manifest.json'),'elapsed_seconds':time.time()-start,'updated':time.time()})
    print(json.dumps({'state':'complete','stats':dict(stats),'fractions':report['fractions'],'manifest':str(DATA/'manifest.json')}),flush=True)

if __name__=='__main__':main()
