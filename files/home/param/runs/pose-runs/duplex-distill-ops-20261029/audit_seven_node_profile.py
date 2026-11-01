import datetime,hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);health=json.loads((Path('/home/param/pose-runs/duplex-distill-ops-20261029')/(root.name+'-health.json')).read_text())
prof=json.loads((root/'profile-all-ranks.json').read_text());cfg=json.loads((root/'config.json').read_text());plan=json.loads((root/'plan.json').read_text())
rows=[r for rr in prof['records'] for r in rr];events=[e for n in health['nodes'].values() for e in n['progress']]
world=56;steps=prof['steps'];mbs=cfg['micro_batch_size'];expected={(r,s) for r in range(world) for s in range(1,steps+1)}
profile_pairs=[(r['rank'],r['step']) for r in rows];event_pairs=[(e['rank'],e['step']) for e in events]
per_rows=[{'rank':r['rank'],'step':r['step'],'slot':i,**rr} for r in rows for i,rr in enumerate(r['per_row_counts'])]
mismatches=[r for r in per_rows if r['pose_depth_frames'] not in (0,r['speech_depth_frames'])]
conditioning=[c for e in events for c in (e['conditioning'] if isinstance(e['conditioning'],list) else [e['conditioning']])]
checks={
 'world56_seven_configured_nodes':prof['world']==cfg['world']==56 and list(health['nodes'])==[f'landsend-{i}' for i in (0,1,2,3,4,5,12)],
 'all56ranks_all_expected_steps':set(profile_pairs)==expected and len(rows)==world*steps,
 'all56event_logs_all_expected_steps':set(event_pairs)==expected and len(events)==world*steps,
 'all7launchers_exit0':all(n['finished'] and n['finished']['exit_code']==0 for n in health['nodes'].values()),
 'all56gpus_idle':all(not n['gpu_pids'] and len(n['gpus'])==8 and all(r.split(',')[1].strip()=='0' for r in n['gpus']) for n in health['nodes'].values()),
 'no_node_log_errors':all(not n['errors'] for n in health['nodes'].values()),
 'every_actual_batch_row_count_matches_mbs':all(r['row_count']==mbs and len(r['per_row_counts'])==mbs for r in rows),
 'full7500conversation_budget_no_added_crop':cfg['conversation_frames']==7500 and cfg['data']['crop_conv'] is False,
 'valid_pose_count_equals_speech_per_actual_row':not mismatches,
 'plan_digest_expected56':plan['digest']=='05ce3e0b2bbe519f' and plan['epoch0_digest']=='d1a3e3721328da0a',
 'plan_preserves31796real_examples12dummies':plan['n_train_micros']==31796 and plan['n_dummy']==12,
 'two_epoch_schedule_resolved_for_mbs':cfg['steps']==568//mbs and cfg['epochs']==2,
 'real_mbs_global_batch_not_accumulation':cfg['global_batch_size']==56*mbs and cfg['gradient_accumulation']==1 and mbs in (1,2),
 'frozen_teachers_all_completed_events':all(e['teacher_gradients'] is False for e in events),
 'finite_metrics_all_completed_events':all(math.isfinite(e['kl']) and math.isfinite(e['data_ce']) for e in events),
 'pose_teacher_native_empty_all_rows':all(c['pose_teacher_a']=='native_empty' for c in conditioning),
 'full_architecture':cfg['smoke_depth'] is None,
 'production_not_resumed':prof['production_training_resumed'] is False,
 'checkpoint_not_written':prof['checkpoint_written'] is False,
 'shared_ten_percent_depth_sample':cfg['depth_frame_sampling']['shared_conversation_frames'] and cfg['depth_frame_sampling']['speech']==cfg['depth_frame_sampling']['pose']==0.1,
 'forward_kl_temperature1':cfg['objective']=='forward_kl_teacher_to_student_only' and cfg['temperature']==1.0,
 'equal_modality_weights':all(cfg['modality_weights'][k]==1/3 for k in ('text','audio','pose')),
 'audio_pose_cb0quarter_others_mean_threequarters':cfg['w_cb0']==cfg['w_pcb0']==0.25 and cfg['w_cb']==cfg['w_pcb']==0.75,
 'direct_addmm512':cfg['text_head']['direct_addmm'] is True and cfg['text_head']['chunk_size']==512,
}
source_path=root/'verification/source-config-audit.json'
if source_path.exists():
 source=json.loads(source_path.read_text())
 for key in ('all_source_hashes_match_snapshot','all_configs_identical','all_plans_identical'):checks[key]=source[key]
report={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':str(root),'launch':health['launch'],'all_checks_pass':all(checks.values()),'checks':checks,'profile_records':len(rows),'actual_rows':len(per_rows),'valid_pose_rows':sum(r['pose_depth_frames']>0 for r in per_rows),'missing_pose_or_dummy_rows':sum(r['pose_depth_frames']==0 for r in per_rows),'frame_count_mismatches':mismatches,'max_peak_allocated_GB':max(r['max_memory_GB'] for r in rows),'timeline_frames_min_max':[min(r['timeline_frames'] for r in per_rows),max(r['timeline_frames'] for r in per_rows)],'micro_batch_size':mbs,'global_batch_size':cfg['global_batch_size'],'plan':{k:plan[k] for k in ('digest','epoch0_digest','n_train_micros','n_dummy','steps','epoch_steps','world')},'nodes':{h:{'finished':n['finished'],'gpus':n['gpus'],'gpu_pids':n['gpu_pids'],'events':len(n['progress']),'errors':n['errors']} for h,n in health['nodes'].items()},'artifacts':{name:{'bytes':(root/name).stat().st_size,'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in ('profile-all-ranks.json','config.json','metrics.jsonl','plan.json','PROFILE_COMPLETE.json')},'scope_note':'All seven nodes including landsend-0 are explicitly authorized. Per-row profile counts verify shared speech/pose sample count equality; exact index identity also relies on the data path and its CPU tests.'}
(root/'verification/completion-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('all_checks_pass','checks','profile_records','actual_rows','valid_pose_rows','missing_pose_or_dummy_rows','max_peak_allocated_GB')},indent=2))
