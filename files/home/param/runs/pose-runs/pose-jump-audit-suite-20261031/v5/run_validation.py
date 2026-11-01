import json,sys,hashlib,time,unittest
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,'/home/param/pose-repo/tools/pose_jump_audit')
from strict_detection import VERSION,Config,prepare,evaluate
source=Path('/home/param/pose-repo/tools/pose_jump_audit');out=Path(__file__).parent
suite=unittest.TestLoader().discover(str(source),pattern='test_strict_detection*.py')
def ids(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from ids(x)
  else:yield x.id()
tests=list(ids(suite));result=unittest.TestResult();suite.run(result)
cases=[('v1',Path('/home/param/runs/pose-runs/pose-sites-20261030/video-inspection/v1-19-21/pose_overlay.json'),512,True),('rollout-000243',Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000243/pose_overlay.json'),292,True),('rollout-000145',Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000145/pose_overlay.json'),474,False)]
controls=[]
for name,path,f,label in cases:
 a=evaluate(prepare(json.loads(path.read_text())['frames']),f);controls.append({'case':name,'source_pose_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'user_confirmed_jump':label,'assessment':a})
report={'version':VERSION,'checker_sha256':hashlib.sha256((source/'strict_detection.py').read_bytes()).hexdigest(),'created':time.time(),'config':asdict(Config()),'tests_run':result.testsRun,'tests_passed':result.wasSuccessful(),'test_ids':tests,'failures':[(x.id(),trace) for x,trace in result.failures],'errors':[(x.id(),trace) for x,trace in result.errors],'skipped':[(x.id(),why) for x,why in result.skipped],'test_source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('test_strict_detection*.py')},'known_cases':controls,'criteria_history':'Initial pre243 stillness-based proposal failed the second user positive. Final version models a smooth velocity change after a dominant one-frame step and requires persistent pose configuration. No run IDs are excluded.','single_frame_rule':'At least2/3 of first+second-frame path length must occur in first frame, independently for head and shoulders: jump is at least twice subsequent settling. Same duration must explain persistence and all-scale fits.','interpretation':'Controls establish intended algorithm behavior, not population precision/recall. Two user positives and one individually labeled negative are known; other surfaced candidates remain unconfirmed.'}
(out/'controls.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['version','checker_sha256','tests_run','tests_passed','skipped']}))
if not result.wasSuccessful():sys.exit(1)
