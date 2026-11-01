import json,sys,hashlib,time,unittest
from pathlib import Path
from dataclasses import asdict
source=Path('/home/param/pose-repo/tools/pose_jump_audit');sys.path.insert(0,str(source))
from strict_detection_r3 import VERSION,Config
out=Path(__file__).parent
suite=unittest.TestLoader().discover(str(source),pattern='test_strict_detection_r3*.py')
def ids(s):
 for x in s:
  if isinstance(x,unittest.TestSuite):yield from ids(x)
  else:yield x.id()
names=list(ids(suite));result=unittest.TestResult();suite.run(result)
near=json.loads((out/'near-misses.json').read_text());sha=hashlib.sha256((source/'strict_detection_r3.py').read_bytes()).hexdigest();assert near['checker_sha256']==sha
report={'version':VERSION,'checker_sha256':sha,'created':time.time(),'config':asdict(Config()),'tests_run':result.testsRun,'tests_passed':result.wasSuccessful(),'test_ids':names,'failures':[(x.id(),t) for x,t in result.failures],'errors':[(x.id(),t) for x,t in result.errors],'skipped':[(x.id(),why) for x,why in result.skipped],'test_source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('test_strict_detection_r3*.py')},'known_cases':near['cases'],'primary_motion_definition':'One Euclidean/RMS coordinate displacement from frame f-1 to frame f. Head jump must exceed10x the larger median raw frame-difference speed of8frames before and8frames after, including the immediately following frame. No excluded intermediate frame or two-frame jump variant.','confidence_definition':'Only destination anchors >=0.4, the actual yellow/green threshold, vote. At least3head anchors and both shoulders. Earlier low confidence is allowed; confidence-only change is not enough.','smoothness_definition':'Same one-boundary model must fit before/after segments at4/6/8frame scales, with low residual and improvement over a continuous cubic. Broad raw-motion p90context includes all adjacent frames. Persistent head/torso offset must remain.','hold_definition':'No prefix stationary/held-pose gate and no token-code ID or run-length acceptance rule. Prior movement is allowed; code/plateau descriptions are diagnostic only.','interpretation':'Behavioral controls are not measured precision/recall. Originalv1 passes;243 remains a user-confirmed positive but fails the newer confidence eligibility rule;145 is a confirmed negative. Other candidates require actual video review.'}
(out/'controls.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['version','checker_sha256','tests_run','tests_passed','skipped']}))
if not result.wasSuccessful():sys.exit(1)
