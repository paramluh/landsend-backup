from pathlib import Path
import json,shutil
stage=Path('/home/param/websites/pose-mono-dcp227-deploy');site=stage/'site'; ops=stage/'ops';ops.mkdir(exist_ok=True)
worker=site/'worker.py';s=worker.read_text();old='p_max=prefix, mimi_stream=False, silence=self.silence)';new='p_max=prefix, mimi_stream=False, front=False, silence=self.silence)';assert s.count(old)==1;s=s.replace(old,new);worker.write_text(s)
config=json.loads(Path('/home/param/websites/pose-playground/MONO_CK.json').read_text());config.update(ck='/home/param/pose-mono-dcp227/models/dcp-227',label='Pose duplex first-band training, checkpoint 227 · monoplex inference',short='First-band · 227',prompt='v2pure');config['params']['pose_cfg_on']=0
config['run_defaults'].update(voice_clip=True,prefill_person='heldout_02',prefill_src='clip')
(site/'MONO_CK.json').write_text(json.dumps(config,indent=2)+'\n')
shutil.copy2('/home/param/websites/pose-playground/SCHEMA.json',site/'SCHEMA.json')
opts={'models':['monoplex'],'model_label':config['label'],'model_short':config['short'],'persons':[{'id':p.name,'label':json.loads((p/'meta.json').read_text()).get('label',p.name),'max_prefill_s':11.68} for p in sorted((stage/'prompts').iterdir()) if p.is_dir()],'defaults':config['run_defaults'],'max_clip_s':120,'max_prefill_s':11.68}
(site/'OPTIONS.json').write_text(json.dumps(opts,indent=2)+'\n')
base='/home/param/pose-mono-dcp227'
env=f'''#!/usr/bin/env bash
export CANOPY_ROOT={base}
export GENWEB={base}/site
export GENWEB_ROOT="$GENWEB"
export GENWEB_RUNTIME=systemd
export GENWEB_MONO_ENGINE={base}/engine
export GENWEB_LIVE={base}/live_support
export GENWEB_PEOPLE={base}/prompts
export GENWEB_ACTORS={base}/prompts
export GENWEB_PYTHON={base}/.venv/bin/python
export GENWEB_CK={base}/models/dcp-227
export GENWEB_WEIGHTS=base
export GENWEB_RENDERERS=param
export GENWEB_BATCH_STREAM=0
export GENWEB_FFMPEG=/usr/bin/ffmpeg
export MIMI_DIR={base}/models/mimi
export FA3_DIR={base}/kernels/fa3
export PPM_FA3_ROOT="$FA3_DIR"
export POSE_TOKENIZER_CODE={base}/engine/posetok
export POSE_TOKENIZER_DIR={base}/models/pose-tokenizer-32cb-200k-noshift
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR={base}/runtime/triton
export PATH={base}/.venv/bin:/usr/local/cuda/bin:$PATH
'''
(ops/'env.sh').write_text(env)
for role,cmd in [('web','exec "$GENWEB_PYTHON" -u server.py --port 9205'),('worker','export CUDA_VISIBLE_DEVICES=0,1\nexec "$GENWEB_PYTHON" -u worker.py mono-nebius')]:
 (ops/f'start_{role}.sh').write_text(f'#!/usr/bin/env bash\nset -euo pipefail\nsource {base}/ops/env.sh\ncd "$GENWEB"\n{cmd}\n')
 unit=f'''[Unit]
Description=Pose monoplex checkpoint 227 {role}
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory={base}/site
ExecStart=/bin/bash {base}/ops/start_{role}.sh
Restart=on-failure
RestartSec=15
TimeoutStopSec=30
StandardOutput=append:{base}/runtime/{role}.log
StandardError=append:{base}/runtime/{role}.log

[Install]
WantedBy=default.target
'''
 (ops/f'pose-mono-dcp227-{role}.service').write_text(unit)
print('Configured worker front=False, exact checkpoint and isolated VM units; frontend unchanged.')
