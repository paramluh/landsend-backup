#!/usr/bin/env python3
"""Serve a genuine decoded pose preview from recorded inference frame feeds."""
import asyncio,json,re,sys,time
from collections import deque
from pathlib import Path
import numpy as np,torch
from aiohttp import web
SITE=Path('/home/param/websites/pose-playground')
STATE=Path('/home/param/runs/pose-runs/pose-sites-20261030/duplex-live')
sys.path.insert(0,str(SITE));import stick
SESSIONS={};LOCK=asyncio.Lock();SID=re.compile(r'^[A-Za-z0-9_.-]{1,100}$')
PAGE=Path(__file__).with_name('live_pose.html').read_text()

def session(sid):
 if not SID.fullmatch(sid or ''):return None
 if sid not in SESSIONS:SESSIONS[sid]={'offset':0,'pose':deque(maxlen=256),'numbers':deque(maxlen=256),'start':{},'frames':[],'sequence':-1,'decoded':-1,'last_seen':time.time(),'ended':False}
 return SESSIONS[sid]

def geometry(start):
 gs=start.get('frame0_global_state')
 if gs:return gs
 person=start.get('person') or start.get('voice')
 for base in (SITE/'openings',SITE/'openings-raw'):
  try:
   fields=json.loads((base/person/'meta.json').read_text())['sources']['pose_tokens']['frame0']['fields'];return [*fields['frame0_root'],fields['frame0_width']]
  except (OSError,ValueError,KeyError,TypeError):pass
 return None

def update(sid,s):
 paths=[STATE/'sessions'/sid/'feed.jsonl',SITE/'runs'/sid/'feed.jsonl']
 path=next((p for p in paths if p.is_file()),None)
 if path:
  with path.open() as f:
   f.seek(s['offset'])
   while line:=f.readline():
    if not line.endswith('\n'):break
    s['offset']=f.tell()
    try:m=json.loads(line)
    except ValueError:continue
    if m.get('type')=='start':s['start']=m
    elif m.get('type')=='end':s['ended']=True
    elif m.get('type')=='f' and isinstance(m.get('pose'),list) and len(m['pose'])==32 and min(m['pose'])>=0:
     s['pose'].append(m['pose']);s['numbers'].append(int(m['f']))
 if not s['pose'] or s['numbers'][-1]==s['decoded']:return
 gs=geometry(s['start'])
 if gs is None:s['message']='Waiting for the selected person’s pose geometry';return
 codes=np.asarray(s['pose'],np.int64).T
 # Preserve up to20.48s of genuine decoder context; no synthetic motion frames.
 pos,conf=stick.decode_pose(codes,gs)
 first=2*s['numbers'][-1]-len(pos)+2
 valid=[]
 for i in range(max(0,len(pos)-12),len(pos)):
  valid.append({'frame':first+i,'pos':np.round(pos[i],5).tolist(),'conf':np.round(conf[i],4).tolist()})
 s['frames']=valid;s['sequence']=first+len(pos)-1;s['decoded']=s['numbers'][-1]

async def index(request):return web.Response(text=PAGE,content_type='text/html')
async def health(request):return web.json_response({'ready':True,'kind':'decoded_pose_preview','actors':[p.name for p in (SITE/'openings').iterdir() if p.is_dir()]})
async def frame(request):
 sid=request.query.get('live') or request.query.get('session');s=session(sid)
 if s is None:return web.json_response({'frames':[],'message':'Choose a person and start a call'})
 s['last_seen']=time.time()
 try:after=int(request.query.get('after','-1'))
 except ValueError:after=-1
 async with LOCK:await asyncio.to_thread(update,sid,s)
 return web.json_response({'frames':[f for f in s['frames'] if f['frame']>after],'sequence':s['sequence'],'ended':s['ended'],'edges':stick.POSE_EDGES,'message':s.get('message','Waiting for model-generated pose')})
async def startup(app):
 torch.set_num_threads(2);await asyncio.to_thread(stick._tok)
app=web.Application();app.on_startup.append(startup)
for prefix in ('','/avatar3','/avatar2','/avatar'):
 app.router.add_get(prefix+'/',index);app.router.add_get(prefix+'/api/frame',frame);app.router.add_get(prefix+'/api/inputs',health);app.router.add_get(prefix+'/health',health)
web.run_app(app,host='127.0.0.1',port=9227)
