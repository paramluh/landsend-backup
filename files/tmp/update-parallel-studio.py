from pathlib import Path
p=Path('/home/param/pose-demo-website/comparison/request_server.py');s=p.read_text().replace('Authenticated, persistent single-job request studio; GPU worker is separate.','Authenticated, persistent parallel request studio; GPU dispatch is separate.')
s=s.replace("            row.update(id=path.name, request=request)","""            row.update(id=path.name, request=request)
            node, pair = state.get('assigned_node'), state.get('gpu_pair')
            if isinstance(node, str) and re.fullmatch(r'landsend-[1-4]', node):
                row['assigned_node'] = node
            if isinstance(pair, list) and len(pair) == 2 and len(set(pair)) == 2 and all(type(g) is int and 0 <= g <= 7 for g in pair):
                row['gpu_pair'] = pair
            if type(state.get('executor_pid')) is int and state['executor_pid'] > 0:
                row['executor_pid'] = state['executor_pid']""")
needle='    class Handler(BaseHTTPRequestHandler):'
insert='''    def public_capacity():
        path = queue / 'dispatcher-status.json'
        try:
            if not path.is_file() or path.stat().st_size > 262144:
                raise ValueError('Missing or oversized status')
            value = load(path)
            updated = datetime.fromisoformat(value['updated_at'].replace('Z', '+00:00'))
            if updated.tzinfo is None or value['status'] not in ('running', 'stopped'):
                raise ValueError('Invalid dispatcher heartbeat')
            counts = {key: value[key] for key in ('capacity_total', 'capacity_busy', 'capacity_free', 'queued_jobs', 'running_jobs')}
            if not all(type(n) is int and 0 <= n <= 100000 for n in counts.values()):
                raise ValueError('Invalid capacity counts')
            if not 0 < counts['capacity_total'] <= 16 or counts['capacity_busy'] + counts['capacity_free'] != counts['capacity_total']:
                raise ValueError('Inconsistent GPU-pair counts')
            slots, seen = [], set()
            if not isinstance(value['slots'], list) or len(value['slots']) > 16:
                raise ValueError('Invalid GPU-pair list')
            for slot in value['slots']:
                node, pair = slot['node'], slot['gpu_pair']
                if not isinstance(node, str) or not re.fullmatch(r'landsend-[1-4]', node):
                    raise ValueError('Unexpected compute node')
                if not isinstance(pair, list) or len(pair) != 2 or not all(type(g) is int and 0 <= g <= 7 for g in pair) or len(set(pair)) != 2:
                    raise ValueError('Invalid GPU pair')
                key = (node, tuple(pair))
                if key in seen:
                    raise ValueError('Repeated GPU pair')
                seen.add(key)
                jid = slot.get('job_id')
                stage = slot.get('stage', 'unknown')
                if jid is not None and (not isinstance(jid, str) or not re.fullmatch(r'[a-f0-9]{32}', jid)):
                    raise ValueError('Invalid assignment')
                if not isinstance(stage, str) or len(stage) > 80:
                    raise ValueError('Invalid compute stage')
                slots.append({'node': node, 'gpu_pair': pair, 'job_id': jid, 'stage': stage})
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            return {'available': True, 'status': value['status'], 'updated_at': value['updated_at'],
                    'stale': age > 60 or age < -60, **counts, 'slots': slots}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return {'available': False, 'status': 'unavailable', 'slots': []}

'''
assert needle in s;s=s.replace(needle,insert+needle)
s=s.replace("            if route == '/api/requests':\n                return self.send_json(200, {'requests': public_requests()})", "            if route == '/api/requests':\n                return self.send_json(200, {'requests': public_requests()})\n            if route == '/api/capacity':\n                return self.send_json(200, public_capacity())")
p.write_text(s)
p=Path('/home/param/pose-demo-website/comparison/studio.html');s=p.read_text().replace('Requests run one at a time.','Requests run in parallel across available GPU pairs.');s=s.replace('<div id="app" class="hidden"><div class="columns">','<div id="app" class="hidden"><section class="capacity" aria-live="polite"><strong id="capacity-summary">Checking compute capacity…</strong><span id="capacity-note" class="muted"></span><div id="capacity-nodes"></div></section><div class="columns">')
s=s.replace('</style>','.capacity{margin:0 0 24px;padding:15px 18px;background:#e8f2ef;border:1px solid #c9ded7;border-radius:10px}.capacity strong,.capacity>span{display:block}.capacity>span{font-size:12px;margin-top:3px}#capacity-nodes{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.node-status{font-size:12px;padding:4px 8px;background:#fff;border-radius:5px}\n</style>')
s=s.replace("const stages={'queued':'Queued'", "const stages={'claimed':'Assigned to a GPU pair','claiming':'Assigning a GPU pair','staging':'Preparing inputs','launching':'Starting generation','copying':'Transferring completed output','transferring':'Transferring completed output','collecting':'Collecting completed output','waiting-capacity':'Waiting for an available GPU pair','waiting-for-assets':'Waiting for model assets','queued':'Queued · waiting for capacity'")
s=s.replace("const signature=job.status+'|'+job.stage+'|'+job.updated_at;", "const signature=job.status+'|'+job.stage+'|'+job.updated_at+'|'+job.assigned_node+'|'+JSON.stringify(job.gpu_pair);")
s=s.replace(" const params=el('div','jobparams');", " const params=el('div','jobparams');\n if(job.assigned_node)params.append(param('Compute node',job.assigned_node),param('GPU pair',job.gpu_pair?.join(' + ')||'Assigning…'));")
needle='async function refresh(){'
insert='''async function refreshCapacity(){
 try{
  const capacity=await api('/api/capacity');if(!capacity)return;
  $('capacity-nodes').replaceChildren();
  if(!capacity.available){$('capacity-summary').textContent='Compute capacity status unavailable';$('capacity-note').textContent='Request states below remain available.';return}
  if(capacity.stale){$('capacity-summary').textContent='Compute status has not updated recently';$('capacity-note').textContent='Showing the last reported assignments; queued requests remain saved.'}
  else if(capacity.status==='stopped'){$('capacity-summary').textContent='Dispatch is paused';$('capacity-note').textContent='Queued requests remain saved until dispatch resumes.'}
  else{$('capacity-summary').textContent=`${capacity.capacity_busy} / ${capacity.capacity_total} GPU pairs assigned · ${capacity.capacity_free} unassigned`;$('capacity-note').textContent='Four nodes · up to 16 simultaneous requests. Each request uses a pair of GPUs.'}
  for(const node of [...new Set(capacity.slots.map(s=>s.node))].sort()){
   const slots=capacity.slots.filter(s=>s.node===node),busy=slots.filter(s=>s.job_id).length;
   const unavailable=slots.filter(s=>s.stage==='unreachable').length;
   $('capacity-nodes').append(el('span','node-status',`${node}: ${busy}/${slots.length} assigned${unavailable?' · '+unavailable+' unreachable':''}`));
  }
 }catch(e){$('capacity-summary').textContent='Compute capacity status unavailable';$('capacity-note').textContent='Request states below remain available.'}
}
'''
assert needle in s;s=s.replace(needle,insert+needle);s=s.replace("async function refresh(){if(!config)return;try{", "async function refresh(){if(!config)return;refreshCapacity();try{")
p.write_text(s)
p=Path('/home/param/pose-demo-website/comparison/README.md');s=p.read_text().replace('The separate inference worker runs one job at a time on its assigned GPU pair.\nIt must use the same queue root and shared schema. For this deployment the app\nand worker live on landsend-1; landsend-0\'s existing comparison stays untouched.','The central dispatcher on landsend-1 assigns independent requests to sixteen\nGPU pairs across landsend-1 through landsend-4. Each executor owns one pair;\ncompleted media is copied back before the central request becomes playable.\nThe app and dispatcher share the queue root and request schema.\nLandsend-0\'s existing comparison stays untouched.');s+='''\n`GET /api/capacity` reads the dispatcher's atomic `dispatcher-status.json` from\nthe queue root and exposes only capacity counts, node/pair assignments, stages,\nand the heartbeat timestamp. Missing or corrupt status is shown as unavailable;\na heartbeat older than sixty seconds is marked stale. Request cards show their\nassigned node and GPU pair, including staging and output-transfer phases.\nThis endpoint is read-only and requires the same authenticated session.\n''';p.write_text(s)
