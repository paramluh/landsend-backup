#!/usr/bin/env python3
"""Restore matched dataset contents from pinned HF sources or an extracted node-zero backup.

Extract this node's unique datasets archives first. This tool fills the files
which were safely deduplicated, then verifies every regular file in the report.
"""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path

def digest(path,algorithm):
 h=hashlib.sha256() if algorithm=='sha256' else hashlib.sha1()
 if algorithm=='git_blob_sha1':h.update(b'blob '+str(path.stat().st_size).encode()+b'\0')
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()

def child(root,relative):
 relative=Path(relative)
 if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe dataset-relative path')
 return root/relative

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--report',type=Path,required=True)
 ap.add_argument('--destination',type=Path,required=True,help='Destination dataset directory, e.g. restored/home/param/datasets')
 ap.add_argument('--node0-datasets',type=Path,help='Node-zero dataset directory after extracting its dataset archives')
 ap.add_argument('--aliases',type=Path,help='Optional cross-node-aliases.json')
 ap.add_argument('--other-node-datasets',action='append',default=[],metavar='HOST=PATH')
 ap.add_argument('--overwrite',action='store_true')
 args=ap.parse_args();report=json.loads(args.report.read_text())
 if report.get('errors'):raise RuntimeError('Verification report contains unresolved errors')
 others={k:Path(v) for k,v in (x.split('=',1) for x in args.other_node_datasets)}
 others.setdefault(report['host'],args.destination)
 aliases={r['relative']:r for r in json.loads(args.aliases.read_text())} if args.aliases else {}
 args.destination.mkdir(parents=True,exist_ok=True);done=0
 for r in [*report['matched'],*report['upload_required']]:
  dst=child(args.destination,r['relative'])
  if dst.is_file() and dst.stat().st_size==r['bytes'] and digest(dst,r['algorithm'])==r['digest']:
   done+=1;continue
  if dst.exists() and not args.overwrite:raise FileExistsError('Existing file differs; use --overwrite: '+str(dst))
  storage=r.get('storage')
  if storage and storage['kind']=='original_hf':
   from huggingface_hub import hf_hub_download
   source=Path(hf_hub_download(repo_id=storage['repo'],repo_type='dataset',revision=storage['revision'],filename=storage['path']))
  elif storage and storage['kind']=='node0_dataset_archive':
   if args.node0_datasets is None:raise ValueError('--node0-datasets is required for deduplicated derivatives')
   source=child(args.node0_datasets,r['baseline_relative'])
  elif r['relative'] in aliases:
   alias=aliases[r['relative']]
   if alias['owner_host'] not in others:raise ValueError('Missing --other-node-datasets '+alias['owner_host']+'=PATH')
   source=child(others[alias['owner_host']],alias['owner_relative'])
  else:raise FileNotFoundError('Extract this node\'s unique dataset archive first: '+str(dst))
  if source.stat().st_size!=r['bytes'] or digest(source,r['algorithm'])!=r['digest']:
   raise RuntimeError('Source content verification failed: '+str(source))
  dst.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=dst.parent,delete=False) as f:
   temporary=Path(f.name)
   with source.open('rb') as src:shutil.copyfileobj(src,f,4*1024**2)
  os.chmod(temporary,r['mode']);os.utime(temporary,ns=(r['mtime_ns'],r['mtime_ns']));os.replace(temporary,dst)
  done+=1
 links=0;root=args.destination.resolve();original=Path('/home/param/datasets')
 for row in report.get('symlinks',[]):
  dst=child(args.destination,Path(row['path']).relative_to(original));target=row['target']
  if Path(target).is_absolute():target=str(child(args.destination,Path(target).relative_to(original)))
  resolved=(dst.parent/target).resolve()
  if resolved!=root and root not in resolved.parents:raise ValueError('Dataset symlink escapes destination: '+str(dst))
  if dst.is_symlink() and os.readlink(dst)==target:links+=1;continue
  if os.path.lexists(dst):
   if not args.overwrite:raise FileExistsError('Existing symlink destination differs: '+str(dst))
   dst.unlink()
  dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(target);links+=1
 print(json.dumps({'verified_regular_files':done,'restored_symlinks':links}))

if __name__=='__main__':main()
