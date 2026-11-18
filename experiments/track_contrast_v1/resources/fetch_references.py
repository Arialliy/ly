import concurrent.futures,datetime,hashlib,json,time,urllib.request
from pathlib import Path
E=Path(__file__).resolve().parent
SPECS=[('SlotContrast','martius-lab/slotcontrast','slotcontrast/losses.py'),('S2MVP','UESTC-nnLab/S2MVP','mmdet/models/detectors/single_stage.py'),('TrackMAE','rvandeghen/TrackMAE','engine_for_pretraining_cotracker.py'),('DeepPro','TinaLRJ/DeepPro','networks/models/DeepPro-Plus.py')]
def fetch(url):
 for attempt in range(12):
  try:
   actual_url = url if attempt < 6 else ('https://ghproxy.net/'+url if 'raw.githubusercontent.com' in url else url)
   return urllib.request.urlopen(actual_url,timeout=15).read()
  except Exception as exc:last=exc
 raise last
def work(spec):
 name,repo,path=spec;out={'name':name,'repository':'https://github.com/'+repo,'requested_raw_url':'https://raw.githubusercontent.com/'+repo+'/main/'+path}
 commit=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/main'));sha=commit['sha'];out['commit']=sha;out['commit_url']='https://github.com/'+repo+'/commit/'+sha
 src=fetch('https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+path);dest=E/(name+'_source.py');dest.write_bytes(src)
 tree=json.loads(fetch('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1'))
 blob=next(x['sha'] for x in tree['tree'] if x['path']==path);actual=hashlib.sha1(b'blob '+str(len(src)).encode()+b'\0'+src).hexdigest();assert actual==blob
 out.update(local_source=dest.name,source_path=path,sha256=hashlib.sha256(src).hexdigest(),git_blob_sha1=actual,pinned_blob_verified=True)
 for filename in ['README.md','LICENSE']:
  node=next((x for x in tree['tree'] if x['path']==filename),None)
  if node:
   b=fetch('https://raw.githubusercontent.com/'+repo+'/'+sha+'/'+filename);p=E/(name+'_'+filename);p.write_bytes(b);out[filename]={'local_file':p.name,'sha256':hashlib.sha256(b).hexdigest()}
 (E/(name+'_REFERENCE.json')).write_text(json.dumps(out,indent=2)+'\n')
 print(name,sha,len(src),flush=True);return out
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for r in ex.map(work,SPECS):results.append(r)
(E/'REFERENCES.json').write_text(json.dumps({'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'references':results},indent=2)+'\n')
