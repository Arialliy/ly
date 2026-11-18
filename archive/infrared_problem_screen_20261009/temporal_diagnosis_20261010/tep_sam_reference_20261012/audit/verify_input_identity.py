from pathlib import Path
import json,hashlib,csv
N=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
P=N/'PROTOCOL.json';protocol=json.loads(P.read_text())
assert sha(P)=='9c73da144a4a4743c2cc36b6c57a33298cacf5b15468a03e0214f56e18b73374'
checks={}
for p,h in protocol['source_sha256'].items():
 assert sha(p)==h,p
 checks[p]=h
native=json.loads((N/'native_check/COMPLETE.json').read_text())
assert native['status']=='PASS_NATIVE_RUNNING_CHECK_ONLY' and native['metric_export_equality_frames']==100
assert json.loads((N/'runtime/DOC_FOLLOW_GPU3_RESULT.json').read_text())['exit_code']==0
rows=list(csv.DictReader((N/'TIME_ACCESS.csv').open()))
records=[]
for seq in protocol['data']['sequences']:
 d=N/'predictions'/seq;c=json.loads((d/'COMPLETE.json').read_text())
 assert c['status']=='COMPLETE' and c['sequence']==seq
 assert c['checkpoint_sha256']==protocol['author']['checkpoint_sha256']
 assert c['repository_commit']==protocol['author']['commit']
 assert Path(c['protocol_path']).resolve()==P.resolve() and c['protocol_sha256']==sha(P)
 assert c['forward_calls']==160 and c['seed']==46 and c['state_unchanged'] and c['parameters']==95576030
 assert c['source_before']['hashes_verified'] and c['source_after']['hashes_verified']
 for file,key in [('probabilities.npz','probabilities_sha256'),('auxiliary_probabilities.npz','auxiliary_probabilities_sha256')]:
  assert sha(d/file)==c[key]
  checks[str(d/file)]=c[key]
 a=json.loads((d/'FRAME_ACCESS.json').read_text())
 assert a['frame_indices']==list(range(160)) and a['no_annotation_prompts']
 assert a['context_local_indices']==[[min(159,max(0,t+o)) for o in range(-3,4)] for t in range(160)]
 data=N.parent/'stage1_goal/data'/seq
 assert c['raw_sha256']==sha(data/'raw_frames.npy') and c['annotations_sha256']==sha(data/'annotations.json')
 records.append({'sequence':seq,'receipt_sha256':sha(d/'COMPLETE.json'),'actual_identity_equals_frozen_protocol':True,'no_mutation':True,'frame_access_valid':True,'forward_calls':c['forward_calls']})
result={'status':'PASS','protocol_sha256':sha(P),'predictions':records,'sha256_verified':checks,'new_model_calls':0,'purpose':'Exact checkpoint/commit/protocol identity enforcement before scoring, beyond shape-only validation in scorer.'}
(N/'audit/INPUT_IDENTITY_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','sequences':records,'frozen_file_count':len(protocol['source_sha256'])}))
