"""One frozen linear readout per P0/P1/P2, CPU only, no detector adaptation."""
import argparse,collections,csv,hashlib,json,math
from pathlib import Path


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare(features_manifest,proposal):
 import numpy as np
 manifest=json.loads(features_manifest.read_text());inputs={str(features_manifest):sha(features_manifest),str(proposal):sha(proposal)}
 assert manifest['status']=='COMPLETE'
 if 'sample_proposal_sha256' in manifest:assert manifest['sample_proposal_sha256']==sha(proposal)
 samples=json.loads(proposal.read_text())['samples'];features={}
 for entry in manifest['files']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];inputs[str(path)]=sha(path)
  with np.load(path,allow_pickle=False) as z:
   n=len(z['sample_index'])
   assert all(z[k].shape==(n,32) and np.isfinite(z[k]).all() for k in ['q','r1','r2'])
   assert np.isfinite(z['query_max_logit']).all()
   for i in range(n):
    key=int(z['sample_index'][i]),int(z['candidate_index'][i]);assert key not in features
    features[key]={k:z[k][i].astype(np.float64) for k in ['q','r1','r2']}
    features[key]['qmax']=float(z['query_max_logit'][i])
 rows=[];selected=[]
 for index,sample in enumerate(samples):
  assert (index,-1) in features
  candidates=range(len(sample['bg_candidates']));assert all((index,j) in features for j in candidates)
  chosen=max(candidates,key=lambda j:(features[index,j]['qmax'],-j))
  selected.append(dict(sample_index=index,pair_id=sample['pair_id'],negative_candidate_index=chosen,negative_query_max_logit=features[index,chosen]['qmax']))
  for label,candidate in [(1,-1),(0,chosen)]:
   row=dict(sample_index=index,pair_id=sample['pair_id'],label=label,candidate_index=candidate)
   row.update({k:sample[k] for k in ['split','sequence','track_id','frame','source_frame','reference_frame','start','annotation_id','low_all_05','extreme_all_001','both_fixed_success']})
   row.update(features[index,candidate]);rows.append(row)
 assert len(rows)==2*len(samples)
 return rows,selected,inputs


def weights(rows):
 """Supported source -> track -> target frame -> reference edge -> class equal."""
 import numpy as np
 seqs={r['sequence'] for r in rows};tracks=collections.defaultdict(set);frames=collections.defaultdict(set);edges=collections.defaultdict(set)
 for r in rows:
  s,t,f=r['sequence'],r['track_id'],r['frame'];tracks[s].add(t);frames[s,t].add(f);edges[s,t,f].add(r['sample_index'])
 w=np.array([1/(len(seqs)*len(tracks[r['sequence']])*len(frames[r['sequence'],r['track_id']])*len(edges[r['sequence'],r['track_id'],r['frame']])*2) for r in rows],np.float64)
 assert abs(w.sum()-1)<1e-10
 return w


def fit(x,y,weight):
 import numpy as np
 import torch
 torch.set_num_threads(1);torch.manual_seed(46);torch.use_deterministic_algorithms(True)
 X=torch.as_tensor(x,dtype=torch.float64);Y=torch.as_tensor(y,dtype=torch.float64);W=torch.as_tensor(weight,dtype=torch.float64)
 coefficient=torch.nn.Parameter(torch.zeros(64,dtype=torch.float64));intercept=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
 optimizer=torch.optim.LBFGS([coefficient,intercept],lr=1,max_iter=200,max_eval=250,
   tolerance_grad=1e-9,tolerance_change=1e-12,history_size=100,line_search_fn='strong_wolfe')
 calls=[]
 def closure():
  optimizer.zero_grad();logit=X@coefficient+intercept
  loss=(W*torch.nn.functional.binary_cross_entropy_with_logits(logit,Y,reduction='none')).sum()+.5*.01*coefficient.square().sum()
  loss.backward();calls.append(float(loss.detach()));return loss
 optimizer.step(closure)
 optimizer.zero_grad();logit=X@coefficient+intercept;loss=(W*torch.nn.functional.binary_cross_entropy_with_logits(logit,Y,reduction='none')).sum()+.5*.01*coefficient.square().sum();loss.backward()
 grad=torch.cat([coefficient.grad,intercept.grad.reshape(1)]);state=optimizer.state[coefficient]
 result=dict(weight=coefficient.detach().numpy().tolist(),intercept=float(intercept.detach()),
  final_objective=float(loss.detach()),gradient_inf_norm=float(grad.abs().max()),optimizer_iterations=int(state.get('n_iter',0)),function_evaluations=int(state.get('func_evals',0)),
  stopped_before_iteration_cap=int(state.get('n_iter',0))<200,gradient_tolerance_satisfied=bool(grad.abs().max()<=1e-9),
  finite=bool(torch.isfinite(grad).all() and torch.isfinite(loss)),train_matrix_rank=int(np.linalg.matrix_rank(x)),
  coefficient_parameters=64,intercept_parameters=1,effective_rank_limit_note='P0 reference block is fixed zero, rank at most32; P1/P2 may have higher effective rank. Nominal parameter equality does not imply equal capacity.',
  solver='CPU float64 deterministic LBFGS strong_wolfe, max_iter200, max_eval250, tolerance_grad1e-9, tolerance_change1e-12',lambda_l2=.01,intercept_penalized=False)
 assert result['finite'];return result


def metric(rows,logits):
 import numpy as np
 labels=np.array([r['label'] for r in rows]);p=logits[labels==1];n=logits[labels==0]
 auroc=float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
 pairs=collections.defaultdict(dict)
 for r,z in zip(rows,logits):pairs[r['sample_index']][r['label']]=float(z)
 margins=[d[1]-d[0] for d in pairs.values()];assert all(len(d)==2 for d in pairs.values())
 return dict(rows=len(rows),pairs=len(pairs),target_frames=len({(r['sequence'],r['annotation_id']) for r in rows}),
  tracks=len({(r['sequence'],r['track_id']) for r in rows}),sources=len({r['sequence'] for r in rows}),AUROC=auroc,
  probe_BCE=float(np.mean(np.logaddexp(0,logits)-labels*logits)),probe_accuracy_at_logit0=float(np.mean((logits>=0)==labels)),
  paired_positive_above_negative_fraction=sum((v>0)+.5*(v==0) for v in margins)/len(margins),
  paired_margin_mean=float(np.mean(margins)),paired_margin_median=float(np.median(margins)))


def main():
 import numpy as np
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proposal',type=Path,required=True);parser.add_argument('--features-manifest',type=Path,required=True);parser.add_argument('--authorization',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 auth=json.loads(args.authorization.read_text())
 assert auth['probe_fit_authorized'] is True
 assert auth['sample_proposal_sha256']==sha(args.proposal)
 assert auth['probe_readout_sha256']==sha(__file__)
 if args.output.exists() and any(args.output.iterdir()):raise FileExistsError(args.output)
 args.output.mkdir(parents=True,exist_ok=True)
 rows,selected,sources=prepare(args.features_manifest,args.proposal);write_csv(args.output/'HARD_NEGATIVE_SELECTION.csv',selected)
 train=np.array([r['split']=='train' for r in rows]);assert train.any() and (~train).any();train_rows=[r for r in rows if r['split']=='train'];weight=weights(train_rows)
 q=np.stack([r['q'] for r in rows]);r1=np.stack([r['r1'] for r in rows]);r2=np.stack([r['r2'] for r in rows]);y=np.array([r['label'] for r in rows],np.float64)
 mean=(weight[:,None]*q[train]).sum(0);std=np.sqrt((weight[:,None]*(q[train]-mean)**2).sum(0));std=np.maximum(std,1e-6)
 normalized_q=(q-mean)/std
 inputs={'P0':np.concatenate([normalized_q,np.zeros_like(normalized_q)],1),
  'P1':np.concatenate([normalized_q,(r1-mean)/std],1),'P2':np.concatenate([normalized_q,(r2-mean)/std],1)}
 models={};scores={};summary=[];tracks=[];predictions=[]
 for name,x in inputs.items():
  models[name]=fit(x[train],y[train],weight);scores[name]=x@np.array(models[name]['weight'])+models[name]['intercept']
  for split in ['train','dev']:
   for source in ['pooled']+sorted({r['sequence'] for r in rows if r['split']==split}):
    mask=np.array([r['split']==split and (source=='pooled' or r['sequence']==source) for r in rows]);rr=[r for r,m in zip(rows,mask) if m]
    summary.append(dict(condition=name,split=split,group=source,**metric(rr,scores[name][mask])))
   for source,track in sorted({(r['sequence'],r['track_id']) for r in rows if r['split']==split}):
    mask=np.array([r['split']==split and r['sequence']==source and r['track_id']==track for r in rows]);rr=[r for r,m in zip(rows,mask) if m]
    tracks.append(dict(condition=name,split=split,sequence=source,track_id=track,**metric(rr,scores[name][mask])))
 for i,r in enumerate(rows):
  row={k:v for k,v in r.items() if k not in ['q','r1','r2','qmax']}
  row.update({name+'_logit':float(z[i]) for name,z in scores.items()});predictions.append(row)
 for summary_row in summary:
  ts=[r for r in tracks if r['condition']==summary_row['condition'] and r['split']==summary_row['split'] and (summary_row['group']=='pooled' or r['sequence']==summary_row['group'])]
  summary_row['track_equal_AUROC']=sum(r['AUROC'] for r in ts)/len(ts)
  summary_row['track_equal_paired_positive_above_negative_fraction']=sum(r['paired_positive_above_negative_fraction'] for r in ts)/len(ts)
 write_csv(args.output/'SUMMARY.csv',summary);write_csv(args.output/'TRACKS.csv',tracks);write_csv(args.output/'PREDICTIONS.csv',predictions)
 write_json(args.output/'MODELS.json',dict(models=models,normalizer=dict(train_query_mean=mean.tolist(),train_query_std=std.tolist(),reference_uses_same_normalizer=True),weighting='supported source, track, target frame, reference edge, class hierarchical equal',normalizer_uses_dev=False))
 sources[str(args.authorization)]=sha(args.authorization);sources[str(Path(__file__).resolve())]=sha(__file__)
 write_json(args.output/'COMPLETE.json',dict(status='COMPLETE',source_sha256=sources,seed=46,detector_updated=False,new_structure_trained=False,readout_conditions=['P0','P1','P2'],
  artifacts_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
  scope='Small geometric common-support oracle-region feature probe, not deployment detection or Pd/Fa/U. P0 lower effective rank; P1 must face P2 to support same-target reference specificity. No claim beyond actual supported frames/tracks/sources.',
  train_target_frames=len({(r['sequence'],r['annotation_id']) for r in train_rows}),dev_target_frames=len({(r['sequence'],r['annotation_id']) for r in rows if r['split']=='dev'})))
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
