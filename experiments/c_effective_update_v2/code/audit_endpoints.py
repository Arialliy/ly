"""Independent loaded-file endpoint check: hashes and saved difference bytes."""
import numpy as np
import torch
import runtime as rt
OLD=rt.ROOT/'experiments/c_effective_update_v1'
OUT=rt.E/'replay'
cp=torch.load(rt.INIT_CP,map_location='cpu',weights_only=True)
state=cp['model_state_dict']
reconstruction=rt.read_json(OUT/'RECONSTRUCTION.json')
assert reconstruction['status']=='EXACT'
assert rt.sha(OUT/'endpoints.npz')==reconstruction['endpoints_npz_sha256']
records=[]
with np.load(OUT/'endpoints.npz',allow_pickle=False) as endpoints:
 for rec in reconstruction['endpoints']:
  i,c=rec['pair_id'],rec['condition']
  old=rt.read_json(OLD/'pairs'/f'pair_{i:02d}.json')
  assert rt.sha(rec['raw_path'])==rec['raw_sha256']
  with np.load(rec['raw_path'],allow_pickle=False) as raw:
   x=raw['initial_parameters'];d=raw['delta_'+c]
   y=endpoints[f'pair_{i:02d}_{c}']
   assert y.dtype==np.float32 and y.shape==x.shape and np.isfinite(y).all()
   actual=np.subtract(y,x,dtype=np.float32)
   assert actual.tobytes()==d.tobytes(),(i,c,'delta mismatch')
   current=state.copy();offset=0
   for n in old['parameter_names_in_vector_order']:
    count=state[n].numel()
    current[n]=torch.from_numpy(y[offset:offset+count].reshape(tuple(state[n].shape)))
    offset+=count
   assert offset==y.size
   digest=rt.tensor_hash(current)
   assert digest==old['conditions'][c]['model_after_sha256'],(i,c,'model hash mismatch')
   naive=np.add(x,d,dtype=np.float32)
   records.append({'pair_id':i,'condition':c,'saved_difference_bytes_exact':True,
      'whole_model_sha256_exact':True,'whole_model_sha256':digest,
      'noninvertible_coordinates':int(np.count_nonzero(y.view(np.uint32)!=naive.view(np.uint32))),
      'maximum_absolute_naive_error':float(np.max(np.abs(y.astype(np.float64)-naive.astype(np.float64))))})
assert len(records)==27
report={'status':'PASS','checked_endpoints':len(records),'new_optimizer_steps':0,'model_forward_calls':0,'gradient_calls':0,'checks':records,'endpoints_npz_sha256':rt.sha(OUT/'endpoints.npz'),'reconstruction_sha256':rt.sha(OUT/'RECONSTRUCTION.json'),'audit_script_sha256':rt.sha(__file__)}
rt.save_json(OUT/'ENDPOINT_AUDIT.json',report)
print({'status':report['status'],'checked_endpoints':len(records),'max_naive_abs_error':max(r['maximum_absolute_naive_error'] for r in records)},flush=True)
