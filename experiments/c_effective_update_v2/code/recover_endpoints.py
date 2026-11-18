"""Bounded exact inversion of saved FP32 endpoint differences; no model execution."""
import itertools, json, math, time
from pathlib import Path
import numpy as np
import torch
import runtime as rt
OLD=rt.ROOT/'experiments/c_effective_update_v1'
IDS=[5,6,8,9,10,11,12,13,15]
OUT=rt.E/'replay'
MAX_COMBINATIONS=2_000_000
MAX_SECONDS=45

def bits(x):return int(np.float32(x).view(np.uint32))
def order(x):
 b=bits(x)
 return (0xffffffff-b) if b&0x80000000 else b+0x80000000

def from_order(n):
 b=0xffffffff-n if n<0x80000000 else n-0x80000000
 return np.array(b,dtype=np.uint32).view(np.float32).item()

def candidates(x,d):
 # FP32 subtraction rounding bin, translated by the exactly represented x.
 # For these saved small, finite parameters, FP64 exactly represents midpoints.
 dn=np.nextafter(d,np.float32(-np.inf));up=np.nextafter(d,np.float32(np.inf))
 lo=float(x)+(float(dn)+float(d))/2
 hi=float(x)+(float(d)+float(up))/2
 first=np.float32(lo);last=np.float32(hi)
 if float(first)<lo:first=np.nextafter(first,np.float32(np.inf))
 if float(last)>hi:last=np.nextafter(last,np.float32(-np.inf))
 a,b=order(first),order(last)
 # End-point tie behavior is handled by testing actual RN-even subtraction.
 while a<=b and bits(np.float32(from_order(a))-x)!=bits(d):a+=1
 while b>=a and bits(np.float32(from_order(b))-x)!=bits(d):b-=1
 assert a<=b,(float(x),float(d),lo,hi)
 return a,b

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cp=torch.load(rt.INIT_CP,map_location='cpu',weights_only=True)
 state={k:v.clone() for k,v in cp['model_state_dict'].items()}
 result={'status':'RUNNING','method':'Enumerate FP32 y satisfying RN-even(y-x)==stored delta; verify whole old model SHA-256','new_optimizer_steps':0,'model_forward_calls':0,'gradient_calls':0,'max_combinations_per_endpoint':MAX_COMBINATIONS,'max_seconds_per_endpoint':MAX_SECONDS,'checkpoint_sha256':rt.sha(rt.INIT_CP),'endpoints':[]}
 vectors={}
 for pair_id in IDS:
  old=rt.read_json(OLD/'pairs'/f'pair_{pair_id:02d}.json')
  raw=np.load(OLD/'pairs'/f'pair_{pair_id:02d}.npz')
  names=old['parameter_names_in_vector_order'];x=raw['initial_parameters']
  assert np.array_equal(x,np.concatenate([state[k].reshape(-1).numpy() for k in names]))
  def h(y):
   off=0;cur=state.copy()
   for n in names:
    cur[n]=torch.from_numpy(y[off:off+state[n].numel()].reshape(tuple(state[n].shape)))
    off+=state[n].numel()
   return rt.tensor_hash(cur)
  for condition in ['full_F','no_keep','no_track']:
   d=raw['delta_'+condition];y=x+d;expected=old['conditions'][condition]['model_after_sha256']
   rec={'pair_id':pair_id,'condition':condition,'expected_model_sha256':expected,'naive_hash':h(y),'raw_path':str(OLD/'pairs'/f'pair_{pair_id:02d}.npz'),'raw_sha256':rt.sha(OLD/'pairs'/f'pair_{pair_id:02d}.npz')}
   if rec['naive_hash']==expected:
    vectors[f'pair_{pair_id:02d}_{condition}']=y
    rec.update(status='EXACT_NAIVE',ambiguous_coordinates=[],combinations=1,attempted=1)
   else:
    ambiguous=[];prod=1
    for i,(xx,dd) in enumerate(zip(x,d)):
     if xx==0 or dd==0:continue
     a,b=candidates(xx,dd)
     if b>a:
      prod*=b-a+1
      ambiguous.append({'index':i,'initial':float(xx),'delta':float(dd),'lower':from_order(a),'upper':from_order(b),'ordered_lower':a,'ordered_upper':b,'count':b-a+1})
     else:y[i]=from_order(a)
    rec.update(ambiguous_coordinates=ambiguous,combinations=prod,attempted=0)
    print(json.dumps({k:rec[k] for k in ('pair_id','condition','combinations','ambiguous_coordinates')}),flush=True)
    if prod>MAX_COMBINATIONS:rec['status']='BOUND_EXCEEDED'
    else:
     started=time.perf_counter()
     for values in itertools.product(*[range(a['ordered_lower'],a['ordered_upper']+1) for a in ambiguous]):
      for a,value in zip(ambiguous,values):y[a['index']]=from_order(value)
      rec['attempted']+=1
      if h(y)==expected:
       vectors[f'pair_{pair_id:02d}_{condition}']=y.copy();rec['status']='EXACT_RECOVERED';break
      if time.perf_counter()-started>MAX_SECONDS:rec['status']='TIME_BOUND_EXCEEDED';break
     else:rec['status']='NO_HASH_MATCH'
     rec['seconds']=time.perf_counter()-started
   result['endpoints'].append(rec)
   rt.save_json(OUT/'RECONSTRUCTION.json',result)
   print({k:rec[k] for k in ('pair_id','condition','status','combinations','attempted')},flush=True)
 result['exact_count']=len(vectors);result['total_count']=len(result['endpoints']);result['status']='EXACT' if len(vectors)==len(result['endpoints']) else 'INCOMPLETE'
 np.savez_compressed(OUT/'endpoints.npz',**vectors)
 result['endpoints_npz_sha256']=rt.sha(OUT/'endpoints.npz')
 rt.save_json(OUT/'RECONSTRUCTION.json',result)
 print({'status':result['status'],'exact_count':len(vectors),'total_count':len(result['endpoints'])},flush=True)
if __name__=='__main__':main()
