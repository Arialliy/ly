"""V12 independent Current/History endpoints plus History same-weight control."""
import argparse
import fcntl
import time
import numpy as np
import torch
import runtime as rt

def helpers():
    old=rt.old()
    dr=rt.load_module('_m12_diag_runtime',rt.DIAG/'code/runtime.py')
    cohorts=rt.load_module('_m12_diag_cohorts',rt.DIAG/'code/cohorts.py')
    trace=rt.load_module('_m12_bounded_trace',rt.E/'code/trace.py')
    with old.import_aliases({'runtime':dr,'cohorts':cohorts,'trace':trace}):
        driver=rt.load_module('_m12_diag_driver',rt.DIAG/'code/run_diagnostic.py')
    freeze=rt.read_json(rt.DIAG/'PROTOCOL_FREEZE.json')
    assert dr.source_hashes()==freeze['source_sha256']
    return cohorts,trace,driver

def infer(condition,sequence,device='cuda'):
    prior=rt.old();cohorts,trace,driver=helpers();cohorts.verify()
    model,run,checkpoint=rt.load_final(condition,device)
    state_hash=prior.tensor_hash(model.state_dict());sources=rt.source_hashes()
    raw_path=rt.DATA/sequence/'raw_frames.npy';raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
    assert raw.shape==(160,512,640) and raw.dtype==np.uint8
    identity={'sequence':sequence,'source_condition':condition,'checkpoint_step':2176,'main_checkpoint_step':2176,
        'checkpoint_is_main':True,'checkpoint_sha256':rt.sha(checkpoint),'training_run_sha256':rt.sha(checkpoint.with_name('RUN.json')),
        'source_sha256':sources,'frozen_v1_source_sha256':prior.source_hashes(),
        'protocol_sha256':rt.sha(rt.E/'PROTOCOL_FREEZE.json'),'diagnostic_protocol_sha256':rt.sha(rt.DIAG/'PROTOCOL_FREEZE.json'),
        'cohort_receipt_sha256':rt.sha(cohorts.OUT/'COMPLETE.json'),'initial_model_state_sha256':state_hash,
        'reference_checkpoint_sha256':rt.sha(prior.CP),'normalization':[85.0799,47.4845],
        'official_starts':rt.PLANS['O'],'symmetric_starts':rt.PLANS['S'],'diagnostic_starts':rt.STARTS,
        'raw_sha256':rt.sha(raw_path),'annotations_sha256':rt.sha(raw_path.with_name('annotations.json')),
        'all_original_frozen':True,'shared_frontend':True,'model_labels_access':False,
        'state_update_scope':'pixel','trained_mode':'current_only' if condition=='V12Current' else 'history',
        'current_trained_history_trace_is_auxiliary':condition=='V12Current'}
    destination=rt.E/'traces'/condition/sequence;destination.mkdir(parents=True,exist_ok=True)
    if (destination/'COMPLETE.json').exists():
        done=rt.read_json(destination/'COMPLETE.json')
        for k,v in identity.items():assert done[k]==v
        print({'condition':condition,'sequence':sequence,'status':'ALREADY_COMPLETE'},flush=True);return
    with (destination/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        regions=cohorts.read_csv(cohorts.OUT/'REGIONS.csv')
        modes={condition+'Current':('HistoryCurrent','current_only')}
        if condition=='V12History':modes[condition+'History']=('History','history')
        fused={name:np.zeros(raw.shape,np.float32) for name in modes};coverage=np.zeros(160,np.int16)
        receipts={name:dict(identity,condition=name,mode=spec[1],status='RUNNING',window_sha256={}) for name,spec in modes.items()}
        windows=[];began=time.perf_counter()
        for start in rt.STARTS:
            path=destination/f'window_{start:03d}.json'
            if path.exists():
                record=rt.read_json(path);assert record['identity']==identity
                for p,h in record['outputs'].items():assert rt.sha(p)==h
            else:
                x=torch.from_numpy((raw[start:start+40].astype(np.float32)-85.0799)/47.4845)[None,None].to(device)
                torch.cuda.reset_peak_memory_stats()
                result=trace.trace_window(model,x,cohorts.regions_for_window(sequence,start,regions),sequence=sequence,start=start)
                off=result['logits']['HistoryOff'].to(device).sigmoid()[0].cpu().numpy()
                check=driver.compare_reference(off,rt.Q/'stage1_goal/predictions/official'/sequence,start,cross_runtime_release=True)
                outputs={}
                for name,(key,_) in modes.items():
                    probability=result['logits'][key].to(device).sigmoid()[0].cpu().numpy()
                    p=rt.E/'predictions'/name/'step_2176'/sequence/'windows'/f'window_{start:03d}.npz'
                    driver.save_npz(p,probabilities=probability,start=start,end=start+40);outputs[str(p)]=rt.sha(p)
                for kind in ('frame','roi'):
                    p=destination/f'window_{start:03d}_{kind}.csv.gz';driver.write_rows(p,result[kind+'_rows']);outputs[str(p)]=rt.sha(p)
                record={'identity':identity,'window_start':start,'outputs':outputs,'checks':{'off_release':check},
                    'metadata':result['metadata'],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'roi_rows':len(result['roi_rows']),'frame_rows':len(result['frame_rows'])}
                rt.save(path,record)
                print({'condition':condition,'sequence':sequence,'start':start,'seconds':result['metadata']['elapsed_seconds']},flush=True)
                del x,result,off
            windows.append(record)
            for name in modes:
                p=rt.E/'predictions'/name/'step_2176'/sequence/'windows'/f'window_{start:03d}.npz'
                receipts[name]['window_sha256'][p.name]=rt.sha(p)
                if start in rt.PLANS['O']:
                    with np.load(p,allow_pickle=False) as saved:probability=saved['probabilities']
                    fused[name][start:start+40]=np.maximum(fused[name][start:start+40],probability)
            if start in rt.PLANS['O']:coverage[start:start+40]+=1
        assert coverage.sum()==200 and np.all(coverage>0)
        prior.assert_frozen(model,run['original_state_sha256']);assert prior.tensor_hash(model.state_dict())==state_hash
        assert sources==rt.source_hashes()
        for name in modes:
            folder=rt.E/'predictions'/name/'step_2176'/sequence;p=folder/'official_fused.npz'
            driver.save_npz(p,probabilities=fused[name],coverage=coverage,starts=np.array(rt.PLANS['O']))
            receipts[name].update(status='COMPLETE',fused_sha256=rt.sha(p),original_unchanged=True,
                all_model_state_unchanged_during_inference=True,final_model_state_sha256=state_hash)
            rt.save(folder/'COMPLETE.json',receipts[name])
        rt.save(destination/'COMPLETE.json',dict(identity,status='COMPLETE',windows=windows,
            final_model_state_sha256=state_hash,all_model_state_unchanged_during_inference=True,elapsed_seconds=time.perf_counter()-began))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=rt.CONDITIONS)
    p.add_argument('--sequence',choices=rt.SEQS);p.add_argument('--device',default='cuda');a=p.parse_args()
    for seq in [a.sequence] if a.sequence else rt.SEQS:infer(a.condition,seq,a.device)
