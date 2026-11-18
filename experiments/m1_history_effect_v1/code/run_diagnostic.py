"""One shared-feature, same-History-weight diagnostic on the existing windows."""
import argparse
import csv
import fcntl
import gzip
import json
import time
import numpy as np
import torch
import runtime as rt
import cohorts
from trace import trace_window

MODES = {'HistoryCurrent': 'current_only', 'HistoryOff': 'off'}

def write_rows(path, rows):
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(temporary, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict,list)) else v for k,v in row.items()})
    temporary.replace(path)

def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream, **values)
    temporary.replace(path)

def compare_reference(probability, folder, start, *, cross_runtime_release=False):
    receipt = rt.read_json(folder / 'COMPLETE.json')
    path = folder / 'windows' / f'window_{start:03d}.npz'
    assert receipt['status'] == 'COMPLETE'
    assert rt.sha(path) == receipt['window_sha256'][path.name]
    with np.load(path, allow_pickle=False) as saved: reference = saved['probabilities']
    assert reference.shape == probability.shape
    difference = np.abs(probability-reference)
    record = {'reference_path': str(path), 'reference_sha256': rt.sha(path),
        'probability_absmax': float(difference.max()), 'probability_mean_abs': float(difference.mean(dtype=np.float64)),
        'threshold_05_pixel_differences': int(np.count_nonzero((probability >= .5) != (reference >= .5))),
        'exact_probability_equal': bool(np.array_equal(probability, reference))}
    record['absolute_tolerance'] = 2e-5 if cross_runtime_release else 0.
    record['cross_runtime_release'] = cross_runtime_release
    assert record['probability_absmax'] <= record['absolute_tolerance'], record
    assert record['threshold_05_pixel_differences'] == 0, record
    return record

def run(sequence, device='cuda'):
    prior = rt.old()
    assert rt.sha(rt.CHECKPOINT) == rt.CHECKPOINT_SHA
    cohorts.verify()
    model, training, checkpoint, run_path = rt.v1_infer().load_trained('History',2048,device)
    model_hash = prior.tensor_hash(model.state_dict())
    source = rt.source_hashes()
    raw_path = rt.DATA / sequence / 'raw_frames.npy'
    raw = np.load(raw_path, mmap_mode='r', allow_pickle=False)
    assert raw.shape == (160,512,640) and raw.dtype == np.uint8
    identity = {'sequence': sequence, 'checkpoint_step': 2048, 'main_checkpoint_step': 2048,
        'checkpoint_is_main': True, 'checkpoint_sha256': rt.sha(checkpoint),
        'training_run_sha256': rt.sha(run_path), 'source_sha256': source,
        'frozen_v1_source_sha256': prior.source_hashes(), 'protocol_sha256': rt.sha(rt.E/'PROTOCOL.md'),
        'cohort_receipt_sha256': rt.sha(cohorts.OUT/'COMPLETE.json'),
        'reference_checkpoint_sha256': rt.sha(prior.CP), 'initial_model_state_sha256': model_hash,
        'normalization': [85.0799,47.4845], 'official_starts': rt.PLANS['O'],
        'symmetric_starts': rt.PLANS['S'], 'diagnostic_starts': rt.STARTS,
        'raw_sha256': rt.sha(raw_path), 'annotations_sha256': rt.sha(raw_path.with_name('annotations.json')),
        'all_original_frozen': True, 'shared_frontend': True, 'model_labels_access': False,
        'source_condition': 'History', 'precision': 'FP32_no_AMP_no_TF32'}
    destination = rt.E/'traces'/sequence
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination/'COMPLETE.json'
    if receipt.exists():
        old = rt.read_json(receipt)
        for key,value in identity.items(): assert old[key] == value, key
        print({'sequence':sequence,'status':'ALREADY_COMPLETE'},flush=True); return
    with (destination/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        regions = cohorts.read_csv(cohorts.OUT/'REGIONS.csv')
        fused = {mode:np.zeros(raw.shape,np.float32) for mode in MODES}
        coverage = np.zeros(160,np.int16)
        receipts = {mode:dict(identity, condition=mode, mode=meaning, status='RUNNING', window_sha256={}) for mode,meaning in MODES.items()}
        windows = []; began = time.perf_counter()
        for start in rt.STARTS:
            done = destination/f'window_{start:03d}.json'
            if done.exists():
                record = rt.read_json(done)
                assert record['identity'] == identity
                for path,digest in record['outputs'].items(): assert rt.sha(path) == digest
            else:
                x = torch.from_numpy((raw[start:start+40].astype(np.float32)-85.0799)/47.4845)[None,None].to(device)
                torch.cuda.reset_peak_memory_stats()
                result = trace_window(model,x,cohorts.regions_for_window(sequence,start,regions),sequence=sequence,start=start)
                outputs = {}
                checks = {}
                for mode,logits in result['logits'].items():
                    # Keep the original CUDA sigmoid, including FP32 rounding.
                    probability = logits.to(device).sigmoid()[0].cpu().numpy()
                    if mode == 'History':
                        checks['History_cached'] = compare_reference(probability,rt.V1/'predictions/History/step_2048'/sequence,start)
                    else:
                        if mode == 'HistoryOff':
                            checks['off_release'] = compare_reference(probability,rt.Q/'stage1_goal/predictions/official'/sequence,start,cross_runtime_release=True)
                        path = rt.E/'predictions'/mode/'step_2048'/sequence/'windows'/f'window_{start:03d}.npz'
                        save_npz(path,probabilities=probability,start=start,end=start+40)
                        outputs[str(path)] = rt.sha(path)
                for label in ('frame','roi'):
                    path = destination/f'window_{start:03d}_{label}.csv.gz'
                    write_rows(path,result[label+'_rows']); outputs[str(path)] = rt.sha(path)
                record = {'identity':identity,'outputs':outputs,'checks':checks,'window_start':start,
                    'metadata':result['metadata'],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'frame_rows':len(result['frame_rows']),'roi_rows':len(result['roi_rows'])}
                rt.save(done,record)
                print({'sequence':sequence,'start':start,'seconds':result['metadata']['elapsed_seconds'],
                       'checks':checks,'rows':record['roi_rows']},flush=True)
                del result,x
            windows.append(record)
            for mode in MODES:
                path = rt.E/'predictions'/mode/'step_2048'/sequence/'windows'/f'window_{start:03d}.npz'
                receipts[mode]['window_sha256'][path.name] = rt.sha(path)
                if start in rt.PLANS['O']:
                    with np.load(path,allow_pickle=False) as saved: probability = saved['probabilities']
                    fused[mode][start:start+40] = np.maximum(fused[mode][start:start+40],probability)
            if start in rt.PLANS['O']: coverage[start:start+40] += 1
        assert np.all(coverage > 0) and coverage.sum() == 200
        prior.assert_frozen(model,training['original_state_sha256'])
        assert prior.tensor_hash(model.state_dict()) == model_hash
        assert source == rt.source_hashes()
        for mode in MODES:
            folder = rt.E/'predictions'/mode/'step_2048'/sequence
            path = folder/'official_fused.npz'
            save_npz(path,probabilities=fused[mode],coverage=coverage,starts=np.array(rt.PLANS['O']))
            receipts[mode].update(status='COMPLETE',fused_sha256=rt.sha(path),
                original_unchanged=True,all_model_state_unchanged_during_inference=True,final_model_state_sha256=model_hash)
            rt.save(folder/'COMPLETE.json',receipts[mode])
        rt.save(receipt,dict(identity,status='COMPLETE',windows=windows,
            final_model_state_sha256=model_hash,all_model_state_unchanged_during_inference=True,
            elapsed_seconds=time.perf_counter()-began))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--sequence',required=True,choices=rt.SEQS)
    parser.add_argument('--device',default='cuda'); args = parser.parse_args()
    run(args.sequence,args.device)
