"""Frozen local-projection prerequisite probe; no training or method performance."""
import argparse
import collections
import csv
import datetime
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEPENDENCY = ROOT / 'mechanism_followup/run_factorial_frozen.py'
spec = importlib.util.spec_from_file_location('frozen_local_probe_dependency', DEPENDENCY)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

def map_summary(result, prefix, logits, peaks):
    result[prefix + '_box_max_probability'] = float(logits.sigmoid().max())
    result[prefix + '_box_max_logit'] = float(logits.max())
    for name, (py, px) in peaks.items():
        result[prefix + '_' + name + '_peak_probability'] = float(logits[py, px].sigmoid())
        result[prefix + '_' + name + '_peak_logit'] = float(logits[py, px])

def run_pair(model, frames, row, device):
    import torch
    n, height, width = frames.shape
    x0, y0, x1, y1 = frozen.crop_indices(row, height, width)
    assert x1 > x0 and y1 > y0
    xa, ya, xb, yb = max(0,x0-16), max(0,y0-16), min(width,x1+16), min(height,y1+16)
    ys, xs = slice(y0-ya,y1-ya), slice(x0-xa,x1-xa)
    result = dict(row)
    result.update(crop_x0=x0,crop_y0=y0,crop_x1=x1,crop_y1=y1,input_crop_x0=xa,input_crop_y0=ya,input_crop_x1=xb,input_crop_y1=yb)
    originals, peaks = {}, {}
    for tag in ['a','b']:
        start, slot = row[tag+'_start'], row[tag+'_slot']
        assert start+slot == int(row['local_frame']) and 0 <= start <= n-40
        raw = np.asarray(frames[start:start+40,ya:yb,xa:xb],dtype=np.float32)
        normed = np.ascontiguousarray((raw-85.0799)/47.4845)
        assert np.isfinite(normed).all()
        input_tensor = torch.from_numpy(normed)[None,None].to(device)
        capture = {}
        def capture_features(module, inputs, output):
            capture['feature'] = output[...,ys,xs].clone()
        def capture_projection(module, inputs):
            capture['q_official'] = inputs[0][:,:,slot:slot+1,ys,xs].clone()
        h1 = model.layer1.register_forward_hook(capture_features)
        h2 = model.TPro.norm1.register_forward_pre_hook(capture_projection)
        try:
            output = model(input_tensor)
            logits = output[1][0,slot,ys,xs].clone()
            del output
        finally:
            h1.remove()
            h2.remove()
        del input_tensor
        p = logits.sigmoid()
        peaks[tag] = divmod(int(p.flatten().argmax()),x1-x0)
        result[tag+'_peak_y_index'] = y0+peaks[tag][0]
        result[tag+'_peak_x_index'] = x0+peaks[tag][1]
        result[tag+'_probability_peak_tie_count'] = int((p==p.max()).sum())
        replay_error = abs(float(p.max())-row[tag+'_saved_box_max'])
        result[tag+'_replay_abs_error'] = replay_error
        assert replay_error <= 1e-4, (frozen.key(row),tag,replay_error)
        for th in [.1,.5,.9]:
            result[tag+'_replay_threshold_disagreement_'+str(th).replace('.','_')] = int((float(p.max())>=th)!=(row[tag+'_saved_box_max']>=th))
        q = frozen.projection(model,capture['feature'],slot)
        result[tag+'_projection_vs_official_max_abs_error'] = float((q-capture['q_official']).abs().max())
        torch.testing.assert_close(q,capture['q_official'],atol=1e-4,rtol=1e-4)
        originals[tag] = dict(feature=capture['feature'],q=q,logits=logits)
    for tag in ['a','b']:
        slot = row[tag+'_slot']
        feature,q = originals[tag]['feature'],originals[tag]['q']
        weights,bias = frozen.row_weights(model,slot)
        bias = bias[None,:,None,None,None]
        combinations,names = [q.clone()],['full']
        for radius in [0,4]:
            indices = [s for s in range(40) if abs(s-slot)<=radius]
            far_indices = [s for s in range(40) if s not in indices]
            local = frozen.weighted_sum(weights[:,indices],feature[:,:,indices])+bias
            far = frozen.weighted_sum(weights[:,far_indices],feature[:,:,far_indices])
            error = float((local+far-q).abs().max())
            result[tag+'_r%d_partition_max_abs_error'%radius] = error
            torch.testing.assert_close(local+far,q,atol=1e-4,rtol=1e-4)
            result[tag+'_r%d_local_feature_frame_count'%radius] = len(indices)
            result[tag+'_r%d_local_first_slot'%radius] = indices[0]
            result[tag+'_r%d_local_last_slot'%radius] = indices[-1]
            result[tag+'_r%d_local_projection_l2'%radius] = float(local.square().sum().sqrt())
            result[tag+'_r%d_far_projection_l2'%radius] = float(far.square().sum().sqrt())
            for g,suffix in [(0.,'000'),(.5,'050')]:
                combinations.append(local+g*(q-local))
                names.append('r%d_g%s'%(radius,suffix))
        maps = frozen.phi(model,torch.cat(combinations,0))
        original_logits = originals[tag]['logits']
        torch.testing.assert_close(maps[0],original_logits,atol=1e-4,rtol=1e-4)
        error = float((maps[0].sigmoid()-original_logits.sigmoid()).abs().max())
        result[tag+'_backend_max_abs_probability_error'] = error
        result[tag+'_backend_max_abs_logit_error'] = float((maps[0]-original_logits).abs().max())
        assert error <= 1e-4
        map_summary(result,tag+'_original',original_logits,peaks)
        for name,logits in zip(names,maps):
            map_summary(result,tag+'_'+name,logits,peaks)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.csv').exists():
        raise FileExistsError(out/'results.csv')
    protocol_path=HERE/'PROTOCOL.json'
    expected_protocol=(HERE/'PROTOCOL.sha256').read_text().split()[0]
    assert expected_protocol==frozen.digest(protocol_path)
    assert frozen.digest(DEPENDENCY)=='fa2004b402add5a028f45af101c0604dd225b3e69168bc9f3acaef02bfebb238'
    protocol=json.loads(protocol_path.read_text())
    assert protocol['radii']==[0,4] and protocol['gates']==[0.,.5,1.]
    pairs,input_paths=frozen.select_pairs(ROOT)
    support_path=ROOT/'mechanism_followup/SUPPORT_PAIRED.csv'
    support={frozen.key(r):r for r in frozen.read_csv(support_path)}
    counts=collections.Counter()
    for row in pairs:
        prev=support[frozen.key(row)]
        strong = row['a_saved_box_max']<.1 and row['b_saved_box_max']>=.9
        row['in_strong_97']=int(strong)
        row['in_suppression_24']=int(strong and int(prev['a_suppressed_common_strong']))
        row['in_support_42']=int(strong and int(prev['b_requires_exclusive_strong']))
        row['in_stable_high_502']=int(row['a_saved_box_max']>=.5)
        row['in_stable_low_778']=int(row['b_saved_box_max']<.5)
        row['prior_group']=prev['group']
        for field in ['in_strong_97','in_suppression_24','in_support_42','in_stable_high_502','in_stable_low_778']:
            counts[field]+=row[field]
    assert list(counts.values())==[97,24,42,502,778],counts
    selected=pairs[:args.limit] if args.limit else pairs
    with (out/'selected_pairs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selected[0]))
        writer.writeheader();writer.writerows(selected)
    launches,raw_paths={},{}
    for sequence in sorted({row['sequence'] for row in selected}):
        lp=ROOT/'runs'/sequence/'LAUNCH_TILED.json'
        cmd=json.loads(lp.read_text())['cmd']
        launches[sequence]=cmd
        raw_paths[sequence]=Path(frozen.launch_value(cmd,'--input-npy'))
        input_paths.append(lp)
    first=next(iter(launches.values()))
    source=Path(frozen.launch_value(first,'--source'))
    ckpt=Path(frozen.launch_value(first,'--checkpoint'))
    for cmd in launches.values():
        assert frozen.launch_value(cmd,'--source')==str(source)
        assert frozen.launch_value(cmd,'--checkpoint')==str(ckpt)
    started=time.perf_counter()
    metadata=dict(status='READY',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  protocol_sha256=expected_protocol,script_sha256=frozen.digest(__file__),dependency_sha256=frozen.digest(DEPENDENCY),
                  input_sha256={str(p):frozen.digest(p) for p in input_paths+[support_path]},
                  raw_input_sha256={str(p):frozen.digest(p) for p in raw_paths.values()},
                  checkpoint_sha256=frozen.digest(ckpt),selected_count=len(selected),full_count=len(pairs),
                  stratum_counts=dict(counts),limit=args.limit,command=sys.argv)
    frozen.json_write(out/'metadata.json',metadata)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    random.seed(46);np.random.seed(46);torch.manual_seed(46);torch.cuda.manual_seed_all(46)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    model,model_path,epoch,radius=frozen.model_setup(source,ckpt,device)
    metadata.update(status='RUNNING',torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
                    checkpoint_epoch=epoch,model_source_sha256=frozen.digest(model_path),frontend_radius=radius)
    frozen.json_write(out/'metadata.json',metadata)
    arrays={seq:np.load(p,mmap_mode='r',allow_pickle=False) for seq,p in raw_paths.items()}
    maxima=collections.defaultdict(float)
    disagreements=collections.Counter()
    completed=0
    try:
        with torch.inference_mode(),(out/'results.csv').open('w',newline='') as stream:
            writer=None
            for row in selected:
                result=run_pair(model,arrays[row['sequence']],row,device)
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(result));writer.writeheader()
                writer.writerow(result);stream.flush()
                completed+=1
                for field,value in result.items():
                    if 'error' in field and isinstance(value,(int,float)):
                        maxima[field]=max(maxima[field],float(value))
                    if 'replay_threshold_disagreement' in field:
                        disagreements[field]+=int(value)
                if completed%50==0 or completed==len(selected):
                    progress=dict(completed=completed,total=len(selected),elapsed_seconds=time.perf_counter()-started,
                                  maximum_errors=dict(maxima),threshold_disagreements=dict(disagreements))
                    frozen.json_write(out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
    except BaseException as error:
        metadata.update(status='FAILED',completed=completed,error=repr(error),elapsed_seconds=time.perf_counter()-started)
        frozen.json_write(out/'metadata.json',metadata)
        raise
    assert frozen.digest(ckpt)==metadata['checkpoint_sha256']
    metadata.update(status='COMPLETE',completed=completed,elapsed_seconds=time.perf_counter()-started,
                    maximum_errors=dict(maxima),threshold_disagreements=dict(disagreements),
                    result_sha256=frozen.digest(out/'results.csv'),selection_sha256=frozen.digest(out/'selected_pairs.csv'),checkpoint_unchanged=True)
    frozen.json_write(out/'metadata.json',metadata)
    print(json.dumps({'status':'COMPLETE','completed':completed,'elapsed_seconds':metadata['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    main()
