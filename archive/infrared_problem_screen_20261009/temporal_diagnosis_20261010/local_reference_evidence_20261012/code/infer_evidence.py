#!/usr/bin/env python3
"""Final-only N0/N1/N2 eight-window inference with passive sparse telemetry."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import train_evidence as runner

STARTS = [0,24,36,60,72,96,108,120]
OFFICIAL = [0,36,72,108,120]
SYMMETRIC = [0,24,60,96,120]
CHANNELS = ['I','D1','mu1','s1','E1','m1','D4','mu4','s4','E4','m4']
COHORTS = runner.Q / 'low_response_information_20261011/COHORTS.csv'


class SparseRecorder:
    """Coarse centers of all frozen objects plus one old-score background peak.

    Annotations enter this recorder only. No label or coordinate enters forward.
    Each point is read from the tile that owns its output core; no second model
    forward or statistics computation is performed for telemetry.
    """
    def __init__(self, sequence, annotation_path, destination, base):
        with COHORTS.open() as f:
            self.objects = [row for row in csv.DictReader(f) if row['sequence']==sequence]
        annotations = json.loads(annotation_path.read_text())
        images = annotations['images']
        lookup = {item['id']: i for i,item in enumerate(images)}
        self.source_frames = [int(Path(item['file_name']).stem) for item in images]
        self.background = np.ones((160,512,640), dtype=bool)
        for ann in annotations['annotations']:
            if not base.valid_box(ann): continue
            x0,y0,x1,y1 = base.pixel_rect(ann['bbox'],512,640)
            self.background[lookup[ann['image_id']],y0:y1,x0:x1] = False
        for row in self.objects:
            row['query_x'] = max(0,min(639,math.floor(float(row['bbox_x'])+float(row['bbox_w'])/2)))
            row['query_y'] = max(0,min(511,math.floor(float(row['bbox_y'])+float(row['bbox_h'])/2)))
        self.sequence, self.destination = sequence,destination
        self.fields = ['sequence','window_start','kind','frame','source_frame','annotation_id','track_id',
                       'slot','y','x',*CHANNELS,'original_logit','delta_logit']
        self.stream = destination.open('w',newline='')
        self.writer = csv.DictWriter(self.stream,fieldnames=self.fields)
        self.writer.writeheader()
        self.seconds = 0.0; self.target_count=0; self.background_count=0
        self.best = {}

    def begin_window(self, start):
        self.best={}
        self.window_objects=[r for r in self.objects if start<=int(r['frame'])<start+40]

    def capture(self, model, start, y,y1,x,x1,ya,xa,device):
        if device.type=='cuda': torch.cuda.synchronize(device)
        stamp=time.perf_counter()
        vector=model.evidence.last_input[0]
        old=model.last_original_logits[0]
        delta=model.last_delta[0]
        rows=[r for r in self.window_objects if y<=r['query_y']<y1 and x<=r['query_x']<x1]
        if rows:
            ts=torch.tensor([int(r['frame'])-start for r in rows],device=device)
            ys=torch.tensor([r['query_y']-ya for r in rows],device=device)
            xs=torch.tensor([r['query_x']-xa for r in rows],device=device)
            points=torch.cat((vector[ts,:,ys,xs],old[ts,ys,xs,None],delta[ts,ys,xs,None]),dim=1).detach().cpu().tolist()
            for row,values in zip(rows,points):
                frame=int(row['frame'])
                self.writer.writerow({'sequence':self.sequence,'window_start':start,'kind':'target_coarse_box_center',
                    'frame':frame,'source_frame':row['source_frame'],'annotation_id':row['annotation_id'],
                    'track_id':row['track_id'],'slot':frame-start,'y':row['query_y'],'x':row['query_x'],
                    **dict(zip(CHANNELS+['original_logit','delta_logit'],values))})
                self.target_count+=1
        # Candidate maxima are selected solely from the frozen original output.
        # Tile maxima are merged by value and then the global flattened index.
        mask=torch.from_numpy(self.background[start:start+40,y:y1,x:x1]).to(device)
        core=old[:,y-ya:y1-ya,x-xa:x1-xa]
        scores,indices=core.masked_fill(~mask,-torch.inf).reshape(40,-1).max(dim=1)
        ts=torch.arange(40,device=device)
        ys=indices//(x1-x)+y-ya; xs=indices%(x1-x)+x-xa
        points=torch.cat((vector[ts,:,ys,xs],old[ts,ys,xs,None],delta[ts,ys,xs,None]),dim=1).detach().cpu().tolist()
        for slot,(score,idx,values) in enumerate(zip(scores.detach().cpu().tolist(),indices.detach().cpu().tolist(),points)):
            if not math.isfinite(score): continue
            gy=y+idx//(x1-x); gx=x+idx%(x1-x); flat=gy*640+gx
            prior=self.best.get(slot)
            if prior is None or score>prior[0] or (score==prior[0] and flat<prior[1]):
                self.best[slot]=(score,flat,gy,gx,values)
        if device.type=='cuda': torch.cuda.synchronize(device)
        self.seconds+=time.perf_counter()-stamp

    def finish_window(self,start):
        for slot,(_,_,y,x,values) in sorted(self.best.items()):
            self.writer.writerow({'sequence':self.sequence,'window_start':start,'kind':'original_global_hard_background',
                'frame':start+slot,'source_frame':self.source_frames[start+slot],
                'annotation_id':'','track_id':'','slot':slot,'y':y,'x':x,
                **dict(zip(CHANNELS+['original_logit','delta_logit'],values))})
            self.background_count+=1
        self.stream.flush()

    def close(self): self.stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','reference-checkpoint','checkpoint','dataset-root','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=runner.VARIANTS,required=True)
    p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'],required=True)
    p.add_argument('--verify-tiling',action='store_true')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.output.parent.name!=args.variant: raise ValueError('Output parent must equal variant')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Output must be new and empty')
    args.output.mkdir(parents=True,exist_ok=True); (args.output/'windows').mkdir()
    runner.deterministic_setup()
    base=runner.load_module('evidence_frozen_batch_helpers',runner.Q/'top1_background_control_20261011/code/train_pilot.py')
    inference=runner.load_module('evidence_frozen_tiling',runner.Q/'infer_with_windows.py')
    original,model_path=base.load_model(args.source,args.reference_checkpoint)
    original_hash=runner.tensor_hash(original.state_dict().items())
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    run=json.loads((args.checkpoint.parent/'RUN.json').read_text())
    snapshot=next(x for x in run['snapshots'] if x['step']==512)
    sources=runner.source_hashes(model_path)
    if (checkpoint['variant']!=args.variant or checkpoint['step']!=512 or run['status']!='complete'
        or run['variant']!=args.variant or run['mode']!=runner.VARIANTS[args.variant]
        or snapshot['sha256']!=runner.sha(args.checkpoint)
        or checkpoint['initial_checkpoint_sha256']!=runner.sha(args.reference_checkpoint)
        or checkpoint['plan_sha256']!=runner.EXPECTED['plan']
        or checkpoint['selection_sha256']!=runner.EXPECTED['selection']
        or checkpoint['label_config']!=runner.LABEL_CONFIG
        or checkpoint['effective_supervision_config']!=runner.EFFECTIVE_SUPERVISION
        or checkpoint['original_state_sha256']!=original_hash
        or run['original_initial_state_sha256']!=original_hash or run['original_final_state_sha256']!=original_hash
        or any(run[name]!=digest or checkpoint[name]!=digest for name,digest in sources.items())):
        raise AssertionError('Frozen training/checkpoint provenance mismatch')
    model=runner.wrap_model(original,args.variant)
    if checkpoint['model_config']!=model.config: raise AssertionError('Evidence configuration changed')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    if runner.tensor_hash(model.original.state_dict().items())!=original_hash:
        raise AssertionError('Checkpoint original differs from released model')
    device=torch.device(args.device); model=model.to(device).eval()
    raw_path=args.dataset_root/args.sequence/'raw_frames.npy'
    ann_path=args.dataset_root/args.sequence/'annotations.json'
    raw=np.load(raw_path,mmap_mode='r',allow_pickle=False)
    if raw.dtype!=np.uint8 or raw.shape!=(160,512,640): raise ValueError('Unexpected dev data')
    fused=np.zeros(raw.shape,dtype=np.float32); coverage=np.zeros(160,dtype=np.int32)
    halo,core=16,128
    recorder=SparseRecorder(args.sequence,ann_path,args.output/'EVIDENCE_SAMPLES.csv',base)
    metadata={
        'sequence':args.sequence,'condition':args.variant,'variant':args.variant,'training_variant':args.variant,
        'mode':runner.VARIANTS[args.variant],'diagnostic':'none','checkpoint':str(args.checkpoint),
        'checkpoint_sha256':runner.sha(args.checkpoint),'reference_checkpoint_sha256':runner.sha(args.reference_checkpoint),
        'model_source':str(model_path),**sources,'raw_sha256':runner.sha(raw_path),'annotations_sha256':runner.sha(ann_path),
        'script_sha256':runner.sha(__file__),'plan_sha256':runner.EXPECTED['plan'],'selection_sha256':runner.EXPECTED['selection'],
        'original_initial_state_sha256':original_hash,'original_final_state_sha256':original_hash,
        'bn_initial_sha256':run['bn_initial_sha256'],'bn_final_sha256':run['bn_final_sha256'],
        'normalization':[85.0799,47.4845],'seed':46,'precision':'FP32_no_AMP_no_TF32',
        'torch':str(torch.__version__),'cuda':str(torch.version.cuda),
        'official_starts':OFFICIAL,'symmetric_starts':SYMMETRIC,'diagnostic_starts':STARTS,
        'fusion':'pixelwise probability maximum over official starts only',
        'tile_core':core,'tile_halo':halo,'receptive_radius_tyx':[None,9,9],
        'halo_derivation':'max(original9,annulus7+two3x3conv2)=9; shared conservative halo16',
        'model_labels_access':False,'read_only_query_labels_access':True,
        'tiling_checks':[],'window_timing':[],'model_config':model.config,
        'label_config':runner.LABEL_CONFIG,'parameters':sum(v.numel() for v in model.parameters()),
        'trainable_parameters':sum(v.numel() for v in model.evidence.parameters()),
        'main_checkpoint_step':512,'checkpoint_step':512,'checkpoint_is_main':True,
        'peak_memory_scope':'model inference plus passive sparse telemetry and output copy',
        'telemetry_rule':{
            'objects':'all5200 frozen strict-small object-frames; coarse box center floor(x+w/2),floor(y+h/2) per covering candidate window',
            'background':'highest original-model logit outside all valid annotation boxes per frame/window; ties global flat index ascending',
            'input_values':'actual mode input, absent channels exactly zero; no extra forward/statistic construction',
            'cohort_sha256':runner.sha(COHORTS),'model_labels_access':False,
            'limitation':'coarse box center is not precise target centroid; read-only telemetry is not causal evidence'},
    }
    runner.save_json(args.output/'RUNNING.json',metadata)
    begin=time.perf_counter()
    try:
        with torch.inference_mode():
            if args.verify_tiling:
                # One preregistered crop checks actual nonzero learned correction.
                y,x=176,224
                probe=torch.from_numpy(np.ascontiguousarray((raw[:40,y:y+96,x:x+128].astype(np.float32)-85.0799)/47.4845))[None,None].to(device)
                direct=model(probe)[1].cpu()
                correction_max=float(model.last_delta.abs().max())
                tiled=inference.tiled_logits(model,probe,48,halo)
                torch.testing.assert_close(tiled,direct,atol=1e-4,rtol=1e-4)
                disagreement=int(((direct.sigmoid()>=.5)!=(tiled.sigmoid()>=.5)).sum())
                metadata['tiling_checks'].append({'crop_yxhw':[y,x,96,128],
                    'max_logit_error':float((direct-tiled).abs().max()),
                    'max_probability_error':float((direct.sigmoid()-tiled.sigmoid()).abs().max()),
                    'threshold05_disagreements':disagreement,'actual_max_abs_correction':correction_max,
                    'atol':1e-4,'rtol':1e-4,'status':'passed'})
                del probe,direct,tiled
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            for start in STARTS:
                image=torch.from_numpy(np.ascontiguousarray((raw[start:start+40].astype(np.float32)-85.0799)/47.4845))[None,None].to(device)
                output=torch.empty((1,40,512,640),dtype=torch.float32,device='cpu')
                recorder.begin_window(start)
                if device.type=='cuda': torch.cuda.synchronize(device)
                t0=time.perf_counter(); previous_telemetry=recorder.seconds
                for y in range(0,512,core):
                    y1=min(y+core,512)
                    for x in range(0,640,core):
                        x1=min(x+core,640)
                        ya,yb=max(0,y-halo),min(512,y1+halo)
                        xa,xb=max(0,x-halo),min(640,x1+halo)
                        result=model(image[...,ya:yb,xa:xb].contiguous())[1]
                        output[...,y:y1,x:x1]=result[...,y-ya:y1-ya,x-xa:x1-xa].cpu()
                        recorder.capture(model,start,y,y1,x,x1,ya,xa,device)
                        del result
                if device.type=='cuda': torch.cuda.synchronize(device)
                total=time.perf_counter()-t0; telemetry=recorder.seconds-previous_telemetry
                recorder.finish_window(start)
                probabilities=output.sigmoid()[0].numpy()
                if not np.isfinite(probabilities).all(): raise FloatingPointError('Nonfinite window output')
                stamp=time.perf_counter()
                np.savez_compressed(args.output/'windows'/f'window_{start:03d}.npz',probabilities=probabilities,start=start,end=start+40)
                write_seconds=time.perf_counter()-stamp
                if start in OFFICIAL:
                    fused[start:start+40]=np.maximum(fused[start:start+40],probabilities)
                    coverage[start:start+40]+=1
                metadata['window_timing'].append({'start':start,'model_plus_cpu_copy_seconds':total-telemetry,
                    'total_including_provenance_seconds':total,'provenance_seconds':telemetry,'window_write_seconds':write_seconds,
                    'tile_model_calls':20})
                print(json.dumps({'window_complete':start,'condition':args.variant,'elapsed_seconds':time.perf_counter()-begin}),flush=True)
                del image,output,probabilities
        recorder.close()
        if not np.all(coverage>0): raise AssertionError('Uncovered output frames')
        if runner.tensor_hash(model.original.state_dict().items())!=original_hash:
            raise AssertionError('Original state changed during inference')
        if sources!=runner.source_hashes(model_path): raise AssertionError('Sources changed during inference')
        np.savez_compressed(args.output/'official_fused.npz',probabilities=fused,coverage=coverage,starts=np.array(OFFICIAL))
        metadata.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-begin,
            new_model_inference_elapsed_seconds=time.perf_counter()-begin,
            O_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in OFFICIAL),
            S_model_seconds=sum(r['model_plus_cpu_copy_seconds'] for r in metadata['window_timing'] if r['start'] in SYMMETRIC),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
            fused_sha256=runner.sha(args.output/'official_fused.npz'),
            window_sha256={p.name:runner.sha(p) for p in sorted((args.output/'windows').glob('*.npz'))},
            telemetry_sha256=runner.sha(args.output/'EVIDENCE_SAMPLES.csv'),
            telemetry_target_rows=recorder.target_count,telemetry_background_rows=recorder.background_count,
            telemetry_seconds=recorder.seconds,frozen_original_unchanged=True)
        runner.save_json(args.output/'COMPLETE.json',metadata)
        print(json.dumps({'status':'COMPLETE','sequence':args.sequence,'condition':args.variant,'seconds':metadata['elapsed_seconds']}),flush=True)
    except BaseException as error:
        recorder.close()
        metadata.update(status='FAILED',error=repr(error),elapsed_seconds=time.perf_counter()-begin)
        runner.save_json(args.output/'FAILED.json',metadata)
        raise


if __name__=='__main__': main()
