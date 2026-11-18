#!/usr/bin/env python3
"""Freeze calibration thresholds, then descriptively summarize fused confirmation.

No inference and no confirmation-based selection. Use --stage calibrate first;
CALIBRATION.json is exclusively created and can never be silently overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VARIANTS = ['official','paired_mean','paired_max','ohem','random_pair_max']
GRID = [1e-6,1e-5,1e-4,1e-3,.01,.03,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,.999,.9999,.99999,.999999,1.]
COUNT_FIELDS = ['frames','gt','matched','strict_small_gt','strict_small_matched','components',
                'unmatched_components','positive_pixels','positive_pixels_outside_boxes']
FRAME_FIELDS = ['gt','matched','components','unmatched_components','positive_pixels','positive_pixels_outside_boxes']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def write_csv(path,rows,fields=None):
    if not rows and fields is None:raise ValueError('Empty CSV requires explicit fieldnames')
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)


def names(selection,split):
    seqs=selection['splits'][split]
    result=[x if isinstance(x,str) else x['sequence'] for x in seqs]
    if not result or len(set(result))!=len(result):raise ValueError(f'Invalid {split} sequences')
    return result


def object_key(obj):return (int(obj['frame']),int(obj['annotation_id']))


class FusedScore:
    def __init__(self,root,variant,sequence):
        self.variant=variant;self.sequence=sequence;self.directory=Path(root)/variant/sequence
        paths={suffix:self.directory/('official_fused_'+suffix)
               for suffix in ['curves.csv','matches.npz','COMPLETE.json']}
        self.fingerprints={k:digest(v) for k,v in paths.items()}
        self.meta=json.loads(paths['COMPLETE.json'].read_text())
        if self.meta.get('status')!='COMPLETE' or self.meta.get('name')!='official_fused':
            raise ValueError(f'{self.directory}: scorer is not complete fused output')
        if self.meta['start']!=0 or self.meta['end']!=160:
            raise ValueError('Pilot fused score must cover all 160 frames')
        if self.meta.get('frame_count_fields')!=FRAME_FIELDS:raise ValueError('Frame count schema changed')
        self.objects=self.meta['objects']
        self.keys=[object_key(x) for x in self.objects]
        if len(set(self.keys))!=len(self.keys):raise ValueError('Duplicate annotation/frame keys')
        with paths['curves.csv'].open() as f:
            self.curves=[{'threshold':float(row['threshold']),**{k:int(row[k]) for k in COUNT_FIELDS}}
                         for row in csv.DictReader(f)]
        if [r['threshold'] for r in self.curves]!=GRID:raise ValueError('Frozen threshold grid changed')
        with np.load(paths['matches.npz'],allow_pickle=False) as z:
            self.data={k:z[k] for k in z.files}
        nt,n=len(GRID),len(self.objects)
        if self.data['thresholds'].tolist()!=GRID or self.meta['thresholds']!=GRID:
            raise ValueError('CSV / NPZ / COMPLETE thresholds disagree')
        for field in ['matched','center_error','peak_x','peak_y','component_area']:
            if self.data[field].shape!=(nt,n):raise ValueError(f'Wrong {field} array shape')
        if not np.isin(self.data['matched'],[0,1]).all():raise ValueError('Nonbinary matching')
        self.strict=np.array([bool(x['strict_small']) for x in self.objects])
        if not np.array_equal(self.strict,self.data['strict_small']):raise ValueError('Strict strata disagree')
        offsets=self.data['offsets']
        if offsets.shape!=(161,) or offsets[0]!=0 or offsets[-1]!=n or (np.diff(offsets)<0).any():
            raise ValueError('Invalid frame offsets')
        counts=self.data['frame_counts']
        if counts.shape!=(nt,160,6) or (counts<0).any():raise ValueError('Invalid frame counts')
        if not np.all(counts[:,:,0]==np.diff(offsets)[None,:]):raise ValueError('GT count differs from offsets')
        if not np.array_equal(counts[:,:,2]-counts[:,:,1],counts[:,:,3]):raise ValueError('Component conservation failed')
        if (counts[:,:,5]>counts[:,:,4]).any():raise ValueError('Outside positive pixels exceed all positive pixels')
        for frame in range(160):
            lo,hi=int(offsets[frame]),int(offsets[frame+1])
            if any(int(x['frame'])!=frame for x in self.objects[lo:hi]):raise ValueError('Metadata / offsets order differs')
            if not np.array_equal(self.data['matched'][:,lo:hi].sum(1),counts[:,frame,1]):
                raise ValueError('Per-frame matched sum differs')
        for ti,row in enumerate(self.curves):
            totals=counts[ti].sum(0)
            for field,total in zip(FRAME_FIELDS,totals):
                if row[field]!=int(total):raise ValueError(f'CSV / NPZ {field} differs')
            if row['frames']!=160 or row['strict_small_gt']!=int(self.strict.sum()):raise ValueError('Curve denominator differs')
            if row['strict_small_matched']!=int(self.data['matched'][ti,self.strict].sum()):raise ValueError('Strict matches differ')
            matched=self.data['matched'][ti].astype(bool)
            for field in ['center_error','peak_x','peak_y']:
                if not np.isfinite(self.data[field][ti,matched]).all():raise ValueError('Matched localization missing')

    def selected(self,threshold):
        matches=[i for i,t in enumerate(GRID) if t==threshold]
        if len(matches)!=1:raise ValueError(f'Frozen threshold absent: {threshold}')
        ti=matches[0]
        return {'objects':self.objects,'keys':self.keys,'strict':self.strict,'threshold':threshold,
                'curve':self.curves[ti],
                **{k:self.data[k][ti] for k in ['matched','center_error','peak_x','peak_y','component_area']}}


def aggregate_curves(scores):
    if not scores:raise ValueError('Cannot aggregate no sequences')
    return [{'threshold':threshold,**{field:sum(s.curves[i][field] for s in scores) for field in COUNT_FIELDS}}
            for i,threshold in enumerate(GRID)]


def choose_threshold(curves,budget_u,budget_frames):
    feasible=[r for r in curves if r['unmatched_components']*budget_frames<=budget_u*r['frames']]
    if not feasible:return None
    return min(feasible,key=lambda r:(-r['strict_small_matched'],r['unmatched_components'],-r['threshold']))


def load_split(args,selection,split,variant):
    return [FusedScore(args.evaluation_root,variant,seq) for seq in names(selection,split)]


def calibrate(args,selection):
    destination=args.calibration or args.output/'CALIBRATION.json'
    if destination.exists():raise FileExistsError(f'Calibration is already frozen: {destination}')
    seqs=names(selection,'calibration');sources={};choices={}
    original=load_split(args,selection,'calibration','official')
    baseline=next(r for r in aggregate_curves(original) if r['threshold']==.5)
    budget_u,budget_frames=baseline['unmatched_components'],baseline['frames']
    all_curves=[]
    for variant in VARIANTS:
        scores=original if variant=='official' else load_split(args,selection,'calibration',variant)
        for score,reference in zip(scores,original):align(reference.selected(.5),score.selected(.5))
        curves=aggregate_curves(scores)
        chosen=baseline if variant=='official' else choose_threshold(curves,budget_u,budget_frames)
        choices[variant]={'status':'frozen' if chosen else 'calibration_failed',
                          'threshold':chosen['threshold'] if chosen else None,
                          'selected_calibration_counts':chosen,
                          'rule':'fixed official reference at 0.5' if variant=='official' else
                          'U/frames <= official calibration 0.5 budget; maximize strict-small matches, then minimize U, then maximize threshold',
                          'failure_reason':None if chosen else 'No feasible point on the frozen grid; no added threshold'}
        sources[variant]={s.sequence:{**s.fingerprints,'annotations_sha256':s.meta['annotations_sha256'],
                                     'prediction_sha256':s.meta['prediction_sha256']} for s in scores}
        all_curves.extend({'variant':variant,**r} for r in curves)
    payload={'status':'frozen','created_utc':datetime.now(timezone.utc).isoformat(),
             'selection_sha256':digest(args.selection),'calibration_sequences':seqs,
             'budget':{'reference_variant':'official','reference_threshold':.5,
                       'unmatched_components':budget_u,'frames':budget_frames,'U_per_frame':budget_u/budget_frames},
             'threshold_grid':GRID,'variants':choices,'source_fingerprints':sources,
             'summary_script_sha256':digest(__file__),
             'interpretation':'Diagnostic point matches and unmatched components; not official Pd/Fa. No confirmation outcomes read.'}
    args.output.mkdir(parents=True,exist_ok=True)
    # All inputs are validated before the exclusive, immutable decision is written.
    write_json(destination,payload,exclusive=True)
    destination.with_suffix(destination.suffix+'.sha256').write_text(digest(destination)+'\n')
    write_csv(args.output/'CALIBRATION_CURVES.csv',all_curves)
    print(json.dumps({'calibration':str(destination),'sha256':digest(destination),
                      'budget':payload['budget'],'variants':choices}),flush=True)


def distribution(values):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if not len(values):return {'n':0,'mean':None,'median':None,'p95':None,'max':None}
    return {'n':len(values),'mean':float(values.mean()),'median':float(np.median(values)),
            'p95':float(np.quantile(values,.95)),'max':float(values.max())}


def align(reference,candidate):
    if set(reference['keys'])!=set(candidate['keys']):raise ValueError('Compared variants contain different object/frame sets')
    lookup={key:i for i,key in enumerate(candidate['keys'])}
    order=np.array([lookup[k] for k in reference['keys']],dtype=int)
    for ri,ci in enumerate(order):
        a,b=reference['objects'][ri],candidate['objects'][ci]
        for key in ['frame','source_frame','annotation_id','image_id','track_id','category_id','bbox','strict_small']:
            if a[key]!=b[key]:raise ValueError(f'Compared object metadata differs at {key}')
    return order


def compare(reference,candidate,scope):
    order=align(reference,candidate)
    eligible=reference['strict'] if scope=='strict_small' else np.ones(len(order),bool)
    r=reference['matched'].astype(bool);c=candidate['matched'][order].astype(bool)
    gained=eligible & ~r & c;lost=eligible & r & ~c;joint=eligible&r&c
    dx=candidate['peak_x'][order][joint]-reference['peak_x'][joint]
    dy=candidate['peak_y'][order][joint]-reference['peak_y'][joint]
    boxes=np.asarray([o['bbox'] for o in reference['objects']],dtype=float).reshape(-1,4)
    normalized=np.hypot(dx/boxes[joint,2],dy/boxes[joint,3])
    return {'gt':int(eligible.sum()),'reference_matched':int((eligible&r).sum()),
            'candidate_matched':int((eligible&c).sum()),'gained':int(gained.sum()),'lost':int(lost.sum()),
            'net':int(gained.sum()-lost.sum()),'joint_matched':int(joint.sum()),
            'peak_displacement_pixels':distribution(np.hypot(dx,dy)),
            'peak_displacement_bbox_normalized':distribution(normalized),
            'center_error_delta_on_joint_matches':distribution(candidate['center_error'][order][joint]-reference['center_error'][joint]),
            'localization_scope':'Only jointly matched identical annotated object-frames; unmatched peaks are unavailable'}


def longest_miss(objects,matched):
    longest=0;best_start=None;best_end=None;run=0;run_start=None;previous=None
    for obj,ok in sorted(zip(objects,matched),key=lambda x:int(x[0]['source_frame'])):
        frame=int(obj['source_frame'])
        if previous is not None and frame==previous:raise ValueError('Duplicate track/frame observation')
        if ok or previous is None or frame!=previous+1:
            run=0;run_start=None
        if not ok:
            run+=1
            if run_start is None:run_start=frame
            if run>longest:longest=run;best_start=run_start;best_end=frame
        previous=frame
    return {'longest_miss_run':longest,'longest_start_source_frame':best_start,'longest_end_source_frame':best_end}


def track_rows(variant,sequence,selected):
    result=[]
    for scope in ['all','strict_small']:
        groups=defaultdict(list)
        for i,obj in enumerate(selected['objects']):
            if scope=='all' or selected['strict'][i]:groups[obj['track_id']].append(i)
        for track,indices in sorted(groups.items(),key=lambda x:str(x[0])):
            objects=[selected['objects'][i] for i in indices];matched=selected['matched'][indices].astype(bool)
            result.append({'variant':variant,'sequence':sequence,'scope':scope,'track_id':track,
                           'threshold':selected['threshold'],'observed_target_frames':len(indices),
                           'matched':int(matched.sum()),'missed':int((~matched).sum()),
                           **longest_miss(objects,matched),
                           'continuity_rule':'Source-frame gaps and filtered-out observations break a run'})
    return result


def numeric_summary(row,budget):
    frames=row['frames'];pixels=frames*512*640
    u=row['unmatched_components']
    return {**row,'all_match_rate':row['matched']/row['gt'] if row['gt'] else None,
            'strict_small_match_rate':row['strict_small_matched']/row['strict_small_gt'] if row['strict_small_gt'] else None,
            'U_per_frame':u/frames,'positive_pixels_per_frame':row['positive_pixels']/frames,
            'positive_pixel_fraction':row['positive_pixels']/pixels,
            'outside_positive_pixels_per_frame':row['positive_pixels_outside_boxes']/frames,
            'U_per_positive_pixel':u/row['positive_pixels'] if row['positive_pixels'] else None,
            'satisfies_frozen_calibration_U_rate':u*budget['frames']<=budget['unmatched_components']*frames}


def concat_selected(selected):
    # Use scoped keys to prevent any cross-sequence annotation/track ID collision.
    merged={k:[] for k in ['objects','keys','strict','matched','center_error','peak_x','peak_y','component_area']}
    for sequence,s in selected:
        merged['objects'].extend(s['objects']);merged['keys'].extend((sequence,*key) for key in s['keys'])
        for k in ['strict','matched','center_error','peak_x','peak_y','component_area']:merged[k].append(s[k])
    for k in ['strict','matched','center_error','peak_x','peak_y','component_area']:
        merged[k]=np.concatenate(merged[k])
    return merged


def verify_calibration(path,args,selection):
    calibration=json.loads(path.read_text())
    hash_path=path.with_suffix(path.suffix+'.sha256')
    if not hash_path.is_file() or hash_path.read_text().strip()!=digest(path):raise ValueError('Calibration SHA mismatch')
    if calibration['status']!='frozen' or calibration['selection_sha256']!=digest(args.selection):raise ValueError('Calibration / selection changed')
    if calibration['calibration_sequences']!=names(selection,'calibration'):raise ValueError('Calibration split changed')
    if calibration['threshold_grid']!=GRID:raise ValueError('Calibration grid changed')
    if calibration['variants']['official']['threshold']!=.5:raise ValueError('Official reference must remain at 0.5')
    # Read only hashes, never recompute selection from confirmation outcomes.
    for variant,sequences in calibration['source_fingerprints'].items():
        for sequence,hashes in sequences.items():
            directory=args.evaluation_root/variant/sequence
            for suffix in ['curves.csv','matches.npz','COMPLETE.json']:
                if digest(directory/('official_fused_'+suffix))!=hashes[suffix]:
                    raise ValueError('Scored calibration inputs changed after threshold freeze')
    return calibration


def confirmation(args,selection):
    cal_path=args.calibration or args.output/'CALIBRATION.json'
    calibration=verify_calibration(cal_path,args,selection)
    result_path=args.output/'FUSED_CONFIRMATION.json'
    if result_path.exists():raise FileExistsError('Completed confirmation summary already exists')
    args.output.mkdir(parents=True,exist_ok=True)
    sequences=names(selection,'confirmation');budget=calibration['budget']
    per_variant={};sources={};curves=[];tracks=[];selected_by_variant={};sequence_rows=[];object_rows=[]
    for variant in VARIANTS:
        choice=calibration['variants'][variant]
        if choice['status']!='frozen':
            per_variant[variant]={'status':'not_evaluated_due_to_calibration_failure','threshold':None}
            continue
        scores=load_split(args,selection,'confirmation',variant);threshold=choice['threshold']
        selected=[(s.sequence,s.selected(threshold)) for s in scores];selected_by_variant[variant]=dict(selected)
        aggregated=aggregate_curves(scores)
        counts=next(row for row in aggregated if row['threshold']==threshold)
        merged=concat_selected(selected)
        per_variant[variant]={'status':'evaluated_at_frozen_threshold','metrics':numeric_summary(counts,budget),
                              'center_error':{scope:distribution(merged['center_error'][merged['matched'].astype(bool)&(
                                  merged['strict'] if scope=='strict_small' else np.ones(len(merged['matched']),bool))])
                                  for scope in ['all','strict_small']},'comparisons':{}}
        curves.extend({'variant':variant,**r} for r in aggregated)
        sources[variant]={s.sequence:{**s.fingerprints,'annotations_sha256':s.meta['annotations_sha256'],
                                     'prediction_sha256':s.meta['prediction_sha256']} for s in scores}
        for sequence,s in selected:
            tracks.extend(track_rows(variant,sequence,s))
            sequence_rows.append({'variant':variant,'sequence':sequence,**numeric_summary(s['curve'],budget)})
    for variant,seqs in selected_by_variant.items():
        candidate=concat_selected(list(seqs.items()))
        for reference in ['official','paired_mean']:
            if reference not in selected_by_variant:
                per_variant[variant]['comparisons'][reference]={'status':'reference_calibration_failed'}
                continue
            baseline=concat_selected(list(selected_by_variant[reference].items()))
            per_variant[variant]['comparisons'][reference]={scope:compare(baseline,candidate,scope) for scope in ['all','strict_small']}
        for sequence,s in seqs.items():
            sequence_row=next(r for r in sequence_rows if r['variant']==variant and r['sequence']==sequence)
            for reference in ['official','paired_mean']:
                for scope in ['all','strict_small']:
                    comparison=compare(selected_by_variant[reference][sequence],s,scope) if reference in selected_by_variant else None
                    for key in ['gained','lost','net','joint_matched']:
                        sequence_row[f'{scope}_{key}_vs_{reference}']=comparison[key] if comparison else ''
            maps={reference:align(selected_by_variant[reference][sequence],s)
                  for reference in ['official','paired_mean'] if reference in selected_by_variant}
            # Invert candidate order to corresponding reference order for object CSV.
            ref_indices={reference:np.argsort(order) for reference,order in maps.items()}
            for ci,obj in enumerate(s['objects']):
                row={'variant':variant,'sequence':sequence,'threshold':s['threshold'],**obj,
                     'bbox':json.dumps(obj['bbox']),'matched':int(s['matched'][ci]),
                     'center_error':float(s['center_error'][ci]) if s['matched'][ci] else '',
                     'peak_x':float(s['peak_x'][ci]) if s['matched'][ci] else '',
                     'peak_y':float(s['peak_y'][ci]) if s['matched'][ci] else ''}
                for reference in ['official','paired_mean']:
                    ri=ref_indices[reference][ci] if reference in ref_indices else None
                    ok=int(selected_by_variant[reference][sequence]['matched'][ri]) if ri is not None else None
                    row[reference+'_matched']='' if ok is None else ok
                    row['gained_vs_'+reference]='' if ok is None else int(not ok and s['matched'][ci])
                    row['lost_vs_'+reference]='' if ok is None else int(ok and not s['matched'][ci])
                object_rows.append(row)
    candidate=per_variant.get('paired_max',{});control=per_variant.get('paired_mean',{})
    if candidate.get('status')=='evaluated_at_frozen_threshold' and control.get('status')=='evaluated_at_frozen_threshold':
        gain=candidate['comparisons']['paired_mean']['strict_small']['net']
        u_delta=candidate['metrics']['unmatched_components']-control['metrics']['unmatched_components']
        fused_gate={'status':'passed_fused_prerequisite' if gain>0 and u_delta<=0 else 'failed_fused_prerequisite',
                    'strict_small_net_vs_paired_mean':gain,'U_delta_vs_paired_mean':u_delta,
                    'rule':'Net strict-small matches > 0 AND unmatched components do not increase',
                    'stage1_positive_decision':False,
                    'reason':'Fused prerequisite alone cannot pass stage 1; reference-defined context preservation still required'}
    else:fused_gate={'status':'not_evaluable_due_to_calibration_failure','stage1_positive_decision':False}
    payload={'status':'fused_confirmation_complete_context_pending','created_utc':datetime.now(timezone.utc).isoformat(),
             'selection_sha256':digest(args.selection),'calibration_sha256':digest(cal_path),
             'confirmation_sequences':sequences,'variants':per_variant,'source_fingerprints':sources,
             'fused_prerequisite':fused_gate,
             'context_interface':{'status':'pending_window_scores','root_argument_for_followup':'Separate scored context/<variant>/<sequence>/window_START_{matches.npz,COMPLETE.json}',
                 'thresholds':'Exactly the existing CALIBRATION.json thresholds; official remains 0.5',
                 'reference_sets':'Defined by official window matches and GT only, never by candidate outputs',
                 'support_set':'Annotated object-frames with >=1 matched official reference window',
                 'stable_set':'Strict-small object-frames covered by >=2 windows and matched in every covering official window',
                 'required_outputs':['restored/missed window-object pairs','support retained/lost','stable new fused losses and explicit denominator',
                                     'double-low consistency','cross-window peak displacement','existing 97/24/42/778 development strata'],
                 'protected_set':'Stable set intersected with official fused matched at threshold 0.5',
                 'stable_original_fused_misses':'Report stable-set members already missed by official fusion separately; these are not candidate-added losses',
                 'preservation_gate':'<=1% candidate fused losses among protected-set members; empty protected set is not a pass'},
             'summary_script_sha256':digest(__file__),
             'limitations':['Single training seed and short budget','Diagnostic matching and unmatched components, not official Pd/Fa',
                            'MIL maxima may occur on within-box background','Geographic scene independence not guaranteed by source-prefix splitting',
                            'Confirmation curves are descriptive; no threshold/model selection from them']}
    write_csv(args.output/'CONFIRMATION_CURVES.csv',curves)
    write_csv(args.output/'CONFIRMATION_SEQUENCES.csv',sequence_rows)
    write_csv(args.output/'CONFIRMATION_TRACKS.csv',tracks)
    write_csv(args.output/'CONFIRMATION_OBJECTS.csv',object_rows)
    payload['output_sha256']={name:digest(args.output/name) for name in ['CONFIRMATION_CURVES.csv','CONFIRMATION_SEQUENCES.csv',
                                'CONFIRMATION_TRACKS.csv','CONFIRMATION_OBJECTS.csv']}
    write_json(result_path,payload,exclusive=True)
    print(json.dumps({'summary':str(result_path),'fused_prerequisite':fused_gate}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-root',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--stage',choices=['calibrate','confirmation'],required=True)
    args=parser.parse_args();selection=json.loads(args.selection.read_text())
    if set(names(selection,'calibration')) & set(names(selection,'confirmation')):raise ValueError('Split overlap')
    if args.stage=='calibrate':calibrate(args,selection)
    else:confirmation(args,selection)


if __name__=='__main__':main()
