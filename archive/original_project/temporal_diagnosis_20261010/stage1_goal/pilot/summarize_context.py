#!/usr/bin/env python3
"""Reference-defined context diagnostics at already frozen calibration thresholds.

Reads scoring artifacts only. It never performs inference, adjusts thresholds,
selects a checkpoint, or ranks a winning method from confirmation outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import summarize_pilot as fused

STARTS=list(range(0,121,12))
SUM_FIELDS=['strict_small_object_frames','eligible_object_frames','excluded_coverage_lt_2',
            'window_object_pairs','reference_window_matches','candidate_window_matches',
            'gained_window_matches','lost_window_matches','joint_window_matches',
            'reference_any','reference_any_to_candidate_none','reference_any_to_candidate_any',
            'reference_none','reference_none_to_candidate_any','reference_none_to_candidate_none',
            'reference_all','reference_all_to_candidate_all','reference_all_to_candidate_any',
            'reference_all_to_candidate_none','reference_partial',
            'reference_partial_to_candidate_none','reference_partial_to_candidate_all','reference_partial_to_candidate_partial',
            'stable_original_fused_misses','protected_denominator','protected_candidate_fused_losses']


class WindowBundle:
    """All eleven windows aligned to the complete fused object's absolute frame."""
    def __init__(self,root,variant,sequence,threshold,fused_score):
        self.directory=Path(root)/variant/sequence
        expected_npz={f'window_{start:03d}_matches.npz' for start in STARTS}
        expected_meta={f'window_{start:03d}_COMPLETE.json' for start in STARTS}
        if {p.name for p in self.directory.glob('window_*_matches.npz')}!=expected_npz:
            raise ValueError(f'{self.directory}: context NPZ set is not exactly eleven frozen starts')
        if {p.name for p in self.directory.glob('window_*_COMPLETE.json')}!=expected_meta:
            raise ValueError(f'{self.directory}: context metadata set is not exactly eleven frozen starts')
        self.threshold=threshold;self.objects=fused_score.objects;self.keys=fused_score.keys
        self.strict=fused_score.strict;self.fingerprints={}
        n=len(self.objects);lookup={key:i for i,key in enumerate(self.keys)}
        self.covered=np.zeros((len(STARTS),n),bool)
        self.matched=np.zeros((len(STARTS),n),bool)
        self.peak_x=np.full((len(STARTS),n),np.nan,np.float32)
        self.peak_y=self.peak_x.copy();self.center_error=self.peak_x.copy()
        for wi,start in enumerate(STARTS):
            stem=f'window_{start:03d}'
            meta_path=self.directory/(stem+'_COMPLETE.json');npz_path=self.directory/(stem+'_matches.npz')
            meta=json.loads(meta_path.read_text())
            if meta.get('status')!='COMPLETE' or meta.get('name')!=stem or meta['start']!=start or meta['end']!=start+40:
                raise ValueError('Context window completion or absolute frame range differs')
            if meta['thresholds']!=[threshold] or meta['frame_count_fields']!=fused.FRAME_FIELDS:
                raise ValueError('Context threshold/schema differs from frozen calibration')
            if meta['annotations_sha256']!=fused_score.meta['annotations_sha256']:
                raise ValueError('Context and fused annotations hashes differ')
            objects=meta['objects'];keys=[fused.object_key(x) for x in objects]
            expected={key for key in self.keys if start<=key[0]<start+40}
            if len(set(keys))!=len(keys) or set(keys)!=expected:
                raise ValueError('Context objects differ from corresponding fused frame interval')
            indices=np.array([lookup[key] for key in keys],dtype=int)
            for obj,index in zip(objects,indices):
                if obj!=self.objects[index]:raise ValueError('Context / fused object metadata differs')
            with np.load(npz_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
            m=len(objects)
            if data['thresholds'].tolist()!=[threshold]:raise ValueError('NPZ threshold differs from calibration')
            for field in ['matched','peak_x','peak_y','center_error','component_area']:
                if data[field].shape!=(1,m):raise ValueError(f'Context {field} has wrong dimensions')
            if not np.isin(data['matched'],[0,1]).all():raise ValueError('Context matches are not binary')
            if not np.array_equal(data['strict_small'],self.strict[indices]):raise ValueError('Context strict-small mask differs')
            offsets=data['offsets'];counts=data['frame_counts']
            if offsets.shape!=(41,) or offsets[0]!=0 or offsets[-1]!=m or (np.diff(offsets)<0).any():
                raise ValueError('Invalid context-relative offsets')
            if counts.shape!=(1,40,6) or (counts<0).any():raise ValueError('Invalid context frame counts')
            if not np.array_equal(counts[0,:,0],np.diff(offsets)):raise ValueError('Context GT count mismatch')
            if not np.array_equal(counts[0,:,2]-counts[0,:,1],counts[0,:,3]):raise ValueError('Context component conservation failed')
            if (counts[0,:,5]>counts[0,:,4]).any():raise ValueError('Outside positive pixels exceed all positive pixels')
            for local,frame in enumerate(range(start,start+40)):
                lo,hi=int(offsets[local]),int(offsets[local+1])
                if any(int(x['frame'])!=frame for x in objects[lo:hi]):raise ValueError('Context offset / absolute frame mismatch')
                if data['matched'][0,lo:hi].sum()!=counts[0,local,1]:raise ValueError('Context per-frame matching mismatch')
            matched=data['matched'][0].astype(bool)
            for field in ['center_error','peak_x','peak_y']:
                if not np.isfinite(data[field][0,matched]).all():raise ValueError('Matched context localization is missing')
            self.covered[wi,indices]=True;self.matched[wi,indices]=matched
            self.peak_x[wi,indices]=data['peak_x'][0];self.peak_y[wi,indices]=data['peak_y'][0]
            self.center_error[wi,indices]=data['center_error'][0]
            self.fingerprints[stem]={'matches_sha256':fused.digest(npz_path),'complete_sha256':fused.digest(meta_path),
                                     'prediction_sha256':meta['prediction_sha256'],'threshold':threshold}
        expected_coverage=np.array([[start<=obj['frame']<start+40 for obj in self.objects] for start in STARTS])
        if not np.array_equal(self.covered,expected_coverage):raise ValueError('Context coverage differs from frozen starts')


def protection_gate(denominator,losses):
    if denominator<0 or not 0<=losses<=denominator:raise ValueError('Invalid protection counts')
    if denominator==0:
        return {'status':'not_evaluable_empty_protected_set','denominator':0,'new_fused_losses':0,
                'new_fused_loss_rate':None,'passes':False,'criterion':'100 * losses <= denominator (1%)'}
    return {'status':'passed' if 100*losses<=denominator else 'failed','denominator':denominator,
            'new_fused_losses':losses,'new_fused_loss_rate':losses/denominator,
            'passes':100*losses<=denominator,'criterion':'100 * losses <= denominator (1%)'}


def peak_span(bundle):
    """Maximum Euclidean displacement across that model's matched windows only."""
    n=bundle.matched.shape[1];span=np.full(n,np.nan)
    for i in range(len(STARTS)):
        for j in range(i+1,len(STARTS)):
            eligible=bundle.covered[i]&bundle.covered[j]&bundle.matched[i]&bundle.matched[j]
            distance=np.hypot(bundle.peak_x[i]-bundle.peak_x[j],bundle.peak_y[i]-bundle.peak_y[j])
            span[eligible]=np.fmax(span[eligible],distance[eligible])
    return span


def analyze_sequence(variant,sequence,reference,candidate,ref_fused,cand_fused):
    order=fused.align(ref_fused,cand_fused)
    if ref_fused['keys']!=reference.keys:raise ValueError('Reference fused and context order differ')
    if cand_fused['keys']!=candidate.keys:raise ValueError('Candidate fused and context order differ')
    # Align by annotation/frame keys rather than relying on incidental array ordering.
    if not np.array_equal(order,np.arange(len(order))):
        candidate=SimpleNamespace(threshold=candidate.threshold,objects=[candidate.objects[i] for i in order],
                   keys=[candidate.keys[i] for i in order],strict=candidate.strict[order],
                   **{k:getattr(candidate,k)[:,order] for k in ['covered','matched','peak_x','peak_y','center_error']})
        cand_fused={**cand_fused,**{k:cand_fused[k][order] for k in ['matched','center_error','peak_x','peak_y','component_area']}}
    if reference.keys!=candidate.keys or not np.array_equal(reference.covered,candidate.covered):
        raise ValueError('Context variants have different object/frame coverage')
    coverage=reference.covered;number=coverage.sum(0)
    eligible=reference.strict&(number>=2)
    r=reference.matched&coverage;c=candidate.matched&coverage
    rc=r.sum(0);cc=c.sum(0)
    any_r=rc>0;all_r=rc==number;none_r=rc==0;partial_r=any_r&~all_r
    any_c=cc>0;all_c=cc==number;none_c=cc==0;partial_c=any_c&~all_c
    stable=eligible&all_r
    original_fused_matched=ref_fused['matched'].astype(bool)
    candidate_fused_matched=cand_fused['matched'].astype(bool)
    protected=stable&original_fused_matched
    gained=~r&c&coverage;lost=r&~c&coverage;joint=r&c&coverage
    count=lambda mask:int(np.count_nonzero(eligible&mask))
    summary={'variant':variant,'sequence':sequence,'threshold':candidate.threshold,
             'reference_threshold':reference.threshold,
             'strict_small_object_frames':int(reference.strict.sum()),'eligible_object_frames':int(eligible.sum()),
             'excluded_coverage_lt_2':int((reference.strict&~eligible).sum()),
             'window_object_pairs':int(coverage[:,eligible].sum()),
             'reference_window_matches':int(r[:,eligible].sum()),'candidate_window_matches':int(c[:,eligible].sum()),
             'gained_window_matches':int(gained[:,eligible].sum()),'lost_window_matches':int(lost[:,eligible].sum()),
             'joint_window_matches':int(joint[:,eligible].sum()),
             'reference_any':count(any_r),'reference_any_to_candidate_none':count(any_r&none_c),
             'reference_any_to_candidate_any':count(any_r&any_c),
             'reference_none':count(none_r),'reference_none_to_candidate_any':count(none_r&any_c),
             'reference_none_to_candidate_none':count(none_r&none_c),
             'reference_all':count(all_r),'reference_all_to_candidate_all':count(all_r&all_c),
             'reference_all_to_candidate_any':count(all_r&any_c),'reference_all_to_candidate_none':count(all_r&none_c),
             'reference_partial':count(partial_r),'reference_partial_to_candidate_none':count(partial_r&none_c),
             'reference_partial_to_candidate_all':count(partial_r&all_c),
             'reference_partial_to_candidate_partial':count(partial_r&partial_c),
             'stable_original_fused_misses':int((stable&~original_fused_matched).sum()),
             'protected_denominator':int(protected.sum()),
             'protected_candidate_fused_losses':int((protected&~candidate_fused_matched).sum())}
    summary['net_window_matches']=summary['gained_window_matches']-summary['lost_window_matches']
    gate=protection_gate(summary['protected_denominator'],summary['protected_candidate_fused_losses'])
    summary.update(protection_gate_status=gate['status'],protected_loss_rate=gate['new_fused_loss_rate'])
    dx=candidate.peak_x-reference.peak_x;dy=candidate.peak_y-reference.peak_y
    movement=np.hypot(dx,dy);matched_mask=joint&eligible[None,:]
    ref_span=peak_span(reference);cand_span=peak_span(candidate)
    comparable_span=eligible&np.isfinite(ref_span)&np.isfinite(cand_span)
    measurements={'joint_window_peak_displacement_pixels':movement[matched_mask],
                  'reference_cross_window_peak_span_pixels':ref_span[eligible&np.isfinite(ref_span)],
                  'candidate_cross_window_peak_span_pixels':cand_span[eligible&np.isfinite(cand_span)],
                  'peak_span_delta_common_objects_pixels':(cand_span-ref_span)[comparable_span]}
    for name,values in measurements.items():summary[name+'_mean']=float(values.mean()) if len(values) else None
    object_rows=[]
    for i in np.flatnonzero(eligible):
        obj=reference.objects[i];window_indices=np.flatnonzero(coverage[:,i])
        joint_positions=joint[:,i]
        stratum='all_windows_matched' if all_r[i] else 'some_windows_matched' if any_r[i] else 'no_windows_matched'
        object_rows.append({'variant':variant,'sequence':sequence,'threshold':candidate.threshold,
            **obj,'bbox':json.dumps(obj['bbox']), 'reference_context_stratum':stratum,
            'covering_windows':int(number[i]),'window_starts':';'.join(str(STARTS[k]) for k in window_indices),
            'original_window_match_bits':''.join(str(int(r[k,i])) for k in window_indices),
            'candidate_window_match_bits':''.join(str(int(c[k,i])) for k in window_indices),
            'original_window_matches':int(rc[i]),'candidate_window_matches':int(cc[i]),
            'gained_window_matches':int(gained[:,i].sum()),'lost_window_matches':int(lost[:,i].sum()),
            'original_any_to_candidate_none':int(any_r[i]&none_c[i]),
            'original_none_to_candidate_any':int(none_r[i]&any_c[i]),
            'original_all_to_candidate_all':int(all_r[i]&all_c[i]),
            'partial_to_all_unmatched_consistency':int(partial_r[i]&none_c[i]),
            'original_fused_matched':int(original_fused_matched[i]),'candidate_fused_matched':int(candidate_fused_matched[i]),
            'in_stable_set':int(stable[i]),'stable_original_fused_miss':int(stable[i]&~original_fused_matched[i]),
            'in_protected_denominator':int(protected[i]),
            'protected_new_fused_loss':int(protected[i]&~candidate_fused_matched[i]),
            'joint_window_matches':int(joint_positions.sum()),
            'joint_window_peak_displacement_mean_pixels':float(movement[joint_positions,i].mean()) if joint_positions.any() else '',
            'original_cross_window_peak_span_pixels':float(ref_span[i]) if np.isfinite(ref_span[i]) else '',
            'candidate_cross_window_peak_span_pixels':float(cand_span[i]) if np.isfinite(cand_span[i]) else ''})
    window_rows=[]
    for wi,start in enumerate(STARTS):
        mask=eligible&coverage[wi]
        window_rows.append({'variant':variant,'sequence':sequence,'threshold':candidate.threshold,'window_start':start,
                            'eligible_object_frames':int(mask.sum()),'original_matches':int((mask&r[wi]).sum()),
                            'candidate_matches':int((mask&c[wi]).sum()),'gained':int((mask&gained[wi]).sum()),
                            'lost':int((mask&lost[wi]).sum()),'joint_matched':int((mask&joint[wi]).sum())})
    return summary,object_rows,window_rows,measurements


def validate_fused_inputs(args,selection,calibration):
    path=args.fused_summary or args.output/'FUSED_CONFIRMATION.json'
    summary=json.loads(path.read_text())
    if summary['status']!='fused_confirmation_complete_context_pending':raise ValueError('Expected completed fused confirmation summary')
    cal_path=args.calibration or args.output/'CALIBRATION.json'
    if summary['calibration_sha256']!=fused.digest(cal_path) or summary['selection_sha256']!=fused.digest(args.selection):
        raise ValueError('Fused summary uses another calibration or selection')
    if summary['confirmation_sequences']!=fused.names(selection,'confirmation'):raise ValueError('Fused confirmation sequences changed')
    for name,sha in summary['output_sha256'].items():
        if fused.digest(path.parent/name)!=sha:raise ValueError('Fused result table changed after summarization')
    return summary,path


def run(args):
    selection=json.loads(args.selection.read_text())
    cal_path=args.calibration or args.output/'CALIBRATION.json'
    calibration=fused.verify_calibration(cal_path,args,selection)
    fused_summary,fused_path=validate_fused_inputs(args,selection,calibration)
    destination=args.output/'CONTEXT_CONFIRMATION.json'
    if destination.exists():raise FileExistsError('Context summary already exists; refusing silent replacement')
    sequences=fused.names(selection,'confirmation')
    variants=[v for v in fused.VARIANTS if calibration['variants'][v]['status']=='frozen']
    per_sequence=[];all_objects=[];all_windows=[];measurements={v:{} for v in variants};sources={v:{} for v in variants}
    for sequence in sequences:
        reference_score=fused.FusedScore(args.evaluation_root,'official',sequence)
        reference_fused=reference_score.selected(.5)
        reference=WindowBundle(args.context_root,'official',sequence,.5,reference_score)
        for variant in variants:
            threshold=calibration['variants'][variant]['threshold']
            score=reference_score if variant=='official' else fused.FusedScore(args.evaluation_root,variant,sequence)
            expected=fused_summary['source_fingerprints'][variant][sequence]
            if any(score.fingerprints[k]!=expected[k] for k in score.fingerprints):raise ValueError('Fused scores changed after summary')
            candidate_fused=score.selected(threshold)
            candidate=reference if variant=='official' else WindowBundle(args.context_root,variant,sequence,threshold,score)
            # Strict metadata comparison checks annotation IDs, frames and reference labels.
            fused.align(reference_fused,candidate_fused)
            row,objects,windows,values=analyze_sequence(variant,sequence,reference,candidate,reference_fused,candidate_fused)
            per_sequence.append(row);all_objects.extend(objects);all_windows.extend(windows)
            sources[variant][sequence]={'fused_scores':score.fingerprints,'windows':candidate.fingerprints}
            for name,value in values.items():measurements[variant].setdefault(name,[]).append(value)
    aggregate={};variant_rows=[]
    for variant in fused.VARIANTS:
        if variant not in variants:
            aggregate[variant]={'status':'not_evaluated_due_to_calibration_failure'};continue
        rows=[r for r in per_sequence if r['variant']==variant]
        counts={field:sum(r[field] for r in rows) for field in SUM_FIELDS}
        counts['net_window_matches']=counts['gained_window_matches']-counts['lost_window_matches']
        gate=protection_gate(counts['protected_denominator'],counts['protected_candidate_fused_losses'])
        aggregate[variant]={'status':'evaluated_at_frozen_threshold','threshold':calibration['variants'][variant]['threshold'],
                            'counts':counts,'protection_gate':gate,
                            'localization':{name:fused.distribution(np.concatenate(values)) for name,values in measurements[variant].items()}}
        variant_rows.append({'variant':variant,'threshold':calibration['variants'][variant]['threshold'],**counts,
                             'protection_gate_status':gate['status'],'protected_new_fused_loss_rate':gate['new_fused_loss_rate']})
    fused_ok=fused_summary['fused_prerequisite'].get('status')=='passed_fused_prerequisite'
    main=aggregate.get('paired_max',{})
    protection_ok=main.get('protection_gate',{}).get('passes',False)
    decision={'prespecified_main_candidate':'paired_max',
              'fused_condition_status':fused_summary['fused_prerequisite'].get('status'),
              'protection_condition_status':main.get('protection_gate',{}).get('status','not_evaluated'),
              'both_prespecified_conditions_met':bool(fused_ok and protection_ok),
              'status':'both_conditions_met' if fused_ok and protection_ok else 'conditions_not_met',
              'rule':'Paired_max must meet the frozen fused prerequisite AND <=1% protected new fused loss; no winner selection',
              'scope':'Single-seed pilot evidence only; this is not a claim of formal Pd/Fa or a complete literature comparison'}
    payload={'status':'context_confirmation_complete','created_utc':datetime.now(timezone.utc).isoformat(),
             'selection_sha256':fused.digest(args.selection),'calibration_sha256':fused.digest(cal_path),
             'fused_summary_sha256':fused.digest(fused_path),'summary_script_sha256':fused.digest(__file__),
             'confirmation_sequences':sequences,'reference_variant':'official','reference_threshold':.5,
             'window_starts':STARTS,'object_scope':'Original strict-small annotated object-frames covered by >=2 real windows',
             'definitions':{'support':'Original model has a point match in at least one covering real window; cross-window detectability, not a natural-causal context-benefit label',
                 'stable':'Original point match in all covering windows, at least two windows',
                 'protected':'Stable AND original official fused point match at 0.5',
                 'protected_new_fused_loss':'Protected object-frame not matched by candidate official fusion at its frozen threshold',
                 'partial_to_all_unmatched':'Original matches some but not all windows; candidate matches none. Consistency through collapse, not improvement',
                 'aggregation':'Target-frames and window-target-frame pairs are correlated; counts are descriptive, not independent trials',
                 'peak_span':'Maximum pairwise Euclidean displacement across matched windows; undefined with fewer than two matched windows'},
             'variants':aggregate,'prespecified_main_candidate_conditions':decision,'source_fingerprints':sources,
             'limitations':['Unmatched components and point matches are diagnostic, not official Pd/Fa',
                            'No ground-truth-window deployment selection and no confirmation threshold re-selection',
                            'Existing 97/24/42/778 development strata are separate evidence and are not included in this new confirmation scope',
                            'No natural-causal context contribution or independently verified target visibility is inferred from these strata']}
    args.output.mkdir(parents=True,exist_ok=True)
    fused.write_csv(args.output/'CONTEXT_SEQUENCES.csv',per_sequence)
    fused.write_csv(args.output/'CONTEXT_WINDOWS.csv',all_windows)
    fused.write_csv(args.output/'CONTEXT_VARIANTS.csv',variant_rows)
    if all_objects:fused.write_csv(args.output/'CONTEXT_OBJECTS.csv',all_objects)
    else:fused.write_csv(args.output/'CONTEXT_OBJECTS.csv',[],fields=['variant','sequence','frame','annotation_id','strict_small'])
    payload['output_sha256']={name:fused.digest(args.output/name) for name in
                             ['CONTEXT_SEQUENCES.csv','CONTEXT_WINDOWS.csv','CONTEXT_VARIANTS.csv','CONTEXT_OBJECTS.csv']}
    fused.write_json(destination,payload,exclusive=True)
    print(json.dumps({'summary':str(destination),'conditions':decision}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-root',type=Path,required=True)
    parser.add_argument('--context-root',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--fused-summary',type=Path)
    args=parser.parse_args();run(args)


if __name__=='__main__':main()
