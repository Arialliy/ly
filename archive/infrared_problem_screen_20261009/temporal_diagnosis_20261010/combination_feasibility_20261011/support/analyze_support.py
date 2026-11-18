#!/usr/bin/env python3
"""Frozen window support feasibility. No inference; no substitution for fused detection."""
from pathlib import Path
import collections, csv, hashlib, itertools, json, math, time

OUT=Path(__file__).resolve().parent
Q=OUT.parent.parent
SEQS=['DJI_0026_2','DJI_0067_2']
STARTS=list(range(0,121,12))

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def key(r):return (r['sequence'],int(r['frame']),int(r['annotation_id']))
def signature(mask):return [s for j,s in enumerate(STARTS) if mask>>j&1]
def fullcoverage(mask):
    starts=signature(mask)
    return bool(starts and starts[0]==0 and starts[-1]==120 and all(b-a<=40 for a,b in zip(starts,starts[1:])))
def histogram(values):return dict(sorted(collections.Counter(values).items()))
def bcount(x):return x.bit_count()
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    tic=time.time()
    src=[Q/'problem_validity_20261011/OBJECTS.csv',Q/'problem_validity_20261011/WINDOWS.csv',Q/'failure_attribution_20261011/stages/ERROR_STAGE_TABLE.csv',OUT.parent/'LEGAL_COMBINATIONS.json',OUT.parent/'PROTOCOL.md',Q/'analysis/analyze.py',Q/'stage1_goal/pilot/score_pilot.py']
    hashes={str(p):hashfile(p) for p in src}
    objects=sorted([r for r in read(src[0]) if r['sequence'] in SEQS and r['small_geometry']=='1'],key=key)
    assert len(objects)==5200
    idx={key(r):i for i,r in enumerate(objects)};assert len(idx)==5200
    stages={key(r):r for r in read(src[2])};assert len(stages)==5200
    matrices=[[-1]*11 for _ in objects];responses=[[None]*11 for _ in objects]
    peaks=[[None]*11 for _ in objects]
    for r in read(src[1]):
        k=key(r)
        if k not in idx:continue
        i=idx[k];j=STARTS.index(int(r['window_start']));assert matrices[i][j]==-1
        assert STARTS[j]<=k[1]<STARTS[j]+40
        matrices[i][j]=int(r['matched']);responses[i][j]=float(r['box_max'])
        peaks[i][j]=[float(r['peak_x']),float(r['peak_y'])] if r['peak_x'] else None
    identity=['sequence','frame','source_frame','annotation_id','track_id','category_id','category','bbox_x','bbox_y','bbox_w','bbox_h']
    matchrows=[];boxrows=[];supportrows=[]
    windowbits=[0]*11;seqbits={s:0 for s in SEQS};cohortbits=collections.defaultdict(int)
    response_low=0;anysuccess=0
    for i,r in enumerate(objects):
        row={k:r[k] for k in identity};row={'row_index':i,**row}
        expected=[int(s<=int(r['frame'])<s+40) for s in STARTS]
        assert [int(m!=-1) for m in matrices[i]]==expected
        assert max(x for x in responses[i] if x is not None)==float(r['context_box_max'])
        assert any(m==1 for m in matrices[i])==bool(int(r['any11_matched']))
        mr=dict(row);br=dict(row)
        for j,s in enumerate(STARTS):mr[f'w{s:03d}']=matrices[i][j];br[f'w{s:03d}']='NA' if responses[i][j] is None else responses[i][j]
        matchrows.append(mr);boxrows.append(br)
        success=[s for s,m in zip(STARTS,matrices[i]) if m==1]
        smask=sum(1<<STARTS.index(s) for s in success)
        bit=1<<i;seqbits[r['sequence']]|=bit
        for j in range(11):
            if smask>>j&1:windowbits[j]|=bit
        stage=stages[key(r)];cohort=stage['cohort']
        cohortbits[cohort]|=bit
        if stage['official_matched']=='1':cohortbits['original_success']|=bit
        if stage['symmetric_matched']=='1':cohortbits['symmetric_success']|=bit
        if cohort=='gain' and stage['official_any_window_match']=='0' and stage['symmetric_any_window_match']=='1':cohortbits['gain_new_window_dependent']|=bit
        low=max(x for x in responses[i] if x is not None)<.5
        response_low+=low;anysuccess+=bool(success)
        run=longest=0;runs=0
        for m in matrices[i]:
            if m==1:
                if run==0:runs+=1
                run+=1;longest=max(longest,run)
            else:run=0
        supportrows.append(dict(row,cohort=cohort,original_matched=int(stage['official_matched']),symmetric_matched=int(stage['symmetric_matched']),new_window_dependent=int(bool(bit&cohortbits['gain_new_window_dependent'])),covering_window_count=sum(expected),success_window_count=len(success),success_starts=json.dumps(success),success_slots=json.dumps([int(r['frame'])-s for s in success]),success_mask=smask,sampled_start_span=max(success)-min(success) if success else '',sampled_success_runs=runs,longest_sampled_success_run=longest,max_candidate_box_response=max(x for x in responses[i] if x is not None),all_candidate_box_below_05=int(low)))
    assert response_low==4342 and anysuccess==809
    assert bcount(cohortbits['original_success'])==700
    assert bcount(cohortbits['loss'])==54 and bcount(cohortbits['gain_new_window_dependent'])==73
    allbits=(1<<5200)-1
    subsets=[];supports={0:0}
    for mask in range(2048):
        if mask:
            lowbit=mask&-mask;j=lowbit.bit_length()-1;supports[mask]=supports[mask^lowbit]|windowbits[j]
        sup=supports[mask]
        subsets.append(dict(mask=mask,window_count=bcount(mask),full_160_coverage=int(fullcoverage(mask)),starts=json.dumps(signature(mask)),supported=bcount(sup),original_success_supported=bcount(sup&cohortbits['original_success']),original_success_not_supported=bcount(cohortbits['original_success']&~sup),additional_supported=bcount(sup&~cohortbits['original_success']),loss54_supported=bcount(sup&cohortbits['loss']),gain74_supported=bcount(sup&cohortbits['gain']),gain73_new_supported=bcount(sup&cohortbits['gain_new_window_dependent']),**{f'{s}_supported':bcount(sup&seqbits[s]) for s in SEQS}))
    legal=[r for r in subsets if r['window_count']==5 and r['full_160_coverage']]
    expected=json.loads(src[3].read_text())['legal_full_coverage_5']
    legal=sorted(legal,key=lambda r:json.loads(r['starts']))
    assert [json.loads(r['starts']) for r in legal]==expected
    combos=[]
    for n,r in enumerate(legal):
        for s in ['pooled']+SEQS:
            bits=allbits if s=='pooled' else seqbits[s];sup=supports[r['mask']]&bits;old=cohortbits['original_success']&bits
            combos.append(dict(combo_id=n,sequence=s,starts=r['starts'],supported=bcount(sup),original_success_supported=bcount(sup&old),original_success_not_supported=bcount(old&~sup),additional_supported=bcount(sup&~old),loss54_supported=bcount(sup&cohortbits['loss']),gain74_supported=bcount(sup&cohortbits['gain']),gain73_new_supported=bcount(sup&cohortbits['gain_new_window_dependent'])))
    pairs=[]
    for a,b in itertools.product(range(10),repeat=2):
        sup=(supports[legal[a]['mask']]&seqbits[SEQS[0]])|(supports[legal[b]['mask']]&seqbits[SEQS[1]])
        pairs.append(dict(combo_0026=a,combo_0067=b,supported=bcount(sup),original_success_supported=bcount(sup&cohortbits['original_success']),original_success_not_supported=bcount(cohortbits['original_success']&~sup),additional_supported=bcount(sup&~cohortbits['original_success']),loss54_supported=bcount(sup&cohortbits['loss']),gain74_supported=bcount(sup&cohortbits['gain']),gain73_new_supported=bcount(sup&cohortbits['gain_new_window_dependent'])))
    targets={'original700':cohortbits['original_success'],'original700_plus_gain74':cohortbits['original_success']|cohortbits['gain'],'loss54_plus_gain73':cohortbits['loss']|cohortbits['gain_new_window_dependent'],'all809':supports[2047]}
    minrows=[]
    for label,target in targets.items():
        for seq in ['pooled_shared_arrangement']+SEQS:
            bits=allbits if seq=='pooled_shared_arrangement' else seqbits[seq];t=target&bits
            feasible=[r for r in subsets if r['full_160_coverage'] and not(t&~supports[r['mask']])]
            mink=min(r['window_count'] for r in feasible);winners=[r for r in feasible if r['window_count']==mink]
            for r in winners:minrows.append(dict(target=label,sequence=seq,target_count=bcount(t),min_support_windows=mink,starts=r['starts'],supported=r['supported'] if seq=='pooled_shared_arrangement' else bcount(supports[r['mask']]&bits)))
    replace={};patterns=[];mandatory=[]
    for label in ['loss','gain_new_window_dependent','gain','original_success']:
        rr=[r for i,r in enumerate(supportrows) if (cohortbits[label]>>i)&1]
        replace[label]={'count':len(rr),'success_count_histogram':histogram(r['success_window_count'] for r in rr),'coverage_count_histogram':histogram(r['covering_window_count'] for r in rr),'sampled_success_run_count_histogram':histogram(r['sampled_success_runs'] for r in rr),'sampled_span_histogram':histogram(r['sampled_start_span'] for r in rr),'success_start_histogram':histogram(s for r in rr for s in json.loads(r['success_starts'])),'unique_success_start_histogram':histogram(json.loads(r['success_starts'])[0] for r in rr if r['success_window_count']==1),'success_slot_histogram':histogram(s for r in rr for s in json.loads(r['success_slots']))}
        for pattern,n in collections.Counter(r['success_starts'] for r in rr).most_common():patterns.append(dict(cohort=label,success_starts=pattern,count=n))
    for label,target in targets.items():
        for seq in ['pooled_shared_arrangement']+SEQS:
            bits=allbits if seq=='pooled_shared_arrangement' else seqbits[seq];rr=[r for i,r in enumerate(supportrows) if ((target&bits)>>i)&1]
            forced=sorted({json.loads(r['success_starts'])[0] for r in rr if r['success_window_count']==1})
            mandatory.append(dict(target=label,sequence=seq,forced_by_single_success_starts=json.dumps(forced),count=len(forced)))
    witnesses=[]
    for seq in SEQS:
        for s in STARTS:
            rr=[r for r in supportrows if r['sequence']==seq and (r['cohort']=='loss' or r['new_window_dependent']) and r['success_window_count']==1 and json.loads(r['success_starts'])==[s]]
            if rr:
                r=rr[0];witnesses.append({k:r[k] for k in ['sequence','frame','source_frame','annotation_id','track_id','category','cohort','success_starts']})
                witnesses[-1]['same_sequence_unique_support_objects']=len(rr)
    write('FORCED_START_WITNESSES.csv',witnesses)
    # Pairwise incompatible demand sets under one shared five-window arrangement.
    bypat=collections.defaultdict(list)
    for i,r in enumerate(supportrows):
        if ((targets['original700_plus_gain74']>>i)&1):bypat[r['success_mask']].append(i)
    conflict=[]
    for p,q in itertools.combinations_with_replacement(sorted(bypat),2):
        if any((r['mask']&p) and (r['mask']&q) for r in legal):continue
        conflict.append(dict(first_success_starts=json.dumps(signature(p)),second_success_starts=json.dumps(signature(q)),first_objects=len(bypat[p]),second_objects=len(bypat[q]),shared_five_compatible=0))
    budget=[]
    for k in range(12):
        rr=[r for r in subsets if r['full_160_coverage'] and r['window_count']==k]
        if not rr:continue
        best=max(r['supported'] for r in rr)
        budget.append(dict(windows_per_clip=k,legal_shared_count=len(rr),max_pooled_shared_support=best,best_shared_starts=json.dumps([json.loads(r['starts']) for r in rr if r['supported']==best]),max_independent_per_clip_support=sum(max(bcount(supports[r['mask']]&seqbits[s]) for r in rr) for s in SEQS)))
    summary={'objects':5200,'sequences':{s:bcount(v) for s,v in seqbits.items()},'matrix_shape':[5200,11],'matrix_state_counts':histogram(v for row in matrices for v in row),'all_candidate_box_below_05':response_low,'loose_response_upper_bound':5200-response_low,'single_window_any_match':anysuccess,'tighter_peak_matching_upper_bound':809,'response_above_but_no_single_window_match':5200-response_low-anysuccess,'candidate_pool_max_net_gain':809-700,'candidate_pool_max_percentage_point_gain':100*(809-700)/5200,'support_only_not_actual_fusion':True,'all_subsets':2048,'full_coverage_subset_count':sum(r['full_160_coverage'] for r in subsets),'legal5':len(legal),'shared5_support_max':max(r['supported'] for r in legal),'independent5_support_max':max(r['supported'] for r in pairs),'shared5_max_support_preserving_all_original':max(r['supported'] for r in legal if r['original_success_not_supported']==0),'independent5_max_support_preserving_all_original':max(r['supported'] for r in pairs if r['original_success_not_supported']==0),'replaceability':replace,'min_support_budget':minrows,'mandatory':mandatory,'pairwise_incompatible_success_patterns':len(conflict),'budget_support':budget}
    write('MATCH_MATRIX.csv',matchrows);write('BOX_RESPONSE_MATRIX.csv',boxrows);write('OBJECT_SUPPORT.csv',supportrows)
    write('SUBSET_SUPPORT.csv',subsets);write('LEGAL5_SUPPORT.csv',combos);write('PER_CLIP_PAIR_SUPPORT.csv',pairs)
    write('MIN_SUPPORT_BUDGET.csv',minrows);write('SUCCESS_PATTERNS.csv',patterns);write('MANDATORY_STARTS.csv',mandatory)
    if conflict:write('PAIRWISE_CONFLICTS.csv',conflict)
    write('SUPPORT_BUDGET_FRONTIER.csv',budget);save('SUMMARY.json',summary)
    for p in src:assert hashfile(p)==hashes[str(p)]
    save('VERIFICATION.json',{'status':'PASS','input_hashes':hashes,'matrix_rows_unique':True,'coverage_state_exact':True,'responses_equal_original_summary':True,'any_match_equal_original_summary':True,'legal5_exact_geometry_match':True,'all_source_unchanged':True,'elapsed_seconds':time.time()-tic})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['replaceability','min_support_budget','mandatory','budget_support']},indent=2))
    print(json.dumps({'min_support_budget':minrows,'budget_support':budget,'replaceability':replace},indent=2))

if __name__=='__main__':main()
