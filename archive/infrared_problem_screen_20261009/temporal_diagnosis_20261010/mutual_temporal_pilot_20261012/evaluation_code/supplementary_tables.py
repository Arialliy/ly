"""Cost and coarse information-source summaries from immutable inference receipts."""
import collections
import csv
import hashlib
import json

PLANS={'O':[0,36,72,108,120],'S':[0,24,60,96,120]}
CATEGORIES=['same_track_region','other_target_region','outside_all_annotated_boxes','uncertain']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvwrite(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def supplementary_tables(base,conditions,objects,output):
    costs=[];sources={}
    for c in conditions:
        if not c.startswith('E'):continue
        for seq in ['DJI_0026_2','DJI_0067_2']:
            p=base/'predictions'/c/seq/'COMPLETE.json';r=json.loads(p.read_text());sources[str(p)]=sha(p)
            for plan in PLANS:
                timing=[w for w in r['window_timing'] if w['start'] in PLANS[plan]]
                costs.append(dict(condition=c,sequence=seq,arrangement=plan,parameters=r['parameters'],
                     model_plus_cpu_copy_seconds=r[plan+'_model_seconds'],five_windows=5,
                     peak_cuda_allocated_bytes=r['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=r['peak_cuda_reserved_bytes'],
                     peak_includes_provenance_capture=r['read_only_query_labels_access'],
                     provenance_seconds=sum(w.get('provenance_seconds',0) for w in timing),
                     all_eight_windows_wall_seconds=r['elapsed_seconds'],tile_core=r['tile_core'],tile_halo=r['tile_halo'],
                     model_timing_scope='GPU synchronization around tiled prediction and CPU copy; sparse provenance hook time subtracted. File compression excluded. Peak memory includes capture if enabled.'))
    csvwrite(output/'COMPUTE_COST.csv',costs)
    if 'E2' not in conditions:return
    object_lookup={(r['sequence'],int(r['annotation_id']),r['arrangement']):r for r in objects}
    grouped=collections.defaultdict(list)
    for seq in ['DJI_0026_2','DJI_0067_2']:
        folder=base/'predictions/E2'/seq;p=folder/'PROVENANCE_COMPLETE.json'
        receipt=json.loads(p.read_text());assert receipt['status']=='COMPLETE';sources[str(p)]=sha(p)
        path=folder/'PROVENANCE_ROWS.csv';assert sha(path)==receipt['rows_sha256'];sources[str(path)]=sha(path)
        with path.open() as f:rows=list(csv.DictReader(f))
        for r in rows:
            for plan,starts in PLANS.items():
                if int(r['window_start']) in starts:grouped[seq,int(r['annotation_id']),plan].append(r)
    measures=['null_probability','forward_nonnull_mass','mutual_nonnull_mass']+[pre+'_'+cat for pre in ['forward','mutual'] for cat in CATEGORIES]
    aggregate=[]
    for key,rr in sorted(grouped.items()):
        obj=object_lookup[key]
        row={k:obj[k] for k in ['sequence','track_id','frame','source_frame','annotation_id','arrangement','low_all_05','extreme_all_001','both_fixed_success']}
        row['query_reference_observations']=len(rr)
        row['distinct_windows']=len({r['window_start'] for r in rr})
        for m in measures:row[m]=sum(float(r[m]) for r in rr)/len(rr)
        for c in ['release','E0','E1','E2','E2_mass','E2_current']:
            if c in conditions:row[c+'_matched']=obj[c+'_matched']
        for pre in ['forward','mutual']:
            for cat in CATEGORIES:
                row[pre+'_'+cat+'_fraction']=row[pre+'_'+cat]/row[pre+'_nonnull_mass'] if row[pre+'_nonnull_mass'] else ''
        aggregate.append(row)
    assert len(aggregate)==9976
    summary=[]
    for plan in PLANS:
        for group in ['pooled','DJI_0026_2','DJI_0067_2']:
            for cohort in ['all_selected','low_all_05','extreme_all_001','both_fixed_success']:
                rr=[r for r in aggregate if r['arrangement']==plan and (group=='pooled' or r['sequence']==group)
                    and (cohort=='all_selected' or r[cohort])]
                if not rr:continue
                row=dict(condition='E2',arrangement=plan,group=group,cohort=cohort,target_frames=len(rr),
                         tracks=len({(r['sequence'],r['track_id']) for r in rr}),clips=len({r['sequence'] for r in rr}),
                         query_reference_observations=sum(r['query_reference_observations'] for r in rr))
                for m in measures:row[m+'_object_equal_mean']=sum(r[m] for r in rr)/len(rr)
                for pre in ['forward','mutual']:
                    total=sum(r[pre+'_nonnull_mass'] for r in rr)
                    for cat in CATEGORIES:
                        row[pre+'_'+cat+'_mass_fraction']=sum(r[pre+'_'+cat] for r in rr)/total if total else ''
                summary.append(row)
    csvwrite(output/'PROVENANCE_OBJECTS.csv',aggregate);csvwrite(output/'PROVENANCE_SUMMARY.csv',summary)
    info=dict(status='COMPLETE',source_sha256=sources,script_sha256=sha(__import__('pathlib').Path(__file__)),
              query_target_frames=4988,object_arrangement_rows=len(aggregate),
              scope='E2 forward versus mutual probability source mass at geometry-fixed centers; coarse annotations, no exact correspondence or causal credit. Average references/windows within each object-frame first; then object-frame equal aggregation.',
              category_note='Outside_all_annotated_boxes does not establish true background. Missing same-ID, coarse-box overlap, ignored/crowd and source-frame discontinuity are uncertain.',
              compute_note='Same five temporal windows and core/halo for all current models; model runtime differs. E2 telemetry time separately excluded from prediction timing; peak memory includes capture.')
    (output/'SUPPLEMENTARY_COMPLETE.json').write_text(json.dumps(info,indent=2)+'\n')
