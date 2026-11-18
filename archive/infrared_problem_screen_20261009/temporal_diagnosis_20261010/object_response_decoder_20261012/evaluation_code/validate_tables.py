#!/usr/bin/env python3
"""Independent stdlib recomputation of delivered tables, not raw-map rescoring."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    folder = args.directory.resolve()
    receipt = json.loads((folder/'SUMMARY_COMPLETE.json').read_text())
    tables = {name: list(csv.DictReader((folder/(name+'.csv')).open())) for name in [
        'SUMMARY', 'SEQUENCE_SUMMARY', 'OBJECT_COMPARISON', 'FRAME_COUNTS',
        'TRACK_SUMMARY', 'WINDOW_DEPENDENCE', 'COHORT_TRANSITIONS', 'CONTINUOUS_MISS_TRACKS', 'CONTINUOUS_MISS_SUMMARY', 'RELEASE_MISS_SEGMENTS']}
    checks = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    conditions = receipt['conditions']
    objects = tables['OBJECT_COMPARISON']
    objkey = lambda row: (row['sequence'], row['frame'], row['annotation_id'])
    check('status and threshold', receipt['status'] == 'COMPLETE' and receipt['threshold'] == .5)
    check('objects 10400 unique', len(objects) == len({objkey(r)+(r['arrangement'],) for r in objects}) == 10400)
    expected_rows = {'SUMMARY': 2, 'SEQUENCE_SUMMARY': 4, 'FRAME_COUNTS': 640,
                     'TRACK_SUMMARY': 260, 'WINDOW_DEPENDENCE': 27, 'COHORT_TRANSITIONS': 60*len(receipt['comparisons'])}
    for name, per_condition in expected_rows.items():
        check(name+' row count', len(tables[name]) == per_condition*len(conditions))
    for path, expected in receipt['source_sha256'].items():
        check('input hash '+path, digest(path) == expected)
    anchor = Path(receipt['frozen_reference_anchor'])
    check('anchor hash', digest(anchor) == receipt['frozen_reference_anchor_sha256'])
    for path, expected in json.loads(anchor.read_text())['source_sha256'].items():
        check('frozen hash '+path, digest(path) == expected)

    frozen_counts = {'low_all_05':4342, 'extreme_all_001':4170, 'both_fixed_success':646,
                     'extra_only':104, 'new_window_dependent':73, 'release_O_only':54,
                     'release_S_only':74, 'candidate_window_mixed':286}
    for plan in ['O','S']:
        obj = [r for r in objects if r['arrangement'] == plan]
        for cohort, count in frozen_counts.items():
            check(plan+' frozen '+cohort, sum(int(r[cohort]) for r in obj) == count)

    costs = ['gt_count','matched_count','component_count','unmatched_components',
             'positive_pixels','positive_pixels_outside_boxes']
    for row in tables['SUMMARY']+tables['SEQUENCE_SUMMARY']:
        condition, plan, group = row['condition'], row['arrangement'], row['group']
        obj = [r for r in objects if r['arrangement'] == plan and (group == 'pooled' or r['sequence'] == group)]
        tag = condition+'/'+plan+'/'+group
        m = lambda r, c=condition: int(r[c+'_matched'])
        check(tag+' M', sum(map(m,obj)) == int(row['M']))
        for reference in receipt['comparisons']:
            gain = sum(m(r) and not m(r,reference) for r in obj)
            loss = sum(not m(r) and m(r,reference) for r in obj)
            check(tag+' differences '+reference,
                  gain == int(row['G_vs_'+reference+'_same']) and loss == int(row['L_vs_'+reference+'_same']))
        selected_frames = [r for r in tables['FRAME_COUNTS'] if r['condition'] == condition and r['arrangement'] == plan
                           and (group == 'pooled' or r['sequence'] == group)]
        for field in costs:
            check(tag+' frame sum '+field, sum(int(r[field]) for r in selected_frames) == int(row[field]))
        check(tag+' unmatched area', sum(int(r['unmatched_component_area']) for r in selected_frames)
              == int(row['unmatched_component_area_sum']))
        for cohort in frozen_counts:
            selected = [r for r in obj if int(r[cohort])]
            matched = [r for r in selected if m(r)]
            check(tag+' cohort '+cohort, len(matched) == int(row[cohort+'_matched'])
                  and len({(r['sequence'],r['track_id']) for r in matched}) == int(row[cohort+'_matched_tracks']))

    for row in tables['COHORT_TRANSITIONS']:
        condition, plan, group, cohort = (row[k] for k in ['condition','arrangement','group','cohort'])
        obj = [r for r in objects if r['arrangement'] == plan and (group == 'pooled' or r['sequence'] == group)
               and (cohort == 'all' or int(r[cohort]))]
        partition = {'both_matched': [], 'gain': [], 'loss': [], 'neither_matched': []}
        for r in obj:
            before, after = int(r[row['reference']+'_matched']), int(r[condition+'_matched'])
            label = {(1,1):'both_matched',(0,1):'gain',(1,0):'loss',(0,0):'neither_matched'}[before,after]
            partition[label].append(r)
        valid = int(row['target_frames']) == len(obj)
        for label, selected in partition.items():
            valid &= (int(row[label]) == len(selected)
                      and int(row[label+'_tracks']) == len({(r['sequence'],r['track_id']) for r in selected})
                      and int(row[label+'_clips']) == len({r['sequence'] for r in selected}))
        check('set partition '+condition+'/'+row['reference']+'/'+plan+'/'+group+'/'+cohort, valid)

    for row in tables['WINDOW_DEPENDENCE']:
        condition, group, cohort = (row[k] for k in ['condition','group','cohort'])
        selected = [r for r in objects if (group == 'pooled' or r['sequence'] == group)
                    and (cohort == 'all' or int(r[cohort]))]
        o = {objkey(r):int(r[condition+'_matched']) for r in selected if r['arrangement'] == 'O'}
        s = {objkey(r):int(r[condition+'_matched']) for r in selected if r['arrangement'] == 'S'}
        both = sum(o[k] and s[k] for k in o)
        o_only = sum(o[k] and not s[k] for k in o)
        s_only = sum(not o[k] and s[k] for k in o)
        neither = sum(not o[k] and not s[k] for k in o)
        check('window '+condition+'/'+group+'/'+cohort,
              o.keys() == s.keys() and len(o) == int(row['target_frames'])
              and [both,o_only,s_only,neither] == [int(row[k]) for k in ['both_matched','O_only','S_only','neither_matched']])

    for row in tables['SUMMARY']:
        selected = [r for r in tables['TRACK_SUMMARY'] if r['condition'] == row['condition'] and r['arrangement'] == row['arrangement']]
        fields = ['M'] + [side+'_vs_'+ref+'_same' for ref in receipt['comparisons'] for side in ['G','L']]
        for cohort in frozen_counts:
            fields += [cohort+'_matched']
        check('track sums '+row['condition']+'/'+row['arrangement'],
              all(sum(int(t[k]) for t in selected) == int(row[k]) for k in fields))

    # Recompute runs directly from the wide object table without importing producer.
    def run_lengths(rows,condition):
        lengths=[]; length=0; previous=None
        for item in sorted(rows,key=lambda v:(int(v['frame']),int(v['source_frame']))):
            contiguous=(previous is None or (int(item['frame'])==int(previous['frame'])+1
                        and int(item['source_frame'])==int(previous['source_frame'])+1))
            if not contiguous:
                if length:lengths.append(length)
                length=0
            if int(item[condition+'_matched'])==0:length+=1
            else:
                if length:lengths.append(length)
                length=0
            previous=item
        if length:lengths.append(length)
        return lengths

    for row in tables['CONTINUOUS_MISS_TRACKS']:
        condition,plan,cohort=(row[k] for k in ['condition','arrangement','cohort'])
        selected=[r for r in objects if r['arrangement']==plan and r['sequence']==row['sequence']
                  and r['track_id']==row['track_id'] and (cohort=='all' or int(r[cohort]))]
        lengths=run_lengths(selected,condition)
        check('continuous track '+condition+'/'+plan+'/'+row['sequence']+'/'+row['track_id']+'/'+cohort,
              len(selected)==int(row['target_frames']) and sum(lengths)==int(row['missed_frames'])
              and len(lengths)==int(row['miss_run_count']) and max(lengths,default=0)==int(row['longest_miss_run'])
              and lengths==json.loads(row['miss_run_lengths']))
        for ref in receipt['comparisons']:
            before=run_lengths(selected,ref)
            check('continuous delta '+condition+'/'+plan+'/'+row['sequence']+'/'+row['track_id']+'/'+cohort+'/'+ref,
                  max(lengths,default=0)-max(before,default=0)==int(row['longest_miss_delta_vs_'+ref]))

    for row in tables['RELEASE_MISS_SEGMENTS']:
        selected=[r for r in objects if r['arrangement']==row['arrangement'] and r['sequence']==row['sequence']
                  and r['track_id']==row['track_id'] and int(row['frame_start'])<=int(r['frame'])<=int(row['frame_end'])]
        check('release segment '+row['arrangement']+'/'+row['sequence']+'/'+row['track_id']+'/'+row['release_miss_segment'],
              len(selected)==int(row['target_frames']) and not any(int(r['release_matched']) for r in selected)
              and int(row['frame_end'])-int(row['frame_start'])+1==len(selected)
              and int(row['source_frame_end'])-int(row['source_frame_start'])+1==len(selected))
        for condition in conditions:
            lens=run_lengths(selected,condition)
            check('release segment outcome '+condition+'/'+row['arrangement']+'/'+row['sequence']+'/'+row['track_id']+'/'+row['release_miss_segment'],
                  max(lens,default=0)==int(row[condition+'_remaining_miss_longest'])
                  and sum(int(r[condition+'_matched']) for r in selected)==int(row[condition+'_recovered_frames']))

    result = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), checks_passed=len(checks),
                  conditions=conditions, new_models_evaluated=receipt['new_models_evaluated'],
                  rows={name:len(rows) for name,rows in tables.items()}, checks=checks,
                  scope='Independent stdlib recomputation of table sums, true-set partitions and frozen input hashes; not raw-map rescoring.',
                  validator_sha256=digest(__file__),
                  summary_receipt_sha256=digest(folder/'SUMMARY_COMPLETE.json'))
    (folder/'TABLE_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','checks_passed','conditions','new_models_evaluated','rows']}))


if __name__ == '__main__':
    main()
