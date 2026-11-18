#!/usr/bin/env python3
"""Check full public data against the frozen experiment subset without running models."""
import json, hashlib, datetime, collections
from pathlib import Path
B=Path('/home/ly/DeepProPlus_TLFP_Code')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    result={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RUNNING'}
    for name in ['NUDT-MIRSDT','VT-Tiny-MOT']:
        record=json.loads((B/'provenance'/('PUBLIC_ARCHIVE_'+name+'.json')).read_text())
        assert record['status']=='COMPLETE' and record['full_zip_crc']=='PASS'
    selection=json.loads((B/'results/references/DATA_SELECTION.json').read_text())
    subset_annotations={};subset_images={};image_checks=[]
    for clip in selection['clips']:
        local=B/'data/stage1'/clip['sequence']
        annotations=json.loads((local/'annotations.json').read_text())
        partition=clip['archive_partition'];a=subset_annotations.setdefault(partition,{});im=subset_images.setdefault(partition,{})
        for row in annotations['annotations']:
            assert row['id'] not in a;a[row['id']]=row
        for row in annotations['images']:
            assert row['id'] not in im;im[row['id']]=row
        for member in clip['selected_members']:
            original=local/'source_archive'/member;full=B/'data/public'/member
            assert original.stat().st_size==full.stat().st_size and sha(original)==sha(full)
        image_checks.append({'sequence':clip['sequence'],'split':clip['split'],'compared_original_jpegs':len(clip['selected_members']),'all_exact':True})
    result['stage1_original_jpeg_checks']=image_checks;result['stage1_original_jpeg_count']=sum(x['compared_original_jpegs'] for x in image_checks)
    stats=[];annotation_checks=[]
    for path in sorted((B/'data/public/VT-Tiny-MOT/annotations').glob('*.json')):
        j=json.loads(path.read_text());stats.append({'file':str(path.relative_to(B)),'images':len(j.get('images',[])),'annotations':len(j.get('annotations',[])),'categories':len(j.get('categories',[]))})
        for partition,selected in subset_annotations.items():
            if path.name!='instances_01_'+partition+'.json':continue
            found=0
            for row in j['annotations']:
                if row['id'] in selected:assert row==selected[row['id']];found+=1
            assert found==len(selected)
            image_found=0
            for row in j['images']:
                if row['id'] in subset_images[partition]:assert row==subset_images[partition][row['id']];image_found+=1
            assert image_found==len(subset_images[partition])
            annotation_checks.append({'partition':partition,'annotations_equal_field_for_field':found,'image_records_equal_field_for_field':image_found})
        del j
    result['vt_annotation_statistics']=stats;result['stage1_subset_annotation_checks']=annotation_checks
    result['stage1_annotation_count']=sum(x['annotations_equal_field_for_field'] for x in annotation_checks)
    assert result['stage1_annotation_count']==70276
    full=B/'data/public/NUDT-MIRSDT';counts=collections.Counter();seqs=sorted(x for x in full.iterdir() if x.is_dir() and x.name.startswith('Sequence'))
    for seq in seqs:
        for category in ['images','masks','masks_centroid','Mix']:
            counts[category]+=sum(x.is_file() for x in (seq/category).iterdir())
    testlocal=B/'archive/infrared_problem_screen_20261009/official_reproduction/data/NUDT-MIRSDT';checked=0
    for local in testlocal.rglob('*.png'):
        fullfile=full/local.relative_to(testlocal);assert sha(local)==sha(fullfile);checked+=1
    train_list=(full/'train.txt').read_text().splitlines();test_list=(full/'test.txt').read_text().splitlines()
    train_seq={x.split('/')[0] for x in train_list};test_seq={x.split('/')[0] for x in test_list}
    result['nudt']={'sequences':len(seqs),'file_counts_by_role':dict(counts),'trajectory_preview_pngs':len(list((full/'trajectory').glob('*.png'))),'official_train_list_entries':len(train_list),'official_test_list_entries':len(test_list),'official_train_sequences':len(train_seq),'official_test_sequences':len(test_seq),'archive_sequences_not_in_these_two_lists':sorted({x.name for x in seqs}-train_seq-test_seq),'original_test_subset_pngs_exactly_equal':checked}
    assert len(seqs)==120 and all(counts[k]==12000 for k in ['images','masks','masks_centroid','Mix']) and checked==6000
    result['status']='PASS';out=B/'provenance/PUBLIC_DATA_VERIFICATION.json';out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
