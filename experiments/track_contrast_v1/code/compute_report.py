"""Report measured resource use without treating thermal wall time as model speed."""
import csv
import json
from runtime import E,SEQS,sha,save_json


def main():
    rows=[];sources={}
    for tag in ['prepare_0','prepare_1','prepare_2','calibration','formal_A','formal_B','formal_C']:
        path=E/'runtime'/f'{tag}.json';r=json.loads(path.read_text())
        assert r['status']=='COMPLETE';sources[str(path)]=sha(path)
        rows.append(dict(stage=tag,condition='',sequence='',physical_gpu=r.get('gpu'),
            seconds=r['end_unix']-r['start_unix'],measurement='GPU_assigned_container_wall_includes_CPU_IO_and_pauses',peak_memory_bytes=''))
    for condition,gpu in zip('ABC',range(3)):
        p=E/'training'/condition/'run/RUN.json';r=json.loads(p.read_text());sources[str(p)]=sha(p)
        rows.append(dict(stage='training',condition=condition,sequence='',physical_gpu=gpu,
            seconds=r['elapsed_seconds'],measurement='optimization_loop_wall_includes_CPU_IO_and_pauses',
            peak_memory_bytes=r['max_cuda_memory_allocated_bytes']))
        for sequence in SEQS:
            p=E/'predictions'/condition/sequence/'COMPLETE.json';r=json.loads(p.read_text());sources[str(p)]=sha(p)
            for arrangement in ['O','S']:
                rows.append(dict(stage='inference_'+arrangement,condition=condition,sequence=sequence,physical_gpu=gpu,
                    seconds=r[arrangement+'_model_seconds'],measurement='five_windows_model_plus_CPU_copy_not_full_deployment',
                    peak_memory_bytes=max(t['peak_cuda_allocated_bytes'] for t in r['window_timing'])))
    with (E/'COMPUTE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(E/'COMPUTE.json',dict(status='COMPLETE',source_sha256=sources,
        measurements_csv_sha256=sha(E/'COMPUTE.csv'),
        warning='Conditions use different physical cards and observed thermal throttling. Wall times are resource accounting, not evidence of relative method efficiency. Shared preparation is counted once; do not sum formal container plus its training/inference subrows.',
        inference_architecture='all three original unwrapped 70913-parameter DeepPro-Plus; no auxiliary inference module',
        formal_exposure_per_condition={'updates':2048,'detection_windows':8192,'auxiliary_windows':8192,
            'frames_per_window':40,'input_spatial_pixels':[128,128],'temporal_pairs':121904},
        note='A reads identical windows and uses keep; its optimized contrast pair count is zero. B/C each optimize 121904 directed pairs.'))

if __name__=='__main__':main()
