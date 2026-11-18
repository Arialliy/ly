"""One implementation smoke test; cropped inputs are not research evidence."""
import numpy as np
import torch
import runtime as rt
from trace import trace_window, STAGES, _stage_statistics


def main():
    output = rt.E / 'checks/TRACE_SMOKE.json'
    if output.exists():
        raise FileExistsError(output)
    assert rt.sha(rt.CHECKPOINT) == rt.CHECKPOINT_SHA
    model, training, checkpoint, _ = rt.v1_infer().load_trained('History')
    model.eval()
    before = rt.old().tensor_hash(model.state_dict())
    raw = np.load(rt.DATA / rt.SEQS[0] / 'raw_frames.npy', mmap_mode='r')
    frames = np.array((raw[:40,:160,:192].astype(np.float32)-85.0799)/47.4845,copy=True)
    x = torch.from_numpy(frames)[None,None].cuda()
    regions = [[dict(id=f'smoke_{t}_{j}',kind='test_only',x1=a,y1=b,x2=c,y2=d,
                     groups=['implementation_only'],anchor_x=a,anchor_y=b)
                for j,(a,b,c,d) in enumerate([(0,0,4,3),(125,126,132,134),(175,150,192,160)])]
               for t in range(40)]
    result = trace_window(model,x,regions,sequence=rt.SEQS[0],start=0)
    with torch.no_grad():
        expected_history = model.forward_full_spatial(x)[1].cpu()
        original_mode = model.mode
        try:
            model.mode = 'current_only'
            expected_current = model.forward_full_spatial(x)[1].cpu()
        finally:
            model.mode = original_mode
        expected_off = model.base(x)[1].cpu()
    expected = {'History':expected_history,'HistoryCurrent':expected_current,'HistoryOff':expected_off}
    errors = {key:float((result['logits'][key]-value).abs().max()) for key,value in expected.items()}
    assert all(torch.equal(result['logits'][key],value) for key,value in expected.items()),errors
    assert torch.equal(result['logits']['History'][:,0],result['logits']['HistoryCurrent'][:,0])
    assert all(value < 1e-4 for value in result['metadata']['maximum_identity_residual'].values())
    assert len(result['frame_rows']) == 40 and len(result['roi_rows']) == 120
    # Verify ROI coordinate, peak and mean extraction against direct CPU slices.
    for row in result['roi_rows']:
        for key,value in expected.items():
            a = value[row['batch_index'],row['window_t'],row['y1']:row['y2'],row['x1']:row['x2']].numpy()
            assert row[key+'_z_max'] == float(a.max())
            assert abs(row[key+'_z_mean']-float(a.mean(dtype=np.float64))) < 1e-10
    # Independent direct ROI reduction verifies moment integral-image indexing.
    rng = np.random.RandomState(46)
    maps = rng.rand(len(STAGES),5,2,7,9).astype(np.float32)
    rois = [dict(batch_index=1,x1=2,y1=1,x2=6,y2=5)]
    _, values = _stage_statistics(maps,rois)
    for si,stage in enumerate(STAGES):
        expected_mean = float(maps[si,0,1,1:5,2:6].mean(dtype=np.float64))
        expected_rms = float(np.sqrt(maps[si,1,1,1:5,2:6].mean(dtype=np.float64)))
        assert abs(values[0][stage]['delta_mean']-expected_mean) < 1e-12
        assert abs(values[0][stage]['delta_rms']-expected_rms) < 1e-12
    assert before == rt.old().tensor_hash(model.state_dict())
    record = {'status':'PASS','scope':'Implementation test only; no cropped-window research inference',
              'checkpoint_sha256':rt.sha(checkpoint),'source_sha256':rt.source_hashes(),
              'logit_errors_vs_original_execution':errors,'all_three_modes_exact':True,
              'first_frame_history_current_exact':True,'all_weights_and_buffers_unchanged':True,
              'direct_roi_peak_and_mean_check':'PASS','integral_moment_roi_check':'PASS',
              'trace_metadata':result['metadata']}
    rt.save(output,record)
    print(record,flush=True)


if __name__ == '__main__':
    main()
