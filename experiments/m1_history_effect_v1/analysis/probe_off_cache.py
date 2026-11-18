"""Isolate current-runtime Off semantics from old release-cache rounding."""
from pathlib import Path
import numpy as np
import torch
import runtime as rt
from trace import trace_window


def compare(probability,reference):
    delta=np.abs(probability-reference)
    index=np.unravel_index(np.argmax(delta),delta.shape)
    return dict(max_probability_difference=float(delta.max()),mean_probability_difference=float(delta.mean(dtype=np.float64)),
                threshold05_pixel_differences=int(np.count_nonzero((probability>=.5)!=(reference>=.5))),
                exact=bool(np.array_equal(probability,reference)),max_difference_tyx=list(map(int,index)),
                probability_at_max=float(probability[index]),reference_at_max=float(reference[index]))


def main():
    output=rt.E/'analysis/RELEASE_OFF_PARITY.json'
    if output.exists():raise FileExistsError(output)
    model,training,_,_=rt.v1_infer().load_trained('History')
    before=rt.old().tensor_hash(model.state_dict())
    record=dict(status='RUNNING',checkpoint_sha256=rt.sha(rt.CHECKPOINT),source_sha256=rt.source_hashes(),
                probe_source_sha256=rt.sha(__file__),torch=str(torch.__version__),cuda=torch.version.cuda,sequences={})
    rt.save(output,record)
    for seq in rt.SEQS:
        print(seq,flush=True)
        raw=np.load(rt.DATA/seq/'raw_frames.npy',mmap_mode='r')
        x=torch.from_numpy((raw[:40].astype(np.float32)-85.0799)/47.4845)[None,None].cuda()
        with torch.no_grad():
            expected=rt.old().frozen_f().runtime.tiled_logits(model.base,x,core=128,halo=16)
            trace=trace_window(model,x,[[] for _ in range(40)],sequence=seq,start=0)
            actual=trace['logits']['HistoryOff']
            logit_error=float((actual-expected).abs().max())
            assert torch.equal(actual,expected),logit_error
            p_off=actual.cuda().sigmoid()[0].cpu().numpy()
            p_current_baseline=expected.cuda().sigmoid()[0].cpu().numpy()
            p_history=trace['logits']['History'].cuda().sigmoid()[0].cpu().numpy()
        release=rt.Q/'stage1_goal/predictions/official'/seq
        release_receipt=rt.read_json(release/'COMPLETE.json')
        release_path=release/'windows/window_000.npz'
        assert rt.sha(release_path)==release_receipt['window_sha256'][release_path.name]
        with np.load(release_path) as r:p_release=r['probabilities']
        history=rt.V1/'predictions/History/step_2048'/seq
        history_receipt=rt.read_json(history/'COMPLETE.json')
        history_path=history/'windows/window_000.npz'
        assert rt.sha(history_path)==history_receipt['window_sha256'][history_path.name]
        with np.load(history_path) as r:p_history_cached=r['probabilities']
        a,b,c=compare(p_off,p_release),compare(p_current_baseline,p_release),compare(p_history,p_history_cached)
        assert a==b and a['threshold05_pixel_differences']==0
        assert c['exact']
        record['sequences'][seq]=dict(status='PASS',window_start=0,full_spatial_shape=[512,640],
            Off_vs_unmodified_current_runtime_tiled_logits_exact=True,Off_logit_max_difference=logit_error,
            Off_vs_historical_release_cache=a,unmodified_current_baseline_vs_release_cache=b,
            History_vs_frozen_v1_cache=c,release_runtime={k:release_receipt[k] for k in ('torch','cuda','precision','tile_core','tile_halo','tiling_checks')},
            source_receipts_sha256={str(path):rt.sha(path) for path in (release/'COMPLETE.json',release_path,history/'COMPLETE.json',history_path)})
        rt.save(output,record)
        del x,trace,expected,actual,p_off,p_current_baseline,p_history
        torch.cuda.empty_cache()
    assert before==rt.old().tensor_hash(model.state_dict())
    record.update(status='PASS',all_model_state_unchanged=True,
                  conclusion='Off exactly matches unmodified baseline executed in current runtime; old release-cache difference is reproduced without M1 and has zero threshold05 pixel differences on these two first windows.')
    rt.save(output,record)
    print(str(output),flush=True)


if __name__=='__main__':main()
