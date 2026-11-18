"""Run author test.main unchanged on the preregistered Sequence85, with passive export."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
from tep_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    before=source_check();seed()
    assert torch.cuda.is_available()
    x=torch.randn(8,8,device='cuda');witness=float((x@x).sum().item());print('WITNESS_OK',witness,torch.cuda.get_device_name())
    seed()
    spec=importlib.util.spec_from_file_location('tep_author_test',SOURCE/'test.py');test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
    author_load=test.load_state_dict; holder={}
    def wrapped_load(model,path,device):
        assert sha(path)==CHECKPOINT_SHA
        author_load(model,path,device)
        model.requires_grad_(False)
        holder['state_before']=state_hash(model);holder['model']=model
        rec=Recorder(model,(200,300),100);holder['recorder']=rec
        original=model.infer
        model.infer=lambda *a,**kw:rec.invoke(original,*a,**kw)
        torch.cuda.reset_peak_memory_stats()
    test.load_state_dict=wrapped_load
    # Verify that the exact native metric receives the exported native binary maps.
    original_metric=test.SigmoidMetric
    class CheckedMetric(original_metric):
        def __init__(self,*a,**kw): super().__init__(*a,**kw);self.index=0
        def update(self,pred,gt):
            saved=holder['recorder'].prob[self.index]
            assert np.array_equal(pred,(saved>.5).astype(np.uint8))
            self.index+=1
            return super().update(pred,gt)
        def get(self):
            values=super().get();holder['native_iou_raw']=[float(x) for x in values];holder['metric_checked_frames']=self.index
            return values
    test.SigmoidMetric=CheckedMetric
    sys.argv=[str(SOURCE/'test.py'),'--config',str(STAGE/'runtime/native_config.yaml'),'--model',str(CHECKPOINT),'--batch-size','1','--num-workers','0','--bin-thresh','0.5']
    start=time.perf_counter();test.main();wall=time.perf_counter()-start
    model=holder['model'];rec=holder['recorder'];stats=rec.finish(out,range(100),range(1,101))
    assert holder['state_before']==state_hash(model)
    assert holder['metric_checked_frames']==100
    stats.update({'status':'PASS_NATIVE_RUNNING_CHECK_ONLY','sequence':'Sequence85','frame_count':100,'scope':'author native entrypoint on one preselected sequence; not full paper reproduction',
                  'native_test_unchanged':True,'metric_export_equality_frames':100,'native_iou_raw':holder['native_iou_raw'],
                  'checkpoint_sha256':CHECKPOINT_SHA,'parameters':sum(p.numel() for p in model.parameters()),'state_unchanged':True,
                  'wall_seconds_native_loop':wall,'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
                  'seed':46,'dtype':'float32','gpu':torch.cuda.get_device_name(),'kernel_witness':witness,'source_before':before,'source_after':source_check(),
                  'auxiliary_adapter':'author postprocess_masks(256 logits -> bilinear1024 -> crop -> bilinear1024), sigmoid, area original size; project diagnostic only'})
    json.dump(stats,open(out/'COMPLETE.json','w'),indent=2)
    print(json.dumps({k:v for k,v in stats.items() if k!='forward_seconds'},indent=2))
if __name__=='__main__':main()
