"""Author frozen TEP-SAM: full-frame inference once per current frame, no GT prompts."""
import argparse,json,time
from pathlib import Path
from tep_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=['DJI_0026_2','DJI_0067_2'],required=True);p.add_argument('--output',required=True);p.add_argument('--native-receipt',required=True);p.add_argument('--protocol',required=True);args=p.parse_args()
    receipt=json.load(open(args.native_receipt));assert receipt['status']=='PASS_NATIVE_RUNNING_CHECK_ONLY'
    assert receipt['checkpoint_sha256']==CHECKPOINT_SHA
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    before=source_check();seed()
    assert torch.cuda.is_available()
    import datasets
    data_root=STAGE.parent/'stage1_goal/data';seqroot=data_root/args.sequence
    manifest=json.load(open(seqroot/'FRAME_MANIFEST.json'))
    frame_records=manifest['frames'];assert len(frame_records)==160
    names=[Path(f['source_file_name']).stem for f in frame_records]
    source_frames=[f['source_frame_id'] for f in frame_records]
    assert all(source_frames[i+1]==source_frames[i]+1 for i in range(159))
    seq_list=out/'sequence.txt';seq_list.write_text(args.sequence+'\n')
    spec=load_config()['test_dataset']
    spec['dataset']['args'].update(root_path_1=str(data_root),root_path_2=str(data_root),seq_list=str(seq_list),allow_missing_mask=True)
    base=datasets.make(spec['dataset']);assert len(base)==160
    assert [m['frame_name'] for m in base.metas]==names
    ds=datasets.make(spec['wrapper'],args={'dataset':base});assert len(ds)==160
    model=load_model();initial=state_hash(model);rec=Recorder(model,(512,640),160)
    temporal_indices=np.clip(np.arange(160)[:,None]+np.arange(-3,4)[None,:],0,159)
    assert temporal_indices.min()==0 and temporal_indices.max()==159
    json.dump({'sequence':args.sequence,'frame_indices':list(range(160)),'source_frames':source_frames,'context_local_indices':temporal_indices.tolist(),'context_source_frames':np.asarray(source_frames)[temporal_indices].tolist(),'no_annotation_prompts':True},open(out/'FRAME_ACCESS.json','w'),indent=2)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.no_grad():
        for i in range(160):
            batch=ds[i];assert (batch['orig_h'],batch['orig_w'])==(512,640)
            rec.invoke(model.infer,batch['inp'].unsqueeze(0).cuda(),boxes=None)
            if (i+1)%10==0:print(json.dumps({'sequence':args.sequence,'completed':i+1,'of':160}),flush=True)
    wall=time.perf_counter()-start
    stats=rec.finish(out,range(160),source_frames)
    assert initial==state_hash(model)
    stats.update({'status':'COMPLETE','protocol_path':str(Path(args.protocol).resolve()),'protocol_sha256':sha(args.protocol),'raw_sha256':sha(seqroot/'raw_frames.npy'),'annotations_sha256':sha(seqroot/'annotations.json'),'repository_commit':'c5165cfa9139e7d72f1fc962ddc398d31d59093e','auxiliary_head_trained_parameters_loaded':True,'sequence':args.sequence,'checkpoint_sha256':CHECKPOINT_SHA,'parameters':sum(p.numel() for p in model.parameters()),'state_unchanged':True,'wall_seconds_loop':wall,'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'dtype':'float32','batch_size':1,'seed':46,'gpu':torch.cuda.get_device_name(),'source_before':before,'source_after':source_check(),'input_protocol':'author seqwin-test 7 frames centered, clamp to selected clip boundaries, RGB/ImageNet normalized, full-frame resize1024square','main_protocol':'author infer bilinear1024 logits, sigmoid, area to original512x640','auxiliary_protocol':'project diagnostic adapter: author postprocess_masks of auxiliary256 logits, sigmoid, area to original512x640','annotation_use':'none in input or prompt; missing masks are zero placeholders ignored by model','diagnostic_threshold':'>=0.5','native_threshold':'>0.5','native_receipt':str(Path(args.native_receipt).resolve())})
    json.dump(stats,open(out/'COMPLETE.json','w'),indent=2)
    print(json.dumps({k:v for k,v in stats.items() if k!='forward_seconds'},indent=2))
if __name__=='__main__':main()
