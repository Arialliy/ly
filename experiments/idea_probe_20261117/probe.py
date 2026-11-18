import copy, csv, hashlib, io, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn

E=Path(__file__).resolve().parent
B=E.parents[1]
TRAIN=['DJI_0030_9','DJI_0043_2','DJI_0057_1','DJI_0113_3','DJI_0115_2','DJI_0217_2','DJI_0309_1','DJI_0323_2']
DEV=['DJI_0026_2','DJI_0067_2']
LAGS=[-4,-2,0,2,4]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj): (E/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def auc(y,s):
    order=np.argsort(s,kind='stable'); ranks=np.empty(len(s),float); k=0
    while k<len(s):
        j=k+1
        while j<len(s) and s[order[j]]==s[order[k]]:j+=1
        ranks[order[k:j]]=(k+1+j)/2; k=j
    p=y==1;n=int(p.sum());m=len(y)-n
    return float((ranks[p].sum()-n*(n+1)/2)/(n*m))
def feats(im,x,y):
    q=im[y-4:y+5,x-4:x+5].astype(np.float32)
    edge=np.ones((9,9),bool);edge[3:6,3:6]=False
    return (q.reshape(3,3,3,3).mean((1,3)).reshape(-1)-q[edge].mean())/255.0
def overlaps(x,y,bb,margin):
    a,b,w,h=bb
    return a-margin<=x<=a+w+margin and b-margin<=y<=b+h+margin
def main():
    start=time.time();torch.set_num_threads(2);torch.manual_seed(46);np.random.seed(46)
    assert not (E/'RESULTS.json').exists()
    frozen=json.loads((E/'FREEZE.json').read_text())
    for f,h in frozen['files'].items():assert sha(E/f)==h,f
    assert auc(np.array([0,1,0,1]),np.array([0.,1.,0.,1.]))==1
    assert auc(np.array([0,1]),np.array([1.,1.]))==.5
    input_hashes={};manifest=[];allx=[];ally=[];pairid=0;sampling=np.random.default_rng(46)
    for seq in TRAIN+DEV:
        root=B/'data/stage1'/seq;ap=root/'annotations.json';input_hashes[str(ap)]=sha(ap)
        a=json.loads(ap.read_text());ims=sorted(a['images'],key=lambda z:z['frame_id'])
        anns=defaultdict(list)
        for an in a['annotations']:anns[an['image_id']].append(an)
        images={};before=len(manifest)
        def get(t):
            if t not in images:
                p=root/'images'/Path(ims[t]['file_name']).name
                raw=p.read_bytes(); input_hashes[str(p)]=hashlib.sha256(raw).hexdigest()
                images[t]=np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
            return images[t]
        for t in range(4,149,8):
            slots=[t+d for d in LAGS]
            assert all(ims[z]['frame_id']-ims[t]['frame_id']==z-t for z in slots)
            frames=[get(z) for z in slots];H,W=frames[2].shape
            candidates=[]
            for an in sorted(anns[ims[t]['id']],key=lambda q:q['id']):
                x,y,w,h=an['bbox'];cx=int(round(x+w/2));cy=int(round(y+h/2))
                if an.get('ignore',0) or an.get('iscrowd',0) or not(0<w*h<=144 and max(w,h)<=16):continue
                if not(4<=cx<W-4 and 4<=cy<H-4):continue
                candidates.append((an,cx,cy))
            if len(candidates)>24:
                candidates=[candidates[i] for i in sorted(sampling.choice(len(candidates),24,replace=False))]
            boxes=[an['bbox'] for z in slots for an in anns[ims[z]['id']]]
            bg=[]
            for cy in range(8,H-8,8):
                for cx in range(8,W-8,8):
                    if not any(overlaps(cx,cy,q,6) for q in boxes):bg.append((cx,cy))
            if not bg:continue
            bgf=np.array([feats(frames[2],x,y)[4] for x,y in bg]);used=set()
            for an,cx,cy in candidates:
                pf=feats(frames[2],cx,cy)[4];dist=np.abs(bgf-pf)
                if used:dist[list(used)]=np.inf
                j=int(dist.argmin())
                if not np.isfinite(dist[j]):continue
                used.add(j);bx,by=bg[j]
                for label,x,y in [(1,cx,cy),(0,bx,by)]:
                    f=np.stack([feats(im,x,y) for im in frames]);allx.append(f);ally.append(label)
                    manifest.append({'sample_id':len(manifest),'pair_id':pairid,'split':'train' if seq in TRAIN else 'dev','sequence':seq,'source_prefix':'_'.join(seq.split('_')[:2]),'frame':t,'source_frame':ims[t].get('mot_frame_id',t),'annotation_id':an['id'],'track_id':an['track_id'],'label':label,'x':x,'y':y,'current_contrast':float(f[2,4]),'match_contrast_abs_difference':float(dist[j])})
                pairid+=1
        print(json.dumps({'sequence':seq,'samples':len(manifest)-before,'elapsed':time.time()-start}),flush=True)
    dump('INPUT_HASHES.json',input_hashes)
    with (E/'SAMPLE_MANIFEST.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    X=np.stack(allx).astype('float32');Y=np.array(ally,dtype='float32')
    train=np.array([r['split']=='train' for r in manifest]);seqs=np.array([r['sequence'] for r in manifest])
    mu=X[train].reshape(-1,9).mean(0);sd=X[train].reshape(-1,9).std(0).clip(.01)
    real=((X-mu)/sd).reshape(len(X),45)
    current=((np.repeat(X[:,2:3,:],5,axis=1)-mu)/sd).reshape(len(X),45)
    assert np.array_equal(real.reshape(-1,5,9)[:,2],current.reshape(-1,5,9)[:,2])
    assert np.all(current.reshape(-1,5,9)==current.reshape(-1,5,9)[:,2:3])
    np.savez_compressed(E/'FEATURES.npz',raw=X,labels=Y,train=train,mu=mu,sd=sd)
    rng=np.random.default_rng(46);pairs={s:np.flatnonzero((seqs==s)&(Y==1)) for s in TRAIN}
    plan=[]
    for step in range(100):
        idx=[]
        for s in rng.choice(TRAIN,128):
            i=int(rng.choice(pairs[s]));assert manifest[i+1]['pair_id']==manifest[i]['pair_id'];idx.extend([i,i+1])
        plan.append(idx)
    plan=np.array(plan);np.save(E/'EXPOSURE_PLAN.npy',plan)
    model=nn.Sequential(nn.Linear(45,32),nn.ReLU(),nn.Linear(32,1));initial=copy.deepcopy(model.state_dict())
    torch.save(initial,E/'INITIAL.pt');pred={};logs={};seconds={}
    for name,data in [('real',real),('current',current)]:
        model.load_state_dict(initial);opt=torch.optim.Adam(model.parameters(),lr=.003)
        xx=torch.from_numpy(data);yy=torch.from_numpy(Y);losses=[];s=time.time()
        for step,idx in enumerate(plan):
            z=model(xx[idx]).squeeze(1);loss=nn.functional.binary_cross_entropy_with_logits(z,yy[idx]);opt.zero_grad();loss.backward();opt.step()
            assert torch.isfinite(loss);losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            pred[name]=model(xx).squeeze(1).numpy()
            if name=='real':
                pred['real_weight_current']=model(torch.from_numpy(current)).squeeze(1).numpy()
                reverse=real.reshape(-1,5,9)[:,[4,3,2,1,0]].copy().reshape(-1,45)
                pred['real_weight_reverse']=model(torch.from_numpy(reverse)).squeeze(1).numpy()
        torch.save(model.state_dict(),E/(name+'_step100.pt'));seconds[name]=time.time()-s;logs[name]=losses
    dump('TRAINING_LOSSES.json',logs);np.savez_compressed(E/'PREDICTIONS.npz',**pred)
    summary={};counts={}
    for seq in TRAIN+DEV:
        idx=np.flatnonzero(seqs==seq);pi=idx[Y[idx]==1];tracks=sorted(set(manifest[i]['track_id'] for i in pi));counts[seq]={'samples':len(idx),'pairs':len(pi),'tracks':len(tracks),'track_ids':tracks}
        summary[seq]={}
        for name,scores in pred.items():
            pscore=(scores[pi]>scores[pi+1]).astype(float)+.5*(scores[pi]==scores[pi+1])
            trackscore=[float(pscore[[manifest[i]['track_id']==tr for i in pi]].mean()) for tr in tracks]
            summary[seq][name]={'auc':auc(Y[idx],scores[idx]),'paired_accuracy':float(pscore.mean()),'track_macro_paired_accuracy':float(np.mean(trackscore))}
    delta={s:{k:summary[s]['real'][k]-summary[s]['current'][k] for k in summary[s]['real']} for s in DEV}
    qualified=all(counts[s]['pairs']>=20 and counts[s]['tracks']>=5 for s in DEV)
    positive=qualified and all(delta[s][k]>=.02 for s in DEV for k in ['auc','track_macro_paired_accuracy'])
    negative=qualified and all(delta[s][k]<=0 for s in DEV for k in ['auc','track_macro_paired_accuracy'])
    verdict='POSITIVE_SIGNAL' if positive else 'NEGATIVE_WITHIN_PROBE' if negative else 'MIXED_OR_NULL'
    result={'status':'COMPLETE','verdict':verdict,'qualified':qualified,'summary':summary,'dev_real_minus_current':delta,'counts':counts,'train_seconds':seconds,'elapsed_seconds':time.time()-start,'device':'cpu','gpu_hours':0,'parameters':sum(p.numel() for p in model.parameters()),'seeds':[46],'steps':100,'input_files':len(input_hashes),'sources_train':TRAIN,'sources_dev':DEV,'confirmation_loaded':False,'limitations':['Oracle bbox-centered patch classification, not fullframe detection','Previously used development sources, not untouched test','One training seed and small representation; negatives do not prove absent information','Samples share tracks, frames, and backgrounds; independent top-level units are source clips','Background exclusion only covers existing labels','Bidirectional context includes future frames','Current control removes all temporal input for this probe; unrelated to existing DeepPro branch replacement']}
    dump('RESULTS.json',result)
    lines=['# 原始小块探针实测结果','','判定：'+verdict+'。仅为局部可利用信息探针，不是检测成绩。','', '| 开发来源 | 单帧复制AUC | 真实邻帧AUC | 差值 | 单帧轨迹等权配对正确率 | 真实邻帧轨迹等权配对正确率 |','|---|---:|---:|---:|---:|---:|']
    for s in DEV:
        c=summary[s]['current'];r=summary[s]['real'];lines.append(f"| {s} | {c['auc']:.6f} | {r['auc']:.6f} | {r['auc']-c['auc']:+.6f} | {c['track_macro_paired_accuracy']:.6f} | {r['track_macro_paired_accuracy']:.6f} |")
    lines+=['',f'CPU耗时{result["elapsed_seconds"]:.2f}秒；GPU小时0；参数{result["parameters"]}；seed46、100步。','', '两个开发来源分别评价；样本与轨迹计数见RESULTS.json，全部采样见SAMPLE_MANIFEST.csv。正负配对使用当前帧原始对比，不依赖模型响应。训练/开发按来源隔离，但样本内部并非独立。','', '局限：GT框中心定位、手工局部特征、单随机种子、两个已反复使用的开发片段、未来帧上下文；不证明弱目标恢复、全图误报改善、时序机制成立或论文创新。']
    (E/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump('ARTIFACT_HASHES.json',{p.name:sha(p) for p in E.iterdir() if p.is_file() and p.name not in ['ARTIFACT_HASHES.json','LAUNCH.json','run.log']})
    print(json.dumps({'verdict':verdict,'delta':delta,'elapsed':time.time()-start}),flush=True)
if __name__=='__main__':main()
