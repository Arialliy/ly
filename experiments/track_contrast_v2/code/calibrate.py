"""Fixed training-only readout/pathway check and common additional-loss scale."""
import statistics
import numpy as np
import torch
from runtime import E,V1,INIT_CP,MODEL_SOURCE,setup,load_module,save_json,sha,tensor_hash
from common import initial_model,compute,readout_metrics,contrast_metrics,CALIBRATION_INDICES,old
from data import inputs,batch_data


def cosine(a,b):
    numerator=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    na=sum(float(x.detach().double().square().sum()) for x in a if x is not None)**.5
    nb=sum(float(x.detach().double().square().sum()) for x in b if x is not None)**.5
    return numerator/(na*nb) if na*nb else None


def main():
    if (E/'CALIBRATION.json').exists():raise FileExistsError('CALIBRATION.json')
    setup();state=inputs();model,optimizer,names,count,opt_hash=initial_model()
    initial=tensor_hash(model.state_dict())
    config=__import__('json').loads((V1/'CALIBRATION.json').read_text())['fixed_settings']
    criterion=load_module('v2_calibration_official_loss',MODEL_SOURCE).SoftLoUloss().cuda()
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    rows=[]
    for index in CALIBRATION_INDICES:
        batch=batch_data(index,state)
        terms,stats,f,z,r,c=compute(model,batch,index,criterion,config)
        row={'index':index,'losses':{k:float(v.detach()) for k,v in terms.items()},
            'contrast':contrast_metrics(stats),'readout':readout_metrics(r,c),'gradient_norms':{},'gradient_pathway':{}}
        gradients={}
        for key in ('base','det','keep','C','frame','gap'):
            norm,g=old.grad_norm(terms[key],params)
            assert np.isfinite(norm),(index,key,norm)
            gradients[key]=g;row['gradient_norms'][key]=norm
            row['gradient_pathway'][key]={prefix:sum(float(v.detach().double().square().sum())
                for (name,_),v in zip(named,g) if name.startswith(prefix) and v is not None)**.5
                for prefix in ('TPro.','conv_out1.','conv_out2.')}
        row['gradient_cosines']={f'{key}_vs_{other}':cosine(gradients[key],gradients[other])
            for key in ('frame','gap') for other in ('base','det','keep','C')}
        row['readout_gradients']={}
        for key in ('frame','gap'):
            a,b=torch.autograd.grad(terms[key],(r['q_positive'],r['q_negative']),retain_graph=True)
            assert bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            assert float(a.min())<0 and float(b.max())>0
            assert float(a.max())<=1e-7 and float(b.min())>=-1e-7
            row['readout_gradients'][key]={'target_l2':float(a.norm()),'background_l2':float(b.norm()),
                'target_min':float(a.min()),'target_max':float(a.max()),'background_min':float(b.min()),'background_max':float(b.max())}
            assert all(row['gradient_pathway'][key][prefix]>0 for prefix in ('TPro.','conv_out1.','conv_out2.'))
        rows.append(row);print(__import__('json').dumps(row),flush=True)
        del terms,stats,f,z,r,c,gradients,a,b,g
    med={k:statistics.median(r['gradient_norms'][k] for r in rows) for k in ('base','frame','gap')}
    coefficient=.25*med['base']/max(med['frame'],med['gap'])
    assert np.isfinite(coefficient) and 1e-6<coefficient<1e6
    assert tensor_hash(model.state_dict())==initial
    config=dict(config,lambda_readout=coefficient)
    save_json(E/'CALIBRATION.json',{'status':'complete','uses_development':False,'seed':46,
        'calibration_entry_indices':CALIBRATION_INDICES,'initial_checkpoint_sha256':sha(INIT_CP),
        'initial_optimizer_sha256':opt_hash,'no_parameter_updates':True,
        'method':'common lambda = 0.25 * median C-v1 total gradient norm / max(median frame norm, median gap norm); actual C2048 parameters, no warmup, no dev sweep',
        'gradient_norm_medians':med,'weighted_gradient_ratios':{k:coefficient*med[k]/med['base'] for k in ('frame','gap')},
        'fixed_settings':config,'measurements':rows})

if __name__=='__main__':main()
