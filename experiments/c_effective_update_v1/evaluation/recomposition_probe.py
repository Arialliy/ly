"""Zero-update diagnosis of the newly observed FP32 backward recomposition gap."""
import torch
import runtime as rt
from data import inputs,batch_data
import effect_common as c

rt.setup();state=inputs();model,opt,checkpoint,criterion,settings=c.initialize()
params=[p for p in model.parameters() if p.requires_grad]
initial=rt.tensor_hash(model.state_dict());adam=c.optimizer_hash(opt.state_dict());rows=[]
for pair_id,index in [(0,63),(7,32),(2,25)]:
    terms,*_=c.compute(model,batch_data(index,state),index,criterion,settings)
    groups=[c.vector(c.gradients(terms[k],params),params) for k in ('fit','weighted_keep','weighted_track')]
    full=c.vector(c.gradients(terms['full_F'],params),params)
    repeat=c.vector(c.gradients(terms['full_F'],params),params)
    summed=torch.stack(groups).sum(0);residual=full-summed
    rows.append(dict(pair_id=pair_id,index=index,full_l2=c.norm(full),group_l2=[c.norm(g) for g in groups],
        residual_l2=c.norm(residual),relative_to_full=c.norm(residual)/c.norm(full),
        relative_to_group_norm_sum=c.norm(residual)/sum(c.norm(g) for g in groups),
        max_absolute_error=float(residual.abs().max()),cosine=c.cosine(full,summed),
        repeated_full_backward_equal=torch.equal(full,repeat),
        original_scalar=float(terms['full_F'].detach()),group_scalar_sum=float(sum(terms[k] for k in ('fit','weighted_keep','weighted_track')).detach())))
    del terms,_,groups,full,repeat,summed,residual
assert rt.tensor_hash(model.state_dict())==initial and c.optimizer_hash(opt.state_dict())==adam
rt.save_json(rt.E/'evaluation/RECOMPOSITION_PROBE.json',dict(status='COMPLETE',parameter_updates=0,
    model_and_adam_unchanged=True,rows=rows,probe_sha256=rt.sha(__file__)))
print(rows,flush=True)
