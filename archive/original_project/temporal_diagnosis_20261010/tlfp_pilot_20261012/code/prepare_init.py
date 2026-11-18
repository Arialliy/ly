"""Save one untrained spatial initialization and its exact S/T branch mapping."""
import json
from pathlib import Path
import torch
import train_tlfp as r
from temporal_lfp import FramewiseFeatureAdapter, SpatialLFPReference, TemporalLFPAdapter

def main():
    r.deterministic_setup()
    folder=r.N/'init'
    if (folder/'INITIALIZATION.json').exists(): raise FileExistsError('Initialization already frozen')
    spatial=FramewiseFeatureAdapter(SpatialLFPReference(32),32,8)
    torch.save({'adapter_state_dict':spatial.state_dict(),'seed':46},folder/'COMMON_SPATIAL.pth')
    states={}
    for label,mode in r.VARIANTS.items():
        adapter=TemporalLFPAdapter.from_spatial(spatial,mode=mode,frame_chunk=8)
        states[label]=adapter.state_dict()
        torch.save({'adapter_state_dict':states[label],'seed':46,'condition':label},folder/f'{label}.pth')
    assert torch.equal(states['S']['operator.attention.weight'],states['T']['operator.attention.weight'][:,2:4])
    assert torch.count_nonzero(states['T']['operator.attention.weight'][:,:2])==0
    assert torch.count_nonzero(states['T']['operator.attention.weight'][:,4:])==0
    for key in ('scale','operator.sigma'): assert torch.equal(states['S'][key],states['T'][key])
    r.save_json(folder/'INITIALIZATION.json',{'seed':46,'files':{p.name:r.sha(p) for p in folder.glob('*.pth')},'S_parameters':100,'T_parameters':296,'common_center_sigma_scale_equal':True,'neighbor_kernels_zero':True,'trained_weights_used':False})
    print('Common initialization saved; exact center/sigma/gate mapping PASS')
if __name__=='__main__': main()
