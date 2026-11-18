"""Matched B/C correction heads on frozen, pretrained five-frame RGB DUF x2."""
import torch
from torch import nn
import torch.nn.functional as F

if __package__:
    from .geometry import phase_rows, mix_polyphase
    from .transport import transport_evidence
else:
    from geometry import phase_rows, mix_polyphase
    from transport import transport_evidence


def build_evidence(lr, base, forward, backward):
    """Precompute fixed input features. No HR training or evaluation target is used."""
    if lr.ndim != 5 or lr.shape[1:3] != (5,3):
        raise ValueError("Expected B,5,3,H,W RGB observations")
    transported = transport_evidence(lr,base,forward,backward,blur_sigma=.8)
    q = transported['confidence']
    # Forward maps reference pixel p to p+f in the neighbor. Local rigid motion
    # maps a neighboring lattice point to reference phase -f (integer origin ignored).
    offsets = -forward
    rows = phase_rows(offsets)
    weighted_residual = transported['aligned_residual'] * q
    shared = torch.cat([weighted_residual, q, offsets/8, rows],dim=2).flatten(1,2)
    obs = torch.cat([shared,(transported['aligned_lr']*q).flatten(1,2)],dim=1)
    prior = torch.cat([F.pixel_unshuffle(base,2),lr[:,2]],dim=1)
    # Apply the same geometry routine to the four coordinate vectors once.
    # P[out,in] is shared across RGB and independent of trainable parameters.
    b,_,_,h,w = lr.shape
    basis = torch.eye(4,dtype=lr.dtype,device=lr.device).reshape(1,16,1,1).expand(b,-1,h,w)
    packed_gate = mix_polyphase(basis,torch.zeros_like(basis),offsets,q,tau=.1)
    gate = packed_gate.reshape(b,4,4,h,w).transpose(1,2).flatten(1,2).contiguous()
    return {'prior_input':prior,'obs_input':obs,'gate_input':shared,'geometry_gate':gate}


def correction_block(channels, outputs):
    network = nn.Sequential(nn.Conv2d(channels,24,3,padding=1),nn.LeakyReLU(.1),
                            nn.Conv2d(24,24,3,padding=1),nn.LeakyReLU(.1),
                            nn.Conv2d(24,outputs,3,padding=1))
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    return network


class PairedHead(nn.Module):
    """Same three trainable functions and training budget; only the fixed gate differs."""
    def __init__(self,mode='B'):
        super().__init__()
        if mode not in ('B','C'):
            raise ValueError("mode must be B or C")
        self.mode=mode
        self.prior=correction_block(15,12)
        self.obs=correction_block(65,12)
        self.gate=correction_block(50,16)

    def forward(self,evidence,base):
        b,_,hh,ww=base.shape
        h,w=hh//2,ww//2
        prior=self.prior(evidence['prior_input']).reshape(b,3,4,h,w)
        obs=self.obs(evidence['obs_input']).reshape(b,3,4,h,w)
        matrix=self.gate(evidence['gate_input']).reshape(b,4,4,h,w)
        if self.mode=='C':
            matrix=matrix+evidence['geometry_gate'].reshape(b,4,4,h,w)
        else:
            identity=torch.eye(4,device=matrix.device,dtype=matrix.dtype).reshape(1,4,4,1,1)
            matrix=matrix+.5*identity
        detail=prior+torch.einsum('bijhw,bcjhw->bcihw',matrix,obs-prior)
        # Exact constraint on each RGB 2x2 cell's average relative to the frozen
        # reconstruction (individual HR samples remain free to change).
        detail=detail-detail.mean(dim=2,keepdim=True)
        return base+F.pixel_shuffle(detail.reshape(b,12,h,w),2)
