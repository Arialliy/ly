"""Independent formula and computational-path checks for the fixed diagnostic."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from infer_misreference import misreference, infer_from_full_evidence, precomputed_constructor, runner, N


class ToyOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv3d(1, 8, (1, 3, 3), padding=(0, 1, 1))
        self.output = nn.Conv3d(8, 1, 1)
    def forward(self, x):
        h = self.features(x).relu()
        return h, self.output(h)[:, 0]


def main():
    runner.deterministic_setup()
    module = runner.load_module('misreference_cpu_frozen_module', N / 'code/module.py')
    evidence = module.LocalReferenceEvidence('relative')
    x = torch.randn(1, 1, 6, 23, 29)
    v = evidence.construct_evidence(x)
    original = v.clone()
    zero = misreference(v, evidence.s_min, (0, 0))
    assert torch.equal(v, zero)
    shifts = (11, 14)
    changed = misreference(v, evidence.s_min, shifts)
    expected = v.numpy().copy()
    for base in (1, 6):
        valid = expected[:, base+4] != 0
        mu = np.roll(original[:, base+1].numpy(), shift=shifts, axis=(-2,-1))
        s = np.roll(original[:, base+2].numpy(), shift=shifts, axis=(-2,-1))
        expected[:, base+1] = np.where(valid, mu, 0)
        expected[:, base+2] = np.where(valid, s, 0)
        expected[:, base+3] = np.where(valid, (expected[:,base]-mu)/np.maximum(s,evidence.s_min),0)
    np.testing.assert_array_equal(changed.numpy(), expected)
    assert torch.equal(v, original)
    assert torch.equal(changed[:, [0,1,5,6,10]], original[:, [0,1,5,6,10]])
    assert torch.count_nonzero(changed[:, 1:6, :1]) == 0
    assert torch.count_nonzero(changed[:, 6:11, :4]) == 0
    assert torch.isfinite(changed).all()
    # A crop from a full-image roll must differ from rolling that crop itself.
    cropped_global = changed[:, 2, :, 3:18, 4:23]
    wrong_local = torch.roll(original[:, 2, :, 3:18, 4:23], shifts=shifts, dims=(-2,-1))
    assert not torch.equal(cropped_global, wrong_local)
    model = runner.wrap_model(ToyOriginal(), 'N2').eval()
    with torch.no_grad():
        model.evidence.output.weight.normal_(0,.2)
        model.evidence.output.bias.fill_(.1)
        before = runner.tensor_hash(model.state_dict().items())
        direct = model(x)[1].sigmoid()[0].numpy()
        full = model.evidence.construct_evidence(x)
        tiled, _, calls = infer_from_full_evidence(model,x,full,torch.device('cpu'),core=12,halo=16)
        np.testing.assert_allclose(direct,tiled,atol=1e-6,rtol=1e-6)
        assert 'construct_evidence' not in model.evidence.__dict__
        try:
            with precomputed_constructor(model.evidence, full):
                raise RuntimeError('intentional cleanup test')
        except RuntimeError:
            pass
        assert 'construct_evidence' not in model.evidence.__dict__
        assert runner.tensor_hash(model.state_dict().items()) == before
    result=dict(status='PASS',checks={
        'zero_shift_exact':True,'numpy_global_shift_formula_exact':True,
        'input_not_mutated':True,'I_D_m_unchanged':True,'missing_reference_all_zero':True,
        'finite':True,'global_shift_not_tile_local':True,'constructor_restored_including_exception':True,
        'all_weights_unchanged':True,'precomputed_nonzero_readout_equivalence':True},
        max_probability_error=float(np.max(np.abs(direct-tiled))),tile_calls=calls,
        diagnostic_source_sha256=runner.sha(Path(__file__).with_name('infer_misreference.py')),
        test_source_sha256=runner.sha(__file__),frozen_module_sha256=runner.sha(N/'code/module.py'))
    destination=N/'checks/MISREFERENCE_CPU_TESTS.json'
    runner.save_json(destination,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
