"""Single implementation smoke: real History-2048, native PDA, unchanged Adam map."""
from pathlib import Path

import numpy as np
import torch

import runtime as rt
from revision import AlignedResidualTSFM, TSFM


def main():
    output = rt.E / 'checks/REVISION_SMOKE.json'
    if output.exists():
        raise FileExistsError(output)
    prior = rt.old()
    original, parent = rt.load_initial('Continuation')
    revised, revised_parent = rt.load_initial('Revision')
    assert parent == revised_parent
    initial_hash = prior.tensor_hash(original.state_dict())
    assert initial_hash == prior.tensor_hash(revised.state_dict())
    original_parameters = {k: p for k, p in original.named_parameters() if p.requires_grad}
    revised_parameters = {k: p for k, p in revised.named_parameters() if p.requires_grad}
    assert list(original_parameters) == list(revised_parameters) == parent['trainable_parameter_names']
    assert len(original_parameters) == len(revised_parameters) == 53
    checkpoint = torch.load(rt.CHECKPOINT, map_location='cpu', weights_only=True)
    optimizers = [torch.optim.Adam(list(params.values()), lr=1e-5, weight_decay=1e-4)
                  for params in (original_parameters, revised_parameters)]
    for optimizer in optimizers:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        assert all(float(value['step']) == 2048 for value in optimizer.state.values())
    left, right = [optimizer.state_dict() for optimizer in optimizers]
    assert left['param_groups'] == right['param_groups']
    assert list(left['state']) == list(right['state'])
    for key, values in left['state'].items():
        for name, value in values.items():
            other = right['state'][key][name]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    raw = np.load(rt.DATA / rt.SEQS[0] / 'raw_frames.npy', mmap_mode='r')
    frames = np.array((raw[:40, :64, :64].astype(np.float32)-85.0799)/47.4845, copy=True)
    x = torch.from_numpy(frames)[None, None].cuda()
    original.eval()
    revised.eval()
    with torch.no_grad():
        original_logits = original(x)[1]
        f, h = revised.frozen_frontend_tiled(x)
        current = revised.adapter.in_projection(f[:, :, 1])
        previous = revised.adapter.in_projection(f[:, :, 0])
        cell = revised.adapter.recurrent_cell
        assert type(cell.tsfm) is AlignedResidualTSFM
        aligned = cell.pda(previous, current)
        actual_state = cell.tsfm(aligned, current)
        expected_state = aligned + TSFM.forward(cell.tsfm, aligned, current)
        assert torch.equal(actual_state, expected_state)
        initial_state_expected = h[:, :, 0] + revised.adapter.out_projection(
            torch.cat((previous, previous), dim=1))
        first_adapted = revised.adapter(f[:, :, :1], h[:, :, :1], mode='history')[:, :, 0]
        assert torch.equal(first_adapted, initial_state_expected)
    del f, h, current, previous, aligned, actual_state, expected_state
    revised.train()
    rt.old_train().restore_rng(checkpoint['rng_state'])
    features, logits = revised(x)
    assert torch.equal(original_logits[:, 0], logits[:, 0])
    assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(logits).all())
    loss = logits.square().mean() + features.square().mean()
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in revised_parameters.values())
    assert all(p.grad is None for p in revised.base.parameters())
    prior.assert_frozen(revised, parent['original_state_sha256'])
    assert initial_hash == prior.tensor_hash(revised.state_dict())
    paths = [Path(__file__), Path(__file__).with_name('revision.py')]
    record = {
        'status': 'PASS',
        'scope': 'One real-checkpoint 40x64x64 implementation smoke; no optimizer step or research result',
        'checkpoint_sha256': rt.sha(rt.CHECKPOINT),
        'source_sha256': {str(p): rt.sha(p) for p in paths},
        'initial_model_state_sha256': initial_hash,
        'both_conditions_identical_initial_tensors': True,
        'all_53_trainable_parameter_names_and_order_identical': True,
        'restored_Adam_53_states_identical_at_step_2048': True,
        'exact_aligned_plus_official_TSFM_identity': True,
        'first_frame_state_rule_and_logits_exact': True,
        'all_revised_outputs_and_adapter_gradients_finite': True,
        'frozen_base_has_no_gradients_or_state_changes': True,
        'all_model_parameters_and_buffers_unchanged': True,
        'implementation_loss': float(loss.detach()),
        'step_zero_logit_change_absmax': float((logits.detach()-original_logits).abs().max()),
        'gradient_l2': float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in revised_parameters.values()]))),
        'trainable_parameters': sum(p.numel() for p in revised_parameters.values()),
    }
    rt.save(output, record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
