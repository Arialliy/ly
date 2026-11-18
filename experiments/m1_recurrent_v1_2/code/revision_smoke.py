"""Bounded CUDA integration smoke; uses supplied core, real PDA and History-2048.

No research training or persisted parameter update. The active-gate fixture and
one gate-only SGD step exist only on a disposable in-memory model copy.
"""
from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import torch

import runtime as rt
from trace import trace_window


def norm_of_grad(parameters):
    values = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def verify_optimizer_pair(models, parent, checkpoint):
    optimizers, receipts = [], []
    for model in models:
        optimizer, receipt = rt.make_optimizer(model, parent, checkpoint)
        parameters, old_names, gate_names = rt.parameter_groups(model, parent)
        assert len(old_names) == 53 and len(gate_names) == 6
        assert sum(parameters[name].numel() for name in gate_names) == 2017
        group = checkpoint['optimizer_state_dict']['param_groups'][0]
        for name, identifier in zip(old_names, group['params']):
            expected = checkpoint['optimizer_state_dict']['state'][identifier]
            actual = optimizer.state[parameters[name]]
            for key, value in expected.items():
                if torch.is_tensor(value):
                    assert torch.equal(value.cpu(), actual[key].cpu()), (name, key)
                else:
                    assert value == actual[key], (name, key)
        assert all(parameters[name] not in optimizer.state for name in gate_names)
        optimizers.append(optimizer)
        receipts.append(receipt)
    assert receipts[0] == receipts[1]
    left, right = [optimizer.state_dict() for optimizer in optimizers]
    assert left['param_groups'] == right['param_groups']
    assert list(left['state']) == list(right['state'])
    for key, state in left['state'].items():
        assert all(torch.equal(value.cpu(), right['state'][key][name].cpu())
                   if torch.is_tensor(value) else value == right['state'][key][name]
                   for name, value in state.items())
    return receipts[0]


@torch.no_grad()
def gate_tile_check(update, aligned, candidate, current):
    """Only the local supplied gate is tiled, never PDA, TSFM or full M1."""
    expected_state, expected_gate = update(aligned, candidate, current, scope='pixel')
    assert float(expected_gate.std()) > 0
    actual_state, actual_gate = torch.empty_like(expected_state), torch.empty_like(expected_gate)
    height, width = aligned.shape[-2:]
    core, halo = 32, 2
    for y in range(0, height, core):
        y1 = min(height, y+core)
        ya, yb = max(0, y-halo), min(height, y1+halo)
        for x in range(0, width, core):
            x1 = min(width, x+core)
            xa, xb = max(0, x-halo), min(width, x1+halo)
            inputs = [v[..., ya:yb, xa:xb].contiguous() for v in (aligned, candidate, current)]
            state, gate = update(*inputs, scope='pixel')
            actual_state[..., y:y1, x:x1] = state[..., y-ya:y1-ya, x-xa:x1-xa]
            actual_gate[..., y:y1, x:x1] = gate[..., y-ya:y1-ya, x-xa:x1-xa]
    torch.testing.assert_close(actual_gate, expected_gate, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-6, rtol=1e-5)
    return {'scope': 'Gate-only, active nonconstant gate; PDA/TSFM not spatially tiled',
            'core': core, 'halo': halo, 'gate_std': float(expected_gate.std()),
            'gate_absmax_error': float((actual_gate-expected_gate).abs().max()),
            'state_absmax_error': float((actual_state-expected_state).abs().max()),
            'absolute_tolerance': 2e-6, 'relative_tolerance': 1e-5}


def main():
    output = rt.E / 'checks/GPU_INTEGRATION.json'
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('This integration smoke requires the real native CUDA PDA')
    assert rt.sha(rt.CHECKPOINT) == rt.CHECKPOINT_SHA
    current_model, parent = rt.load_initial('V12Current')
    history_model, history_parent = rt.load_initial('V12History')
    assert parent == history_parent
    assert current_model.mode == 'current_only' and history_model.mode == 'history'
    prior = rt.old()
    initial_hash = prior.tensor_hash(history_model.state_dict())
    assert prior.tensor_hash(current_model.state_dict()) == initial_hash
    checkpoint = torch.load(rt.CHECKPOINT, map_location='cpu', weights_only=True)
    new_state = history_model.state_dict()
    assert all(torch.equal(value.cpu(), new_state[name].cpu())
               for name, value in checkpoint['model_state_dict'].items())
    gate_keys = [name for name in new_state if '.state_update.' in name]
    assert len(gate_keys) == 6 and sum(new_state[name].numel() for name in gate_keys) == 2017
    assert all(torch.equal(new_state[name], current_model.state_dict()[name]) for name in gate_keys)
    assert Path(inspect.getfile(type(history_model.adapter.recurrent_cell.state_update))).resolve() == rt.REFERENCE.resolve()
    assert bool(torch.count_nonzero(history_model.adapter.out_projection.weight))
    migration = verify_optimizer_pair((current_model, history_model), parent, checkpoint)

    raw = np.load(rt.DATA / rt.SEQS[0] / 'raw_frames.npy', mmap_mode='r', allow_pickle=False)
    x = torch.from_numpy(np.array((raw[:40, :128, :128].astype(np.float32)-85.0799)/47.4845,
                                 copy=True))[None, None].cuda()
    with torch.no_grad():
        f, h = history_model.frozen_frontend_tiled(x)
        first = history_model.adapter.in_projection(f[:, :, 0])
        next_current = history_model.adapter.in_projection(f[:, :, 1])
        cell = history_model.adapter.recurrent_cell
        aligned = cell.pda(first.contiguous(), next_current.contiguous())
        candidate = cell.tsfm(aligned, next_current)
        state, retention = cell.state_update(aligned, candidate, next_current, scope='pixel')
        assert torch.equal(retention, torch.full_like(retention, .5))
        assert torch.equal(state, .5*candidate+.5*aligned)
        assert torch.equal(cell(first, next_current), state)
        first_expected = h[:, :, 0] + history_model.adapter.out_projection(torch.cat((first, first), 1))
        first_actual = history_model.adapter(f[:, :, :1], h[:, :, :1], mode='history')[:, :, 0]
        assert torch.equal(first_actual, first_expected)
        del aligned, candidate, state, retention, first, next_current

    # Use real frozen features on a smaller spatial crop to exercise recurrence
    # semantics. This is an implementation intervention, not video evidence.
    sf, sh = f[..., :64, :64].contiguous(), h[..., :64, :64].contiguous()
    changed_f = sf.clone()
    changed_f[:, :, 0] += .25
    with torch.no_grad():
        history = history_model.adapter(sf, sh, mode='history')
        current = current_model.adapter(sf, sh, mode='current_only')
        changed_history = history_model.adapter(changed_f, sh, mode='history')
        changed_current = current_model.adapter(changed_f, sh, mode='current_only')
        assert torch.equal(current[:, :, 1:], changed_current[:, :, 1:])
        history_dependency = float((history[:, :, 1:4]-changed_history[:, :, 1:4]).abs().max())
        assert history_dependency > 0
        assert torch.equal(history[:, :, 0], current[:, :, 0])
        chunked = history_model.adapter(sf, sh, mode='history', temporal_chunk_size=7)
        assert torch.equal(history, chunked)
        # A different intervening window cannot contaminate the next A call.
        history_model.adapter(changed_f, sh, mode='history')
        after_other_window = history_model.adapter(sf, sh, mode='history')
        assert torch.equal(history, after_other_window)
    del history, current, changed_history, changed_current, chunked, after_other_window, changed_f

    scratch = deepcopy(history_model)
    scratch.train()
    prior.assert_frozen(scratch, parent['original_state_sha256'])
    gate = scratch.adapter.recurrent_cell.state_update
    parameters = dict(scratch.adapter.named_parameters())

    def backward_probe():
        scratch.zero_grad(set_to_none=True)
        leaf = sf[:, :, :4].detach().clone().requires_grad_(True)
        probe_h = sh[:, :, :4].detach()
        scratch.adapter.recurrent_cell.begin_gate_statistics()
        adapted = scratch.adapter(leaf, probe_h, mode='history')
        statistics = scratch.adapter.recurrent_cell.end_gate_statistics()
        assert statistics['calls'] == 3
        features = scratch.base.conv_out1(adapted)
        logits = scratch.base.conv_out2(features)
        loss = features[:, :, -1].square().mean() + logits[:, :, -1].square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert scratch.adapter.recurrent_cell._gate_statistics is None
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters.values())
        assert leaf.grad is not None and bool(torch.isfinite(leaf.grad).all())
        return {'loss': float(loss.detach()), 'early_feature_gradient_l2': float(leaf.grad[:, :, 0].double().norm()),
                'gate_first_layers_gradient_l2': norm_of_grad(list(gate.gate_net[0].parameters())+list(gate.gate_net[2].parameters())),
                'gate_output_gradient_l2': norm_of_grad(gate.gate_net[-1].parameters()),
                'PDA_gradient_l2': norm_of_grad(scratch.adapter.recurrent_cell.pda.parameters()),
                'TSFM_gradient_l2': norm_of_grad(scratch.adapter.recurrent_cell.tsfm.parameters()),
                'forward_gate_statistics': statistics}

    zero_gate_probe = backward_probe()
    assert zero_gate_probe['gate_first_layers_gradient_l2'] == 0
    assert zero_gate_probe['gate_output_gradient_l2'] > 0
    # Explicit disposable fixture: activate the output layer, without changing
    # the persisted initialization or either official training branch.
    with torch.no_grad():
        last = gate.gate_net[-1]
        last.weight.copy_(torch.linspace(-.02, .02, last.weight.numel(), device=last.weight.device).reshape_as(last.weight))
        last.bias.fill_(.1)
    active_gate_probe = backward_probe()
    assert all(active_gate_probe[key] > 0 for key in ('gate_first_layers_gradient_l2',
               'gate_output_gradient_l2', 'early_feature_gradient_l2', 'PDA_gradient_l2', 'TSFM_gradient_l2'))
    before_gate = {name: p.detach().clone() for name, p in gate.named_parameters()}
    # One gate-only SGD update on the disposable test copy verifies that the
    # observed early-layer gradient can change parameters; not a model run.
    torch.optim.SGD(gate.parameters(), lr=.1).step()
    gate_changes = {name: float((p.detach()-before_gate[name]).abs().max()) for name, p in gate.named_parameters()}
    assert all(value > 0 for value in gate_changes.values())
    assert all(p.grad is None for p in scratch.base.parameters())
    prior.assert_frozen(scratch, parent['original_state_sha256'])
    scratch.zero_grad(set_to_none=True)
    scratch.eval()
    with torch.no_grad():
        first = scratch.adapter.in_projection(f[:, :, 0])
        next_current = scratch.adapter.in_projection(f[:, :, 1])
        aligned = scratch.adapter.recurrent_cell.pda(first.contiguous(), next_current.contiguous())
        candidate = scratch.adapter.recurrent_cell.tsfm(aligned, next_current)
        tile_result = gate_tile_check(gate, aligned, candidate, next_current)
    del sf, sh, first, next_current, aligned, candidate, f, h

    # Check the new hand-written trace against the actual complete model after
    # the gate is nonconstant. The ordinary 40-frame crop forward is sufficient;
    # the old full-spatial frontend equivalence audit is deliberately reused.
    trace = trace_window(scratch, x, [[] for _ in range(40)], sequence='implementation_crop', start=0)
    with torch.no_grad():
        original_mode = scratch.mode
        try:
            scratch.mode = 'history'
            normal_history = scratch(x)[1].cpu()
            scratch.mode = 'current_only'
            normal_current = scratch(x)[1].cpu()
        finally:
            scratch.mode = original_mode
        normal_off = scratch.base(x)[1].cpu()
    expected = {'History': normal_history, 'HistoryCurrent': normal_current, 'HistoryOff': normal_off}
    parity = {key: float((trace['logits'][key]-value).abs().max()) for key, value in expected.items()}
    assert all(torch.equal(trace['logits'][key], value) for key, value in expected.items()), parity
    assert trace['metadata']['state_update_calls_per_mode'] == {'History': 39, 'HistoryCurrent': 39, 'HistoryOff': 0}
    assert trace['frame_rows'][0]['retention_History_mean'] is None
    assert any(row['retention_History_std'] > 0 for row in trace['frame_rows'][1:])
    assert prior.tensor_hash(history_model.state_dict()) == initial_hash
    assert prior.tensor_hash(current_model.state_dict()) == initial_hash
    prior.assert_frozen(history_model, parent['original_state_sha256'])
    prior.assert_frozen(current_model, parent['original_state_sha256'])
    record = {'status': 'PASS',
        'scope': 'Native CUDA integration on real-checkpoint crops; no research training; no persisted parameter changes',
        'checkpoint_sha256': rt.sha(rt.CHECKPOINT), 'reference_sha256': rt.sha(rt.REFERENCE),
        'source_sha256': rt.source_hashes(), 'shared_initial_model_state_sha256': initial_hash,
        'both_conditions_initial_tensors_exact': True, 'all_old_tensors_exact_History2048': True,
        'old_out_projection_preserved_nonzero': True, 'gate_added_parameters': 2017,
        'gate_initial_retention_exact_half': True, 'actual_cell_matches_supplied_core_exact': True,
        'first_frame_rule_and_paired_output_exact': True, 'optimizer_migration': migration,
        'history_response_to_earlier_feature_change_absmax': history_dependency,
        'current_unaffected_by_earlier_feature_change_after_t0': True,
        'temporal_chunk_7_exact': True, 'window_state_reset_exact': True,
        'zero_output_layer_gradient_probe': zero_gate_probe, 'active_output_layer_gradient_probe': active_gate_probe,
        'disposable_gate_only_SGD_parameter_absmax_changes': gate_changes,
        'gate_only_spatial_tile_check': tile_result,
        'active_gate_trace_vs_standard_model_logit_absmax_errors': parity,
        'active_gate_trace_all_three_modes_exact': True, 'trace_metadata': trace['metadata'],
        'original_parameters_BN_and_initial_models_unchanged': True,
        'full_spatial_policy': 'Reuse validated frontend core128 halo16; M1/PDA/TSFM global, never spatially tiled'}
    rt.save(output, record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
