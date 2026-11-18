"""Read-only fixed training-input diagnostics at the four saved checkpoints.

Uses the pre-existing eight calibration entries, fixed target/background cache,
and original task statistics. No backward, optimizer step, or sample selection.
These training summaries cannot establish development-set actual recovery.
"""
import argparse
import time
import torch
import runtime as rt


class UpdateMoments:
    def __init__(self):
        self.parts = []

    def hook(self, module, inputs, output):
        a, r, _ = inputs
        m, g = output
        # One tiny detached accumulator per update, transferred after the batch.
        g = g.double()
        self.parts.append(torch.stack((g.sum(), g.square().sum(),
            g.new_tensor(g.numel()), g.min(), g.max(),
            a.square().mean(), r.square().mean(), m.square().mean())).detach())

    def result(self):
        values = torch.stack(self.parts).double()
        mean = values[:, 0].sum()/values[:, 2].sum()
        variance = (values[:, 1].sum()/values[:, 2].sum()-mean.square()).clamp_min(0)
        stats = torch.stack((mean, variance.sqrt(), values[:, 3].min(), values[:, 4].max(),
                             *values[:, 5:].mean(0).sqrt())).cpu().tolist()
        return dict(zip(('retention_mean', 'retention_std', 'retention_min', 'retention_max',
                         'aligned_rms', 'candidate_rms', 'updated_state_rms'), stats),
                    calls=len(self.parts), meaning='all noninitial update pixels; not target-only')


@torch.no_grad()
def run(condition, device='cuda'):
    out = rt.E/'training_diagnostics'/condition/'FIXED_INPUTS.json'
    if out.exists():
        raise FileExistsError(out)
    model, _ = rt.load_initial(condition, device)
    run_path = rt.E/'training'/condition/'RUN.json'
    run_info = rt.read_json(run_path)
    assert run_info['status'] == 'COMPLETE'
    assert run_info['source_sha256'] == rt.source_hashes()
    prior = rt.old()
    frozen = prior.frozen_f()
    indices = list(frozen.common.CALIBRATION_INDICES)
    assert indices == [5, 6, 18, 25, 30, 32, 51, 7]
    state = frozen.data.inputs()
    criterion = prior.load_module('_m12_probe_loss', prior.MODEL_SOURCE).SoftLoUloss().to(device)
    rows, snapshots = [], {}
    began = time.perf_counter()
    for step in (2048, 2080, 2112, 2176):
        path = run_path.with_name(f'snapshot_{step}.pth')
        cp = torch.load(path, map_location='cpu', weights_only=True)
        assert cp['step'] == step and cp['condition'] == condition
        snapshots[str(step)] = rt.sha(path)
        model.load_state_dict(cp['model_state_dict'], strict=True)
        model.eval()
        identity = prior.tensor_hash(model.state_dict())
        modes = ['current_only'] if condition == 'V12Current' else ['history', 'current_only']
        for mode in modes:
            model.mode = mode
            for index in indices:
                batch = frozen.data.batch_data(index, state)
                moments = UpdateMoments()
                handle = model.adapter.recurrent_cell.state_update.register_forward_hook(moments.hook)
                try:
                    terms, stats, features, logits, risk, cache = frozen.common.compute(
                        model, batch, index, criterion, prior.SETTINGS)
                finally:
                    handle.remove()
                rows.append({'condition': condition, 'mode': mode, 'step': step,
                    'additional_step': step-2048, 'plan_index': index,
                    'loss_terms': {k: float(v) for k, v in terms.items()},
                    'readout': frozen.common.readout_metrics(risk, cache),
                    'state_update': moments.result(),
                    **{k: batch[k] for k in ('input_sha256', 'label_sha256', 'auxiliary_input_sha256')}})
                del terms, stats, features, logits, risk, cache, batch, moments
            print({'condition': condition, 'step': step, 'mode': mode,
                   'rows': len(rows), 'elapsed_seconds': time.perf_counter()-began}, flush=True)
        prior.assert_frozen(model, run_info['original_state_sha256'])
        assert prior.tensor_hash(model.state_dict()) == identity
        assert all(p.grad is None for p in model.parameters())
    rt.save(out, {'status': 'COMPLETE', 'condition': condition, 'rows': rows,
        'calibration_indices': indices, 'snapshots_sha256': snapshots,
        'training_run_sha256': rt.sha(run_path), 'source_sha256': rt.source_hashes(),
        'protocol_sha256': rt.sha(rt.E/'PROTOCOL_FREEZE.json'),
        'all_model_state_unchanged': True, 'backward_calls': 0, 'optimizer_steps': 0,
        'scope': 'Existing training calibration inputs and fixed cached regions; no development matching',
        'gate_statistics': 'All noninitial update pixels; target/background readout uses original fixed cache',
        'elapsed_seconds': time.perf_counter()-began})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True, choices=rt.CONDITIONS)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run(args.condition, args.device)
