"""State/control contracts for the new M1; test aligner is not native PDA."""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.m1_source_relations_v1.code.adapter import (
    SourceRelationsAdapter, BankState, Entry, start_window,
)


class TestOnlyPDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.6))
        self.calls = []

    def forward(self, source, current):
        self.calls.append((source.detach().clone(), current.detach().clone()))
        return source * self.scale + current * 0.2


def make_adapter(readout="simple", reduction="relations", active=True):
    torch.manual_seed(46)
    model = SourceRelationsAdapter(pda=TestOnlyPDA(), native_pda=False,
        readout=readout, reduction_mode=reduction, checkpoint_recurrence=False)
    if active:
        with torch.no_grad():
            layer = model.out_projection if readout == "simple" else model.reader.delta
            layer.weight.normal_(0, 0.1)
            layer.bias.zero_()
    return model


def inputs(batch=2, time=8):
    g = torch.Generator().manual_seed(1024)
    shape = (batch, 32, time, 4, 4)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


class TestAdapter(unittest.TestCase):
    def test_zero_initialized_readers_preserve_h0(self):
        f, h = inputs()
        for reader in ("simple", "selective"):
            for reduction in ("relations", "early_merge"):
                model = make_adapter(reader, reduction, active=False).eval()
                for mode in model.MODES:
                    self.assertTrue(torch.equal(model(f, h, mode), h))
        # Actual alignment still executes behind zero-initialized residuals.
        self.assertGreater(len(model.recurrent_cell.pda.calls), 0)

    def test_bypass_skips_m1_and_m2(self):
        f, h = inputs()
        for reader in ("simple", "selective"):
            model = make_adapter(reader)
            model.enabled = False
            self.assertIs(model(f, h), h)
            self.assertEqual(len(model.recurrent_cell.pda.calls), 0)

    def test_fifo_source_identity_and_current_replacement_before_pda(self):
        f, h = inputs(batch=1, time=7)
        model = make_adapter().eval()
        with torch.no_grad():
            x = [model.in_projection(f[:, :, t]) for t in range(7)]
            for mode in model.MODES:
                model.recurrent_cell.pda.calls.clear()
                model(f, h, mode)
                expected = [(j, t) for t in range(7) for j in range(max(0, t - 4), t)]
                self.assertEqual(len(model.recurrent_cell.pda.calls), len(expected))
                for (source, current), (j, t) in zip(model.recurrent_cell.pda.calls, expected):
                    torch.testing.assert_close(source, x[j if mode == "history" else t])
                    torch.testing.assert_close(current, x[t])

    def test_40_frame_operator_counts(self):
        model = make_adapter().eval()
        f, h = inputs(batch=1, time=40)
        calls = []
        handle = model.reducer.relation.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            model(f, h)
        handle.remove()
        self.assertEqual(len(model.recurrent_cell.pda.calls), 150)
        self.assertEqual(len(calls), 220)

    def test_current_does_not_depend_on_other_projected_frames(self):
        f, h = inputs(batch=1)
        model = make_adapter().eval()
        changed = f.clone()
        changed[:, :, 4] += 10
        with torch.no_grad():
            current = model(f, h, "current_only")
            changed_current = model(changed, h, "current_only")
            history = model(f, h, "history")
            changed_history = model(changed, h, "history")
        self.assertTrue(torch.equal(current[:, :, 5:], changed_current[:, :, 5:]))
        self.assertFalse(torch.equal(history[:, :, 5:], changed_history[:, :, 5:]))

    def test_batch_and_window_isolation(self):
        f, h = inputs()
        model = make_adapter().eval()
        with torch.no_grad():
            expected = model(f, h)
            separate = torch.cat([model(f[i:i+1], h[i:i+1]) for i in range(2)])
            model(f + 100, h)
            repeated = model(f, h)
        torch.testing.assert_close(expected, separate, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(expected, repeated))

    def test_temporal_chunks_preserve_output_and_gradients(self):
        for reader in ("simple", "selective"):
            for mode in ("history", "current_only"):
                full = make_adapter(reader)
                chunked = copy.deepcopy(full)
                f, h = inputs(batch=1)
                f1, f2 = f.clone().requires_grad_(), f.clone().requires_grad_()
                a = full(f1, h, mode)
                b = chunked(f2, h, mode, temporal_chunk_size=3)
                torch.testing.assert_close(a, b, atol=0, rtol=0)
                a[:, :, -1].square().mean().backward()
                b[:, :, -1].square().mean().backward()
                torch.testing.assert_close(f1.grad, f2.grad, atol=0, rtol=0)
                for p, q in zip(full.parameters(), chunked.parameters()):
                    torch.testing.assert_close(p.grad, q.grad)
                if mode == "history":
                    self.assertGreater(f2.grad[:, :, 3:7].abs().sum().item(), 0)
                    self.assertEqual(f2.grad[:, :, :3].count_nonzero().item(), 0)
                else:
                    self.assertEqual(f2.grad[:, :, :-1].count_nonzero().item(), 0)

    def test_checkpointing_preserves_gradients(self):
        for reduction in ("relations", "early_merge"):
            plain = make_adapter(reduction=reduction).train()
            saved = copy.deepcopy(plain)
            saved.checkpoint_recurrence = True
            f, h = inputs(batch=1, time=5)
            f1, f2 = f.clone().requires_grad_(), f.clone().requires_grad_()
            a, b = plain(f1, h), saved(f2, h)
            torch.testing.assert_close(a, b)
            a.square().mean().backward()
            b.square().mean().backward()
            torch.testing.assert_close(f1.grad, f2.grad)
            for p, q in zip(plain.parameters(), saved.parameters()):
                self.assertIsNotNone(p.grad)
                self.assertIsNotNone(q.grad)
                torch.testing.assert_close(p.grad, q.grad)

    def test_only_original_sources_are_written(self):
        model = make_adapter()
        state = start_window("one")
        x = torch.randn(1, 16, 4, 4, requires_grad=True)
        for t in range(6):
            current = x + t
            evidence, state = model.read_then_write(current, state, window_id="one", step_index=t, mode="history")
            self.assertTrue(torch.equal(state.entries[-1].feature, current))
            self.assertIsNot(state.entries[-1].feature, current)
            self.assertIsNot(state.entries[-1].feature, evidence)
        self.assertEqual([entry.step_index for entry in state.entries], [2, 3, 4, 5])
        state.entries[0].feature.sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_wrong_window_duplicate_and_out_of_order_sources_rejected(self):
        model = make_adapter()
        x = torch.randn(1, 16, 4, 4)
        cases = [
            (start_window("other"), 0),
            (start_window("one"), 1),
            (BankState("one", 2, 4, (Entry(0, x), Entry(0, x))), 2),
        ]
        for state, t in cases:
            with self.assertRaises(ValueError):
                model.read_then_write(x, state, window_id="one", step_index=t, mode="history")

    def test_no_history_and_single_history_have_zero_relation(self):
        model = make_adapter()
        x = torch.randn(1, 16, 4, 4)
        state = start_window("one")
        for t in (0, 1):
            evidence, state = model.read_then_write(x, state, window_id="one", step_index=t, mode="history")
            self.assertEqual(evidence[:, 8:].count_nonzero().item(), 0)

    def test_zero_readout_learns_then_evidence_receives_signal(self):
        for reader in ("simple", "selective"):
            model = make_adapter(reader, active=False)
            f, h = inputs(batch=1, time=4)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
            model(f, h).square().mean().backward()
            self.assertEqual(model.reducer.value[0].weight.grad.count_nonzero().item(), 0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model(f, h).square().mean().backward()
            self.assertGreater(model.reducer.value[0].weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.recurrent_cell.pda.scale.grad.abs().item(), 0)

    def test_native_pda_never_silently_uses_cpu_test_backend(self):
        model = make_adapter()
        model.native_pda = True
        x = torch.randn(1, 16, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model.align(x, x)

    def test_invalid_shapes_and_chunk_sizes_rejected(self):
        model = make_adapter()
        f, h = inputs()
        for chunk in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                model(f, h, temporal_chunk_size=chunk)
        with self.assertRaises(ValueError):
            model(f, h[:, :, :-1])
        with self.assertRaises(ValueError):
            model(f, h, mode="cross_video")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
