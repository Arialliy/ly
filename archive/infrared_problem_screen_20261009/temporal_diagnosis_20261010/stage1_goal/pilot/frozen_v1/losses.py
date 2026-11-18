"""Weak-box MIL losses for the frozen stage-1 comparison; no mask reconstruction."""
from __future__ import annotations

import torch
import torch.nn.functional as F

STRATEGIES = ('paired_mean', 'paired_max', 'ohem', 'random_pair_max')


def box_mil(logits: torch.Tensor, rect: tuple[int, int, int, int]) -> torch.Tensor:
    """One positive object; rect is half-open (x0,y0,x1,y1) in pixel indices."""
    x0, y0, x1, y1 = rect
    if not (0 <= x0 < x1 <= logits.shape[-1] and
            0 <= y0 < y1 <= logits.shape[-2]):
        raise ValueError(f'Invalid nonempty MIL rectangle: {rect}')
    return F.softplus(-logits[y0:y1, x0:x1].amax())


def reduce_extra(losses: torch.Tensor, strategy: str,
                 random_pairs: list[list[int]] | None = None) -> torch.Tensor:
    """Same 2N object/window terms; N<2 is disabled for ALL strategies."""
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    if losses.ndim != 2 or losses.shape[1] != 2:
        raise ValueError(f'Expected [N,2], got {tuple(losses.shape)}')
    n = losses.shape[0]
    if n < 2:
        return losses.sum() * 0.0
    if strategy == 'paired_mean':
        return losses.mean()
    if strategy == 'paired_max':
        return losses.max(dim=1).values.mean()
    if strategy == 'ohem':
        return losses.flatten().topk(n).values.mean()
    if random_pairs is None or len(random_pairs) != n:
        raise ValueError('Exactly N precomputed random pairs are required')
    indices = [i for pair in random_pairs for i in pair]
    if sorted(indices) != list(range(2*n)):
        raise ValueError('Random pairs must use each of the 2N terms exactly once')
    if any(len(pair) != 2 or pair[0]//2 == pair[1]//2 for pair in random_pairs):
        raise ValueError('Random pairs must not rejoin the same target/frame')
    index = torch.as_tensor(random_pairs, dtype=torch.long, device=losses.device)
    return losses.flatten()[index].max(dim=1).values.mean()


def pilot_loss(logits: torch.Tensor, positive_boxes: list[list[tuple]],
               background_masks: torch.Tensor, extra_specs: list[dict],
               strategy: str, random_pairs: list[list[int]]) -> tuple[torch.Tensor, dict]:
    """logits[B=4,T=40,H=128,W=128], object-balanced then window-balanced base."""
    if logits.ndim != 4 or logits.shape[0] != 4:
        raise ValueError('Expected four windows in two adjacent pairs')
    if background_masks.shape != logits.shape or background_masks.dtype != torch.bool:
        raise ValueError('Background masks must be bool and match logits')
    window_positive, window_negative = [], []
    for wi in range(4):
        terms = [box_mil(logits[wi, slot], rect)
                 for slot, rect in positive_boxes[wi]]
        window_positive.append(torch.stack(terms).mean() if terms else logits[wi].sum()*0.0)
        selected = logits[wi][background_masks[wi]]
        window_negative.append(F.softplus(selected).mean() if selected.numel()
                               else logits[wi].sum()*0.0)
    pos = torch.stack(window_positive).mean()
    neg = torch.stack(window_negative).mean()
    extra_terms = []
    for spec in extra_specs:
        wi = 2 * spec['pair_index']
        sa, sb = spec['slots']
        rect = tuple(spec['rect'])
        extra_terms.append(torch.stack([box_mil(logits[wi, sa], rect),
                                        box_mil(logits[wi+1, sb], rect)]))
    terms = torch.stack(extra_terms) if extra_terms else logits.new_empty((0, 2))
    extra = reduce_extra(terms, strategy, random_pairs)
    total = pos + neg + extra  # Frozen lambda=1; no score-dependent sample inclusion.
    return total, {'positive': pos, 'negative': neg, 'extra': extra,
                   'extra_objects': len(extra_specs),
                   'extra_disabled_n_lt_2': int(len(extra_specs) < 2),
                   'base_objects_per_window': [len(x) for x in positive_boxes],
                   'background_pixels_per_window': [int(x.sum()) for x in background_masks]}


def self_test() -> dict:
    # Distinct maxima give distinct analytic gradients; the four reductions differ.
    expected = {'paired_mean': (2.5, [[.25,.25],[.25,.25]]),
                'paired_max': (3., [[0.,.5],[0.,.5]]),
                'ohem': (3.5, [[0.,0.],[.5,.5]]),
                'random_pair_max': (3.5, [[0.,0.],[.5,.5]])}
    for strategy, (value, gradient) in expected.items():
        x = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
        y = reduce_extra(x, strategy, [[0,2],[1,3]])
        torch.testing.assert_close(y, torch.tensor(value))
        y.backward()
        torch.testing.assert_close(x.grad, torch.tensor(gradient))
        for n in [0,1]:
            small = torch.ones((n,2), requires_grad=True)
            z = reduce_extra(small, strategy, [])
            z.backward()
            assert z.item() == 0 and torch.count_nonzero(small.grad) == 0
    # This case distinguishes random regrouping from global OHEM.
    x = torch.tensor([[1.,4.],[2.,3.]], requires_grad=True)
    random_value = reduce_extra(x, 'random_pair_max', [[0,2],[1,3]])
    torch.testing.assert_close(random_value, torch.tensor(3.))
    torch.testing.assert_close(reduce_extra(x, 'ohem'), torch.tensor(3.5))
    random_value.backward()
    torch.testing.assert_close(x.grad, torch.tensor([[0.,.5],[.5,0.]]))
    z = torch.tensor([[0., 1., 100.], [2., 3., 100.]], requires_grad=True)
    b = box_mil(z, (0,0,2,2))
    torch.testing.assert_close(b, F.softplus(torch.tensor(-3.)))
    b.backward()
    assert z.grad[1,1] < 0 and torch.count_nonzero(z.grad) == 1
    try:
        reduce_extra(torch.ones((2,2)), 'random_pair_max', [[0,1],[2,3]])
    except ValueError:
        pass
    else:
        raise AssertionError('Same-target random pairs were accepted')
    return {'status':'passed', 'checks':['analytic four-reduction values and gradients',
            'all-strategy N<2 zero', 'random regrouping differs from global OHEM',
            'half-open box max gradient', 'invalid pair rejection']}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test()))
