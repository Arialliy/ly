"""Deterministic CPU acceptance checks for the first standalone detector operator."""
import hashlib
import json
from pathlib import Path
import sys

import torch

from span_relations import span_relations

HERE = Path(__file__).resolve().parent
SAMPLER = HERE.parent / "motion_joint_factorial"
sys.path.insert(0, str(SAMPLER))
from motion_evidence import extract_evidence, prepare_window


def near(actual, expected):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype),
                               rtol=1e-5, atol=1e-6)


def toy_checks():
    # Point 0 tightens along vx only. Point 1 is flat; point 2 is already sharp.
    # Point 3 has an anchor weaker than a neighboring velocity.
    strengths = torch.full((4, 2, 2, 4, 5, 5), 8.0)
    strengths[0, :, :, 1:, 2, 3] = torch.tensor([7., 5., 2.])
    strengths[2, :, :, 1:, 2, 3] = 2.
    strengths[3, :, :, 1:, 2, 3] = 9.
    patches = torch.zeros(*strengths.shape, 3, 3)
    patches[..., 1, 1] = strengths
    valid = torch.ones_like(strengths)
    values, mask = span_relations(dict(patches=patches, valid=valid))
    assert values.shape == (4, 2, 2, 3, 5, 5) and mask.dtype == torch.bool and mask.all()
    reconstructed = values.cumsum(3)
    profile = reconstructed[0, 0, 0, :, 2, 3]
    near(profile, [1/15.1, 3/13.1, 6/10.1])
    near(reconstructed[0, 0, 0, :, 3, 2], [0, 0, 0])
    near(values[1, 0, 0, :, 2, 3], [0, 0, 0])
    near(values[2, 0, 0, :, 2, 3], [6/10.1, 0, 0])
    near(reconstructed[3, 0, 0, :, 2, 3], [-1/17.1]*3)
    cold, cold_mask = span_relations(dict(patches=-patches, valid=valid))
    assert torch.equal(cold, values) and torch.equal(cold_mask, mask)

    # Only one neighbor has an incomplete medium patch, then an incomplete
    # short anchor invalidates the entire short grid and its adjacent difference.
    valid[0, 1, 1, 2, 2, 3] = .75
    valid[2, 0, 0, 1, 2, 2] = .75
    partial, active = span_relations(dict(patches=patches, valid=valid))
    assert active[0, 1, 1, :, 2, 3].tolist() == [True, False, False]
    assert active[0, 1, 1, :, 3, 2].all()
    assert active[0, 0].all() and active[0, 1, 0].all()
    assert not active[2, 0, 0, :2].any() and active[2, 0, 0, 2].all()
    assert (partial[~active] == 0).all()
    return {"short_medium_long_Q_vx_neighbor": profile.tolist(),
            "short_medium_long_Q_vy_neighbor": reconstructed[0, 0, 0, :, 3, 2].tolist(),
            "changing_encoded_channels": values[0, 0, 0, :, 2, 3].tolist(),
            "steady_sharp_encoded_channels": values[2, 0, 0, :, 2, 3].tolist(),
            "stronger_neighbor_signed_Q": reconstructed[3, 0, 0, 0, 2, 3].item(),
            "cold_target_checked": True, "validity_isolated_by_anchor_exchange_and_velocity": True}


def sampler_check():
    y, x = torch.meshgrid(torch.arange(65), torch.arange(65), indexing="ij")
    frames = torch.stack([100. + 90.*torch.exp(-((x-32-t)**2 + (y-32+t)**2) / 2.)
                          for t in range(-8, 9)])
    points = torch.tensor([[32, 32], [0, 0]])
    evidence = extract_evidence(prepare_window(frames), points, selection="split")
    near(evidence["velocity"][0, :, 1], [[1., -1.], [1., -1.]])
    relations, mask = span_relations(evidence)
    assert mask[0].all()
    assert (~mask[1]).any() and not relations[1][~mask[1]].any()
    assert not relations[..., 2, 2].any()
    # Check a noncentral velocity using the actual normalized frames at integer
    # displacement. Preserve the ordering (vy,vx) rather than swapping x and y.
    normalized = prepare_window(frames)
    spans = ((1,-2,-3,4), (1,-2,-5,6), (-3,4,7,-8))
    references = []
    for span in spans:
        a = torch.stack([normalized[t+8,32-t,32+t] for t in span]).mean().abs()
        b = torch.stack([normalized[t+8,32-t,32+2*t] for t in span]).mean().abs()
        references.append((a-b)/(a+b+.1))
    actual = relations.cumsum(3)[0, 0, 1, :, 2, 4]
    near(actual, torch.stack(references))
    return {"shape": list(relations.shape), "known_velocity_xy": [1, -1],
            "estimated_velocity_xy_per_exchange": evidence["velocity"][0,:,1].tolist(),
            "independent_Q_at_one_neighbor": [q.item() for q in references],
            "reconstructed_Q_at_one_neighbor": actual.tolist(), "border_mask_checked": True}


if __name__ == "__main__":
    torch.set_num_threads(1)
    measurements = {"torch": torch.__version__, "device": "cpu", "toy": toy_checks(),
                    "raw_sampler_integration": sampler_check(), "status": "pass",
                    "scope": "Synthetic correctness and existing sampler interface only; no training or collected video.",
                    "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in [Path(__file__), HERE/"span_relations.py", SAMPLER/"motion_evidence.py"]}}
    path = HERE / "runs/relation_check.json"
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(measurements, indent=2) + "\n")
    temp.replace(path)
    print(json.dumps(measurements))
