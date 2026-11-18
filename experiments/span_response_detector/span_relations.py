"""Cross-span comparisons on the traffic sampler's fixed 2-D velocity grid."""
import torch


def span_relations(evidence):
    """Return (values, mask) with shape [P,2 exchanges,2 anchors,3 relations,5,5].

    Inputs are motion_evidence.extract_evidence's aligned signed-mean 'patches'
    [P,2,2,4 spans,5 vy,5 vx,3 y,3 x] and 'valid' frame fractions. Span 0
    is the whole clip; spans 1, 2 and 3 are the short, medium and long spans.
    The velocity at grid index [2,2] is the same proposed anchor across spans.
    Relation channels contain Q_short, Q_medium-Q_short and Q_long-Q_medium.
    Each Q compares two magnitudes from four complete frames. A difference is
    present only if both Q measurements are valid. Missing entries equal zero.
    """
    patches, fractions = evidence["patches"], evidence["valid"]
    if patches.shape[1:] != (2, 2, 4, 5, 5, 3, 3) or fractions.shape != patches.shape[:-2]:
        raise ValueError("Expected aligned [P,2,2,4,5,5,3,3] patches and matching valid fractions")

    response = patches[:, :, :, 1:, :, :, 1, 1].abs()
    anchor = response[..., 2:3, 2:3]
    relative = (anchor - response) / (anchor + response + 0.1)
    complete = fractions[:, :, :, 1:] == 1
    measured = complete & complete[..., 2:3, 2:3]
    short, medium, long = relative.unbind(3)
    short_ok, medium_ok, long_ok = measured.unbind(3)
    values = torch.stack((short, medium - short, long - medium), dim=3)
    mask = torch.stack((short_ok, short_ok & medium_ok, medium_ok & long_ok), dim=3)
    return torch.where(mask, values, 0.0), mask
