"""S1 geometry only: fixed hypotheses and differentiable per-time sampling."""
from .sampler import ALIGN_CORNERS, MissingHaloError, SampledEvidence, sample_clip_features, sample_context
from .velocity_grid import make_velocity_grid

__all__ = [
    "ALIGN_CORNERS", "MissingHaloError", "SampledEvidence", "make_velocity_grid",
    "sample_context", "sample_clip_features",
]
