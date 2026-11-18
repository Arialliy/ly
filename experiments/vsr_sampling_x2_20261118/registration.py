"""Global translation from a pair of LR frames. No HR, axis, or oracle inputs."""
import numpy as np


def _gray(frame):
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame @ np.array([0.299, 0.587, 0.114])
    if frame.ndim == 3 and frame.shape[-1] == 1:
        return frame[:, :, 0]
    if frame.ndim == 2:
        return frame
    raise ValueError("Expected [H,W], [H,W,1], or [H,W,3] LR frame")


def _edge_taper(n):
    # A short raised-cosine ramp limits edge wraparound on natural video crops.
    distance = np.minimum(np.arange(n), np.arange(n)[::-1])
    position = np.minimum(1.0, (distance + 0.5) / max(2, n // 8))
    return 0.5 - 0.5 * np.cos(np.pi * position)


def estimate_translation(reference, moving):
    """Return LR (dx,dy) for moving(z) approximately equal to reference(z+delta).

    Common input support and coordinates are required. Only frequencies up to
    0.23 cycles/LR pixel take part. An integer phase-correlation estimate
    seeds a robust continuous phase-plane fit. Flat frames return zero.
    """
    ref, mov = _gray(reference), _gray(moving)
    if ref.shape != mov.shape or min(ref.shape) < 12:
        raise ValueError("Expected equal frame shapes with both LR dimensions >=12")
    if not np.isfinite(ref).all() or not np.isfinite(mov).all():
        raise ValueError("LR frame contains nonfinite values")
    h, w = ref.shape
    taper = _edge_taper(h)[:, None] * _edge_taper(w)[None, :]
    def transform(frame):
        mean = (frame * taper).sum() / taper.sum()
        return np.fft.fft2((frame - mean) * taper)
    cross = transform(mov) * transform(ref).conj()
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    radius = np.hypot(fx, fy)
    band = (radius > 0) & (radius <= 0.23)
    amplitude = np.abs(cross)
    peak = amplitude[band].max()
    if peak <= 0:
        return np.zeros(2)
    valid = band & (amplitude > peak * 1e-10)
    if np.count_nonzero(valid) < 2:
        return np.zeros(2)
    phase = cross[valid] / amplitude[valid]
    weight = np.sqrt(amplitude[valid] / peak)
    angles = 2 * np.pi * np.column_stack((fx[valid], fy[valid]))
    spectrum = np.zeros_like(cross)
    spectrum[valid] = phase * weight
    # fft2 here (negative exponent) makes the maximum have the shift-to-sample sign.
    corr = np.fft.fft2(spectrum).real
    max_motion = max(2, min(h, w) // 4)
    grid = np.arange(-max_motion, max_motion + 1)
    scores = corr[np.ix_(grid % h, grid % w)]
    best = scores.max()
    ties = np.argwhere(scores >= best - max(abs(best), 1.0) * 1e-10)
    closest = ties[np.argmin(np.sum(grid[ties] ** 2, axis=1))]
    delta = grid[closest[::-1]].astype(np.float64)
    for _ in range(8):
        error = np.angle(phase * np.exp(-1j * (angles @ delta)))
        # A 0.35-radian Huber transition lessens alias and local motion outliers.
        robust = weight * np.minimum(1.0, 0.35 / np.maximum(np.abs(error), 1e-12))
        design = angles * np.sqrt(robust)[:, None]
        step = np.linalg.lstsq(design, error * np.sqrt(robust), rcond=1e-6)[0]
        delta += step
        if np.linalg.norm(step) < 1e-5:
            break
    return delta


def estimate_shifts(frames, reference=0):
    """Return [T,2] (dx,dy) LR displacements relative to frames[reference]."""
    frames = np.asarray(frames)
    if frames.ndim not in (3, 4) or not 0 <= reference < len(frames):
        raise ValueError("Expected a sequence of LR images and valid reference index")
    shifts = np.zeros((len(frames), 2), dtype=np.float64)
    for i, frame in enumerate(frames):
        if i != reference:
            shifts[i] = estimate_translation(frames[reference], frame)
    return shifts


def self_check():
    """Verify sign and subpixel accuracy with known band-limited LR translations."""
    rng = np.random.default_rng(381)
    h, w = 96, 112
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    radius = np.hypot(fx, fy)
    spectrum = np.fft.fft2(rng.normal(size=(h, w))) * ((radius > 0.03) & (radius < 0.175))
    truth = np.array([[0, 0], [0.25, -0.50], [-0.50, 0.25], [2.35, -1.40], [-2.05, 2.50]])
    frames = [np.fft.ifft2(spectrum * np.exp(2j * np.pi * (fx*dx + fy*dy))).real for dx, dy in truth]
    estimates = estimate_shifts(frames)
    errors = np.linalg.norm(estimates - truth, axis=1)
    assert np.max(errors) < 0.09, (truth, estimates, errors)
    return dict(truth=truth.tolist(), estimates=estimates.tolist(), max_lr_endpoint_error=float(max(errors)))


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2))
