"""Fourier x2 sampling and classical reconstruction of a single periodic channel.

All image arrays are (y,x); offsets are (dx,dy) in LR pixels. Positive dx
samples farther to the right in the reference. All FFTs use forward
normalization so the LR coefficient is the sum of its four shifted HR aliases.
"""
import numpy as np


def _check_image(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2-D image")
    return image


def _check_shifts(shifts):
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.ndim != 2 or shifts.shape[1] != 2 or not len(shifts) or not np.isfinite(shifts).all():
        raise ValueError("Expected finite (dx,dy) offsets with shape [T,2]")
    return shifts


def _pack(array):
    """Group the four aliased HR frequencies above each LR frequency."""
    h, w = array.shape[0] // 2, array.shape[1] // 2
    return np.stack([array[i*h:(i+1)*h, j*w:(j+1)*w]
                     for i in range(2) for j in range(2)], axis=-1)


def _unpack(grouped):
    return np.block([[grouped[..., 0], grouped[..., 1]],
                     [grouped[..., 2], grouped[..., 3]]])


def _real_ifft(spectrum):
    image = np.fft.ifft2(spectrum, axes=(-2, -1), norm="forward")
    if np.max(np.abs(image.imag)) > 2e-9 * max(1.0, np.max(np.abs(image.real))):
        raise ValueError("Fourier coefficients violate real-image symmetry")
    return image.real


def prepare_target(hr):
    """Discard HR Nyquist axes for a strictly real, fractionally shifted target."""
    hr = _check_image(hr)
    height, width = hr.shape
    if height % 2 or width % 2:
        raise ValueError("x2 samples require even HR dimensions")
    spectrum = np.fft.fft2(hr, norm="forward")
    spectrum[height//2, :] = 0
    spectrum[:, width//2] = 0
    return _real_ifft(spectrum)


def _operator(shape, shifts, blur_sigma):
    """Rows of the observation matrix at each LR frequency [h,w,T,4]."""
    if not np.isfinite(blur_sigma) or blur_sigma < 0:
        raise ValueError("Blur sigma is a nonnegative HR pixel scale")
    h, w = shape
    fy, fx = np.meshgrid(np.fft.fftfreq(2*h), np.fft.fftfreq(2*w), indexing="ij")
    transfer = np.exp(-2 * np.pi**2 * blur_sigma**2 * (fx**2 + fy**2))
    transfer[h, :] = 0
    transfer[:, w] = 0
    fx, fy = _pack(fx)[:, :, None, :], _pack(fy)[:, :, None, :]
    phase = fx * shifts[None, None, :, 0, None] + fy * shifts[None, None, :, 1, None]
    return _pack(transfer)[:, :, None, :] * np.exp(4j * np.pi * phase)


def observe(hr, shifts, blur_sigma=0.0):
    """Return real LR observations [T,h,w], sampled at reference position + offset."""
    hr = _check_image(hr)
    if any(size % 2 for size in hr.shape):
        raise ValueError("x2 samples require even HR dimensions")
    shifts = _check_shifts(shifts)
    operator = _operator((hr.shape[0]//2, hr.shape[1]//2), shifts, blur_sigma)
    hr_coeff = _pack(np.fft.fft2(hr, norm="forward"))
    lr_coeff = np.einsum("hwtk,hwk->thw", operator, hr_coeff)
    return _real_ifft(lr_coeff)


def _bilinear_prior(lr):
    """Periodic bilinear interpolation with LR samples centered at even HR indices."""
    h, w = lr.shape
    out = np.empty((2*h, 2*w), np.float64)
    out[::2, ::2] = lr
    out[::2, 1::2] = 0.5 * (lr + np.roll(lr, -1, axis=1))
    out[1::2] = 0.5 * (out[::2] + np.roll(out[::2], -1, axis=0))
    return out


def reconstruct(observations, shifts, ridge=0.05, blur_sigma=0.0, reference=0):
    """Return prior, scalar, classic and swapped HR arrays without output clipping.

    classic minimizes mean_t ||A_t x - y_t||² + ridge ||x-prior||² in
    each normalized Fourier alias block. scalar takes the same gradient
    with step 1/(lambda_max(G) + ridge). swapped uses a diagnostic inverse
    based on transposed offsets but retains the original residual and RHS.
    """
    observations = np.asarray(observations, dtype=np.float64)
    shifts = _check_shifts(shifts)
    if (observations.ndim != 3 or len(observations) != len(shifts)
            or not np.isfinite(observations).all() or min(observations.shape[1:]) < 2):
        raise ValueError("Expected finite LR observations [T,h,w], one per offset")
    if not 0 <= reference < len(shifts) or not np.allclose(shifts[reference], 0, atol=1e-12, rtol=0):
        raise ValueError("Reference observation must have shift zero")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Expected a strictly positive ridge")
    prior = _bilinear_prior(observations[reference])
    x0 = _pack(np.fft.fft2(prior, norm="forward"))
    op = _operator(observations.shape[1:], shifts, blur_sigma)
    obs = np.moveaxis(np.fft.fft2(observations, axes=(-2,-1), norm="forward"), 0, -1)
    residual = obs - (op @ x0[..., None])[..., 0]
    adj = np.swapaxes(op.conj(), -1, -2)
    rhs = adj @ residual[..., None] / len(shifts)
    gram = adj @ op / len(shifts)
    damp = ridge * np.eye(4)
    full = np.linalg.solve(gram + damp, rhs)[..., 0]
    top = np.linalg.eigvalsh(gram)[..., -1]
    scalar = rhs[..., 0] / (top + ridge)[..., None]
    wrong_op = _operator(observations.shape[1:], shifts[:, ::-1], blur_sigma)
    wrong_gram = np.swapaxes(wrong_op.conj(), -1, -2) @ wrong_op / len(shifts)
    swapped = np.linalg.solve(wrong_gram + damp, rhs)[..., 0]
    result = {"prior": prior}
    for key, correction in (("scalar", scalar), ("classic", full), ("swapped", swapped)):
        result[key] = _real_ifft(_unpack(x0 + correction))
    return result


def selfcheck():
    """Direct spatial sampler, rank, axis-equivariance, sign, and linear solve tests."""
    rng = np.random.default_rng(46)
    hr = prepare_target(rng.standard_normal((30, 38)))
    shifts = np.array([(0,0), (.25,0), (-.25,0), (.5,0), (-.5,0)])
    sigma = 0.45
    h, w = hr.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    transfer = np.exp(-2*np.pi**2 * sigma**2 * (fx**2 + fy**2))
    fft = np.fft.fft2(hr, norm="forward")
    direct = [_real_ifft(fft * transfer * np.exp(4j*np.pi*(fx*dx + fy*dy)))[::2,::2]
              for dx,dy in shifts]
    lr = observe(hr, shifts, sigma)
    np.testing.assert_allclose(lr, direct, atol=1e-12)
    integer_lr = observe(hr, np.array([(0,0), (1,0)]))
    np.testing.assert_allclose(integer_lr[1], np.roll(integer_lr[0], -1, axis=1), atol=1e-12)
    op = _operator(lr.shape[1:], shifts, 0)
    block = op[2, 3]
    transposed = _operator(lr.shape[1:], shifts[:, ::-1], 0)[2, 3]
    assert np.linalg.matrix_rank(block) == np.linalg.matrix_rank(transposed) == 2
    np.testing.assert_allclose(np.linalg.svd(block, compute_uv=False),
                               np.linalg.svd(transposed, compute_uv=False), atol=1e-12)
    xhat = reconstruct(lr, shifts, blur_sigma=sigma)
    flipped = reconstruct(lr.transpose(0,2,1), shifts[:, ::-1], blur_sigma=sigma)
    for key in xhat:
        np.testing.assert_allclose(xhat[key].T, flipped[key], atol=1e-11)
    prior = np.fft.fft2(xhat["prior"], norm="forward")
    op = _operator(lr.shape[1:], shifts, sigma)[2, 3]
    obs = np.fft.fft2(lr, axes=(-2,-1), norm="forward")[:, 2, 3]
    x0 = _pack(prior)[2, 3]
    # Independent augmented least-squares check against the normal-equation solution.
    augmented = np.concatenate([op, np.sqrt(len(shifts)*0.05)*np.eye(4)])
    rhs = np.r_[obs, np.sqrt(len(shifts)*0.05)*x0]
    direct_coef = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    solved = _pack(np.fft.fft2(xhat["classic"], norm="forward"))[2, 3]
    np.testing.assert_allclose(direct_coef, solved, atol=1e-12)
    reproject_error = {}
    for key, image in xhat.items():
        reproject_error[key] = float(np.mean((observe(image, shifts, sigma)-lr)**2))
    assert reproject_error["classic"] < reproject_error["scalar"] < reproject_error["prior"]
    return {"direct_sampler_max_difference": float(np.max(np.abs(lr - direct))),
            "augmented_lstsq_max_difference": float(np.max(np.abs(direct_coef-solved))),
            "reprojection_mse": reproject_error}


if __name__ == "__main__":
    import json
    print(json.dumps(selfcheck(), indent=2))
