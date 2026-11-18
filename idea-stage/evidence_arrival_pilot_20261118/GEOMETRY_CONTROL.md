# Geometry controls added after the frozen DUF probe

All four methods use the same 32 worlds and the same scoring readout. Ground-truth labels and the carrier are unavailable to the new reconstructors. The public carrier is supplied to the **evaluation** readout only.

1. Known sampling positions. At a measured address copy the observed LR RGB pixel to its HR address. For unmeasured odd columns or odd rows use bilinear center interpolation as a prior. In particular, an observation at (0.5,0) LR directly measures every odd column of the even HR rows.
2. Known sampling positions, classic Fourier reconstruction. Apply the existing ×2 Tikhonov reconstruction with fixed ridge 0.05, no blur, center slot 2, to all five frames. Scores depend only on output luminance. For this linear, per-channel solver, reconstructing luminance is identical to evaluating the luminance of separately reconstructed RGB channels.
3. Estimated sampling positions, same classic solver. Apply the pre-existing LR-only subpixel registration routine to the same five RGB input slots. It receives neither the moving slot, the true displacement nor the carrier. Record endpoint error at the moved slot. Both known and estimated Fourier methods use the same code, prior and regularization.

Run all-neutral, slot 3 and slot 4 separately; map these to the original early/late deadlines. Each external slot is used exactly once in these classical methods. Treat changes across order as a numerical sanity check; these batch solvers are permutation-invariant for non-reference observations. Record all worlds including failed shifts. Do not interpret this post hoc comparison as a new benchmark.
