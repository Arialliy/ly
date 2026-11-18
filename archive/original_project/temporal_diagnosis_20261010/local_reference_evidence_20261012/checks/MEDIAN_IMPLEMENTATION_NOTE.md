# Median implementation compatibility correction

Before any formal training, the first CPU-only formula receipt was found insufficient to establish CUDA compatibility with the frozen `torch.use_deterministic_algorithms(True)` contract. PyTorch 2.7 lists CUDA `kthvalue` among operations that raise under this setting, including calls without gradients.

The original receipt is preserved unchanged as `CPU_KTHVALUE_ONLY_SUPERSEDED.json`. The implementation now uses `torch.sort(values, dim=-1, stable=True).values`, then averages the central two order statistics for the 104-element annulus. The annulus, mathematical median and MAD, missing-reference behavior, model, and training settings are unchanged. Deterministic algorithms remain enabled. Sorting requires additional values and index buffers; actual GPU memory is checked in the separate preflight.

The replacement is an implementation compatibility correction before any training, not a failed model trial or a new scientific variant. The existing CPU formula and integration tests are rerun into `MODULE_CPU_TESTS.json`; CUDA execution is accepted only after the separate GPU gate.

Source: https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html
