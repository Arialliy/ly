# Native logit export verification

Status: **PASS**. Sixteen windows across two clips were re-exported in 28.24 seconds; independent CPU verification took 13.34 seconds.

| Sequence | Plan | Pixels checked | Probability differences | 0.5 mask differences | Logit range |
|---|---|---:|---:|---:|---|
| DJI_0026_2 | O | 52428800 | 0 | 0 | -95.932999 to 47.237015 |
| DJI_0026_2 | S | 52428800 | 0 | 0 | -95.932999 to 47.237015 |
| DJI_0067_2 | O | 52428800 | 0 | 0 | -140.645142 to 86.698868 |
| DJI_0067_2 | S | 52428800 | 0 | 0 | -122.798866 to 77.129333 |

The 16 saved window maps contain 209,715,200 pixels. Their CPU sigmoid probabilities are bitwise identical to the old window predictions. The four arrangement maps contain another 209,715,200 pixel comparisons with the old O/S fused predictions. Independent maximum-logit recomputation, probability values, probability masks at 0.5, and native-logit masks at zero all agree exactly.

All 30 frozen source/input fingerprints remained unchanged. All 24 array fingerprints passed. The original and current PyTorch/CUDA versions match (2.7.0+cu128 / 12.8), as do seed, normalization, tiling and precision. Peak GPU allocated memory was 3.839 GiB. Both containers exited 0 without OOM.

The raw logits resolve saturation without inverse sigmoid. Native logit argmax remains a diagnostic field: it must not replace the old probability component peak or its tie-breaking in actual detection matching.

Provenance: [pipeline contract](PIPELINE.md), [export metadata](COMPLETE.json), [independent verification](INDEPENDENT_VERIFICATION.json), [warm reuse check](WARM_REUSE_CHECK.json).
