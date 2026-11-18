# Next three fixed real development clips

## Selection and provenance

- Used exactly `INSPECTION_QUEUE.json` zero-based entries 1, 2, 3, preserving the frozen order. No new screen and no detector outputs were read.
- Only the selected 160 thermal JPEG frames per clip were extracted from the officially linked VT-Tiny-MOT archive, train2017/01. Native 640 × 512 grayscale L pixels; no resize or normalization.
- All source image IDs, annotation IDs, track IDs and annotations for every selected frame are retained. Bounding-box polygons are not treated as true pixel masks.
- Each sequence name is absent from both official DeepPro RGB-T train and test lists. This exact-name exclusion does not prove independent recording scenes or trajectories. These remain extra development clips.

## Delivered clips

| Candidate | Raw frames | All target annotations | Track IDs | Raw JPEG bytes |
|---|---:|---:|---:|---:|
| DJI_0028_4__4400005__00285__00444 | 285–444 | 1280 | 9 | 1,468,141 |
| DJI_0032_5__4801536__00063__00222 | 63–222 | 1748 | 57 | 4,349,886 |
| DJI_0047_3__4621500__00000__00159 | 0–159 | 665 | 10 | 7,232,562 |

## Verification

- Every JPEG was fully decoded and its exact byte SHA-256 was checked against the extraction record. The ZIP members use data descriptors (flag 8), so the supplied extractor did not verify CRC; `crc_verified: false` is retained and no CRC verification is claimed.
- Each selected target has exactly one annotation per selected frame; original IDs and boxes match the frozen candidate.
- Each `raw_frames.npy` is uint8 with shape `(160, 512, 640)`. NumPy 2.2.5 in the immutable CPU Docker image independently loaded every NPY and verified every frame pixel SHA-256 against its decoded JPEG record.
- `FRAME_MANIFEST.json`, `annotations.json`, `DOWNLOAD_PROVENANCE.json`, `VALIDATION.json`, and `raw_frames.npy` are saved in each candidate directory.
- Total returned HTTP body bytes, including public forms and retries: **12,928,847**, below the 100,000,000-byte budget. `NETWORK_TRANSFER.json` records each request.
- No GPU or inference was executed by acquisition. Camera motion, visibility, and detector failure remain unverified.

Supporting records: `FIXED_SELECTION.json`, `ACQUISITION_REPORT.json`, `INDEPENDENT_NPY_LOAD_CHECK.json`.
