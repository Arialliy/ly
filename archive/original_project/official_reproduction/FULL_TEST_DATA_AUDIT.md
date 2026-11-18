# Full official NUDT-MIRSDT test data audit

**PASS** — all 2,000 frames from all 20 official test sequences are present as 6,000 original PNG files.

- Data root: `/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_reproduction/data/NUDT-MIRSDT`
- Official repository: https://github.com/TinaLRJ/DeepPro, pinned commit `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`.
- Authoritative split: `datasets/NUDT-MIRSDT/test.txt`; copied byte for byte, SHA-256 `eb039aa35aaabab2eae559b9d627e53d3fa114eb0410f343498f154c81a9d5d0`.
- Official archive: https://drive.google.com/file/d/1mjFJmsaNoLhALaaL1_3N4qpAPCiBCqXy/view; archive size 1,027,614,117 bytes.
- Range bodies received: 45,557,179 bytes; total body upper bound including public download form: 45,577,179 bytes, within the 400,000,000-byte budget.
- Only official test PNG triples were extracted; no training sequence was extracted. Public ZIP range requests can include small adjacent bytes to cover variable ZIP headers.
- Images, masks and centroid masks have matching names and dimensions for every frame. Each PNG passed Pillow verification and full decoding.
- All 6,000 CRC32 values match the archive's central directory; SHA-256, dimensions and modes are recorded per file in `data_prep/FILE_CHECKSUMS.json`.
- Dimensions (width, height), counted across all three kinds: `{'(300, 200)': 300, '(313, 261)': 300, '(334, 221)': 300, '(329, 215)': 300, '(275, 164)': 300, '(303, 200)': 300, '(354, 237)': 300, '(315, 224)': 300, '(298, 215)': 300, '(347, 210)': 300, '(280, 187)': 300, '(287, 192)': 300, '(320, 240)': 300, '(320, 256)': 300, '(283, 182)': 300, '(300, 209)': 300, '(233, 161)': 300, '(257, 257)': 300, '(291, 215)': 300, '(379, 246)': 300}`; modes: `{'L': 6000}`.
- The `.mat` entries in the official split identify frame stems. The official loader takes sequence names in first occurrence order, then sorts the PNG filenames. The downloaded layout matches both conventions.
- Official SNR≤3 subset: Sequence47, Sequence56, Sequence59, Sequence76, Sequence92, Sequence101, Sequence105, Sequence119 (800 frames). Membership comes from the pinned official README, without reestimating SNR.

## Official sequence order and coverage

| Sequence | Frames | PNG files | SNR≤3 |
|---|---:|---:|---|
| Sequence85 | 100 | 300 | no |
| Sequence86 | 100 | 300 | no |
| Sequence87 | 100 | 300 | no |
| Sequence88 | 100 | 300 | no |
| Sequence89 | 100 | 300 | no |
| Sequence90 | 100 | 300 | no |
| Sequence91 | 100 | 300 | no |
| Sequence92 | 100 | 300 | yes |
| Sequence93 | 100 | 300 | no |
| Sequence94 | 100 | 300 | no |
| Sequence95 | 100 | 300 | no |
| Sequence96 | 100 | 300 | no |
| Sequence97 | 100 | 300 | no |
| Sequence47 | 100 | 300 | yes |
| Sequence56 | 100 | 300 | yes |
| Sequence59 | 100 | 300 | yes |
| Sequence76 | 100 | 300 | yes |
| Sequence101 | 100 | 300 | yes |
| Sequence105 | 100 | 300 | yes |
| Sequence119 | 100 | 300 | yes |

## Reproducibility artifacts

- `data_prep/prepare_full_test.py`: bounded acquisition and verification script.
- `data_prep/SELECTED_ARCHIVE_MEMBERS.json`: exact ZIP members and central CRC32 values.
- `data_prep/CENTRAL_DIRECTORY_PROVENANCE.json` and `ARCHIVE_CENTRAL_DIRECTORY.bin`: CRC authority and metadata hash.
- `data_prep/DOWNLOAD_STATE.json`: completed groups and transferred range bytes.
- `data_prep/FILE_CHECKSUMS.json`: per-file SHA-256, CRC32, sizes and decode results.

Archive authenticity is tied to the URL linked by the pinned official repository. Per-file SHA-256 values establish local reproducibility; no upstream cryptographic digest is published or asserted.
