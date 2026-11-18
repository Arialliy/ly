# Real infrared sequence data entry audit — 2026-10-09

**Acquired and ready:** a real 160-frame thermal clip with an original 11×11-pixel ship track, plus 40 synthetic NUDT frames with original masks/centroids for native protocol checks. Only public HTTP byte ranges were downloaded; no full archive, GPU or model inference was used.

## Newly acquired real small-target development clip

- Images: `runtime/public_subset_vt/VT-Tiny-MOT/train2017/DJI_0024_2/01/00000.jpg` through `00159.jpg`.
- **160 real grayscale frames, 640×512, 15 fps** from source video metadata. All images pass PIL decoding/header verification.
- Original `track_id = 3300000`, category **ship**; every selected frame has an **11×11** box. Its annotated center displacement has median 0, p90 0 and maximum 1 pixel/frame. This is annotation-coordinate residence, not proof of world-stationarity or a motion-to-stop event.
- Selection used only annotations: VT train sequences absent from both DeepPro manifests; one original track with `ignore=iscrowd=0`, width/height ≤12 over 160 consecutive numeric image names; lexicographic sequence, numeric track ID, earliest frame as tie breakers. There were 290 qualifying tracks across 26 sequences. No model outputs were consulted.
- This is an **extra development subset of VT train2017**, not an official validation split. Absence from DeepPro manifests does not establish independent scene identity.
- `runtime/public_subset_vt/annotations.json` retains **all 1,920 frame annotations**, original category/track/image IDs, ignore/crowd fields and rectangle polygons. Those rectangle polygons are not verified pixel segmentation masks. `FRAME_MANIFEST.json` preserves frame order and original image attributes; `SUBSET_VALIDATION.json` and `DOWNLOAD_PROVENANCE.json` record checks and SHA256s.

The [RGBT-Tiny repository explicitly links HGTMT for MOT labels](https://github.com/XinyiYing/RGBT-Tiny#mot-label). [HGTMT](https://github.com/xuqingyu26/HGTMT) supplies the public [VT-Tiny-MOT archive](https://drive.google.com/file/d/17CgPc3huCf-KMr-I9zFVqmV0hzBVky9w/view). Its archive contains 86 train and 30 test sequence names; all 45 DeepPro train names and 15 test names occur in the corresponding partition without crossing. This relation does not prove byte-identical images or identical annotation conventions.

## Native synthetic protocol sample

`runtime/public_subset_nudt/NUDT-MIRSDT/Sequence85/{images,masks,masks_centroid}/00001.png`–`00040.png` contains **40 original triples, 300×200 grayscale**, all verified. Sequence85 is the first sequence in the checked-out official DeepPro `datasets/NUDT-MIRSDT/test.txt`. This explicitly synthetic test sample is for loader/evaluator engineering only.

## Public archive access and budget

| Official-linked archive | Full archive bytes | Availability | Retrieved |
|---|---:|---|---|
| NUDT-MIRSDT.zip | 1,027,614,117 | Public form, HTTP206 range verified, ZIP header | Central directory + 40 original frame/mask/centroid triples |
| VT-Tiny-MOT.zip | 2,448,061,811 | Public form, HTTP206 range verified, ZIP header | Central directory + train IR annotation member + selected 160 IR frames |
| IRDST_real.rar | Displayed 4.61 GB | Official public Drive folder HTTP200 | Folder metadata only |

Subset transfer totals: **12,516,392 bytes** (VT images 1,162,347; VT source annotations 10,479,087; NUDT 874,958). ZIP central-directory/footer reads add 17,444,404 bytes. A separate public GT.zip annotation-only archive was inspected in memory (1,616,313 bytes). This remains below the **100 MB** initial budget; no full archive was fetched. Large source COCO JSON is retained losslessly as `runtime/VT_TRAIN_01_ANNOTATIONS.json.gz` (9.47 MB).

The reusable `runtime/fetch_public_zip_subset.py` uses the normal public Drive download form, requires exact HTTP206 ranges, enforces a byte budget, rejects path traversal/absolute paths and checks member sizes/CRCs. Public source URLs and SHA256s are saved without credentials. Both complete ZIP member manifests are saved as `VT_TINY_MOT_PUBLIC_ZIP_MANIFEST.json` and `NUDT_PUBLIC_ZIP_MANIFEST.json`.

## Initial local scan and Anti-UAV engineering clip

The initial local scan below found only Anti-UAV videos. Subsequent public subset acquisition above provides the real small-target clip now selected for development. The Anti-UAV validation clip remains available for a separate engineering smoke check.

## Initial deterministic Anti-UAV development clip

- Directory: `/home/ly/data/红外小目标数据集/Anti-UAV-RGBT/val/20190925_101846_1_4`
- Input: `infrared.mp4`; annotation: `infrared.json`.
- Rule: sort validation sequence directories lexicographically; choose the first with both files. No model result influenced selection.
- MP4 container: **640 × 512, 1,000 frames, 20 fps, 50 seconds**. Parsed `tkhd/mdhd/stts/stsz`: time scale 10,240, sample delta 512, sample count 1,000. Decoder has not yet checked the video.
- JSON: `exist` and `gt_rect` each contain 1,000 items; boxes use `[x,y,w,h]`. First box is `[210,224,30,22]`. First contiguous annotated visible span is zero-based frames **0–999**.
- Video SHA256: `6d873a57897005b62cd92bab982d692b78e15b1470f8b45351cda757fa7db372`.
- Label SHA256: `b3c30b23b4b0579ea2bdc421db31f729dd202d083a17d8613286556afa35fb82`.
- Local `label_new/val.json` and the val directory identify the development split. Original download checksum / exact archive provenance is unavailable. The [official Anti-UAV repository](https://github.com/ucas-vg/Anti-UAV) documents the annotation format's existence flag and start-frame corrections; this sequence is not among the listed correction cases.

## Validation-only annotation prescreen

All **67** paired validation sequences were scanned using annotations only. Requirement: at least **160 consecutive frames**, `exist == 1`, positive box size, and **w ≤ 12, h ≤ 12 in every frame**. **No qualifying run exists.**

As a transparent fallback, for each sequence find its 160-frame all-visible window minimizing the largest box side; break ties by earliest start, then rank sequences by bound and name. These are alternatives to the requested ≤12-pixel criterion, not qualifying results.

| Sequence | Zero-based span | Width range | Height range | Max-side bound | Center displacement median / p90 |
|---|---:|---:|---:|---:|---:|
| 20190925_101846_1_5 | 274–433 | 18–25 | 13–22 | 25 | 1.00 / 1.41 px per frame |
| 20190925_141417_1_6 | 111–270 | 23–27 | 17–26 | 27 | 1.00 / 1.12 px per frame |
| 20190925_141417_1_1 | 392–551 | 22–28 | 19–28 | 28 | 2.06 / 4.00 px per frame |

Full per-sequence width/height percentiles, center-displacement summaries, and candidates are in `VAL_ANNOTATION_PRESCREEN.json`. Box center displacement is in image coordinates. It cannot establish physical stationarity or a motion-to-stop event; annotation jumps and camera motion remain possible. Boxes are not the segmented target support.

## Initial local availability, verified by actual files

| Dataset | Actual content | Use decision |
|---|---|---|
| IRDST real | No images; empty scene/annotation directories | Unavailable |
| IRDST simulation | 1 PNG and 7,500 text files | Incomplete synthetic data; unusable temporal clip |
| NUDT-MIRSDT | 120 sequence directories but **0 images** | Empty placeholders; official source says synthesized |
| Anti-UAV-RGBT | 632 MP4, 635 JSON; IR pairs train 158 / val 67 / test 91 | Validation clips available for development |
| Anti-UAV410 | One local test sequence with 1,500 ordered JPGs and 1,500 labels; another with 960 JPGs and no label | Test excluded from development |
| RGBT-Tiny / SatVideoIRSDT | No matching local data found | Need original dataset |

Anti-UAV-RGBT has 160 train directories, but `20190926_200510_1_6` and `20190926_195921_1_8` lack paired IR files. Its videos and annotations occupy about 3.8 GB. Anti-UAV410's complete local sequence is `test/02_6319_1500-2999` (`000001.jpg`–`001500.jpg`, `IR_label.json`); its exact official split membership was not independently crosschecked. Local test inventory was read before the parent limited development selection to validation; no test inference was run.

[IRDST's official dataset page](https://xzbai.buaa.edu.cn/datasets.html) separates real and simulated data and documents masks, boxes, and centroids. [DeepPro's README](https://github.com/TinaLRJ/DeepPro) explicitly describes NUDT-MIRSDT as synthesized. [Anti-UAV410's official repository](https://github.com/HwangBo94/Anti-UAV410) supplies manually annotated thermal videos and warns that frame attributes in `IR_label.json` are incomplete.

## Official online entry points

- **RGBT-Tiny:** [official repository](https://github.com/XinyiYing/RGBT-Tiny) requires a [Google form](https://forms.gle/EeRooNEYzXXporQt9) or [Microsoft form](https://forms.cloud.microsoft/r/dQEc6dU0A2) for dataset access. Public [GIF example](https://raw.githubusercontent.com/XinyiYing/RGBT-Tiny/main/pics/DJI_0101_2_01_gif.gif) verified HTTP 200, 4,022,870 bytes, GIF89a header, reading only 8 bytes. It has no raw-frame annotations and `DJI_0101_2` is in DeepPro's **training** manifest. None of the six GIF scene names is in the 15-sequence [DeepPro test manifest](https://raw.githubusercontent.com/TinaLRJ/DeepPro/main/datasets/RGB-T/test.txt). Do not use these GIFs for quantitative reproduction.
- **IRDST:** [official sequence link](https://pan.baidu.com/s/1icmYAeRfPdL5n0YcUtuNhg?pwd=kk6h). The [RDIAN example image](https://raw.githubusercontent.com/sun11999/RDIAN/main/datasets/example/images/1%281%29.bmp) is directly public (HTTP 200, 1,036,854 bytes, BMP header) with a [paired mask](https://raw.githubusercontent.com/sun11999/RDIAN/main/datasets/example/masks/1%281%29.bmp). Only three examples exist; ordering and test identity unverified, too short for T40.
- **SatVideoIRSDT:** [official challenge page](https://videoirstd.github.io/) and DeepPro provide Baidu archives. No directly downloadable small labelled SatVideo clip was identified; real/synthetic composition not verified here.

## What this enables

The acquired VT clip now enables inference on a real 11×11-box target with original track labels. The synthetic NUDT clip enables native loader/evaluation checks. A stop/restart failure remains unverified; event labels and a matching box-based evaluator are required. The initial Anti-UAV validation clip remains a cross-dataset engineering diagnostic.
