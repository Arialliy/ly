# Annotation-only development clip

Fixed candidate: `DJI_0028_2`, original track `3800006`, thermal channel `01`, raw frames `00173`–`00332` inclusive (160 frames). Selection was frozen before raw-image inspection.

The original JPEG bytes are under `VT-Tiny-MOT/train2017/DJI_0028_2/01/`. Every image fully decodes as native grayscale mode `L`, 640 × 512. `FRAME_MANIFEST.json` pairs every frame with its SHA-256, archive member, source image and annotation IDs, selected-target bbox, and all source annotation IDs.

`annotations.json` retains all 1,280 source annotations across eight original tracks in the selected frames. No category, size, ignore, or target filtering was applied to this annotation subset.

`CONTACT_SHEET.png` and the five `INSPECTION_FRAME_*.png` files are annotation-guided inspection visualizations at raw frames 173, 213, 253, 293, and 332. Their panels show the raw full frame, a copy with the selected bbox in red, and an unboxed 64 × 64 crop enlarged by nearest-neighbor interpolation. They do not alter the JPEGs.

Image-coordinate annotation events remain candidates. These artifacts do not establish physical stop/restart, camera-independent motion, continuously visible identity, or a detector failure. No prediction or model-output files were read, and no inference or GPU work was performed.

Transfer records: the first attempt failed at the public-page redirect TLS handshake before any response body was returned to the acquisition wrapper; its zero-returned-body record is `NETWORK_TRANSFER_ATTEMPT_1.json`. The successful attempt is in `NETWORK_TRANSFER.json`: 1,228,881 ranged ZIP bytes plus a 2,463-byte public form, totaling 1,231,344 returned response-body bytes, below the 50,000,000-byte budget. These byte counts exclude HTTP/transport headers. Preserved JPEG files total 1,518,675 bytes.

`DOWNLOAD_PROVENANCE.json`, `SELECTED_ZIP_MANIFEST.json`, `INSPECTION_VISUALIZATION.json`, and `VALIDATION.json` provide the corresponding provenance, source ZIP offsets, deterministic rendering settings, and validation results.
