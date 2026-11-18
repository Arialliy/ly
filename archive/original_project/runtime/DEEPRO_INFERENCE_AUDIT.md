# DeepPro checkpoint inference audit

Audited 2026-10-09 against official commit `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`.
The files in `deeppro_audit_source` are an isolated runtime subset. No training,
model instantiation, forward pass, GPU invocation, or dependency installation was
performed in this audit. CPU import of the model module and restricted checkpoint
metadata loads succeeded inside the existing `ppg/ultralytics:8.3.203` image.

## Checkpoints and model

- RGB-T: `log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth`, epoch metadata 21; SHA256 `4fd4aa2736d9a33f2bf1459099068d8b4a7e202ff80aecdb4a3277fd13c7abcc`.
- NUDT: `log/sem_seg/NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth`, epoch metadata 30; SHA256 `e2ad2d5c9b39c3d428f57d8d50edefd6a09f8b45dff53fb9d31134272e5cc83e`.
- Each is 937131 bytes, with `epoch`, `class_avg_iou`, `model_state_dict`, `optimizer_state_dict`; each state has 100 tensors and eight temporal projection weights of shape `[40,40]`. Stored IoU metadata is historical checkpoint metadata, not a result of this audit.
- Official evaluation imports the checkpoint experiment's `DeepPro-Plus.py`, instantiates `detector(1,40,40)`, loads `checkpoint['model_state_dict']` strictly, and calls `.cuda().eval()`.
- Original current `DeepPro-Plus_forMovingScenes.py` has `super(generator,self)` although its class is named `detector`; it is a different architecture and is not the entry point for the audited RGB-T or NUDT checkpoint.

## Inputs, temporal windows, outputs

- Raw images become float32. If an image has three channels, the loader selects channel zero; it does not calculate RGB luminance.
- RGB-T normalization: `(raw - 85.0799) / 47.4845`; NUDT: `(raw - 105.4025) / 26.6452`. No division by 255, min/max scaling, clipping, or test resizing for these two datasets.
- Input batch shape `[1,1,40,H,W]`. Output is `(features, logits)`; logits shape `[1,40,H,W]`. Apply `torch.sigmoid(logits)`. Every input frame gets a prediction. The normal Plus model preserves spatial resolution.
- The test loader uses `cat_len=int(0.1*40)=4`. For sequence length N, number of windows is `ceil((N-4)/36)`; for index i, exclusive end is `min(N,(i+1)*36+4)`, start is `max(0,end-40)`. Example N=100: `[0,40), [36,76), [60,100)`.
- The final window anchors at N and may overlap much more than four frames. There is no external temporal padding: commented padding code is inactive. N<40 does not fit the learned Linear input size; the provided runner rejects it explicitly.
- Local convolution layers use symmetric zero padding. `TPro` has unrestricted learned full-window 40-by-40 temporal projections, so inference is acausal and stateless across windows. Overlap probabilities are fused by pointwise maximum, not averaging.
- Official visual saving is `uint8(probability * 255)` PNG; RGB-T numbering begins at zero, NUDT at one. This loses floating point precision; the runner instead saves raw float32 probabilities plus window indices, coverage, and provenance in NPZ. It computes no detection metrics or labels.
- Official mask evaluation defaults to probability `>0.5`; bounding boxes are not native model outputs. Cross-dataset video diagnostics must not be labeled an official benchmark reproduction.

## Seeds and checkpoint loading

`train.py` defines `seed_everything(seed=46)` but never calls it. Its training
DataLoader seeds each worker's NumPy RNG from `worker_id + int(time.time())`.
`test.py` neither sets a seed nor configures deterministic backends. Therefore
46 cannot be reported as an effective official training or test seed.

Direct `torch.load(...,weights_only=True)` initially rejects NumPy metadata.
Static inspection found only `numpy.core.multiarray.scalar` and `numpy.dtype`
outside the default safe globals. Restricted CPU loading succeeded with those
two globals plus the NumPy float64 dtype class allowed; `infer_checkpoint.py`
uses the same narrow context. It never falls back to unrestricted pickle.

## Minimal call

Run with an existing PyTorch environment from any directory:

```bash
python /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/infer_checkpoint.py \
  --source /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/deeppro_audit_source \
  --checkpoint /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/runtime/deeppro_audit_source/log/sem_seg/RGB-T__2025-01-14_15-27__SoftLoUloss_DeepPro-Plus_DataL40/checkpoints/best_model.pth \
  --model-type DeepPro-Plus --normalization RGB-T \
  --input-npy RAW_GRAYSCALE_T_H_W.npy --output-npz predictions.npz --device cuda:0
```

For NUDT change checkpoint directory and normalization together. Input must
contain original raw grayscale intensity values, retain temporal ordering, and
have at least 40 frames. Downsampling, color conversion, selection of frame
ranges, and interpretation of external annotations are upstream diagnostic
choices and must be recorded separately.

## Source links

- [Official test.py](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/test.py)
- [Official test data loader](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/data_utils/TestDataLoader.py)
- [Official temporal projector](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/networks/layers/TPro.py)
- [Official training entry point](https://github.com/TinaLRJ/DeepPro/blob/8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28/train.py)
