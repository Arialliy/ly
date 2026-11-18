# DeepPro-Plus official training recipe audit

Audit date: 2026-10-09. Scope: read-only source/configuration and CPU metadata checks. **No training was launched.** Current user-approved execution order starts with official checkpoint inference reproduction; this audit prepares a possible later training reproduction.

## 1. Source and checkpoint identity

- Repository: `https://github.com/TinaLRJ/DeepPro`.
- Pinned commit: `8fa1a68b94eb22e94ccd0529e6c5ceccdaa7ec28`.
- Source root: `/home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro` (abbreviated `S` below).
- NUDT-MIRSDT Plus experiment: `log/sem_seg/NUDT-MIRSDT__2024-12-28_16-21__SoftLoUloss_DeepPro-Plus_DataL40` (abbreviated `E` below).
- `networks/models/DeepPro-Plus.py` and `E/DeepPro-Plus.py` are byte-identical, SHA256 `fa46a034a30d8bbb0f677271d6368f033fcb389d682478d331551d8bae087668`.
- CPU checkpoint inspection with PyTorch restricted safe globals: `epoch=30` (zero based; training epoch 31), `class_avg_iou=0.8717585863994493`; keys are `epoch`, `class_avg_iou`, `model_state_dict`, `optimizer_state_dict`. **There is no seed or RNG state in this checkpoint.**
- Checkpoint Adam state: learning rate `0.00034299999999999993`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, `amsgrad=False`. This agrees with the epoch-31 learning rate from the published schedule.

## 2. Seed: declared default versus actual entrypoint behavior

`train.py:33–40` defines `seed_everything(seed=46)` and seeds Python `random`, NumPy, Torch CPU, CUDA, and all CUDA devices; it sets cuDNN deterministic true and benchmark false.

However:

1. `train.py:285–287` only parses arguments and calls `main(args)`. It does **not** call `seed_everything`.
2. `train.py:120–122` initializes each worker's NumPy RNG from `worker_id + int(time.time())`.
3. The argument parser has no `--seed` option (`train.py:43–66`). The archived log has no seed value.
4. Checkpoints save model and optimizer state but no RNG state (`train.py:207–218`, `260–274`). Restart cannot restore the original sampling stream.

Consequences:

- It is correct to say **“the official helper declares default seed 46; our local reproducibility policy fixes 46 explicitly.”**
- It is unsupported to say **“the released checkpoint was trained with seed 46.”** Its training seed is unknown from the available evidence.
- A future training run meeting the user's fixed-seed requirement must explicitly call the helper before model/dataloader construction and replace wall-clock worker seeding with deterministic worker seeding derived from the fixed Torch generator. These are reproducibility adaptations and must be logged separately from the unchanged detector/loss/recipe.
- Merely wrapping `train.main(...)` with `train.seed_everything(46)` does not fix the wall-clock worker seed.

## 3. Data split and loader semantics

| Item | Source-supported behavior |
|---|---|
| NUDT training split | `datasets/NUDT-MIRSDT/train.txt`: 8,000 lines, 80 unique sequences, 100 listed frames per sequence. SHA256 `31e71368775adbc6c893bcb29ae1f39df6ffc83a693f42f84a4322bcf5235a8c`. |
| NUDT test split | `datasets/NUDT-MIRSDT/test.txt`: 2,000 lines, 20 unique sequences, 100 listed frames per sequence. SHA256 `eb039aa35aaabab2eae559b9d627e53d3fa114eb0410f343498f154c81a9d5d0`. |
| Frame selection | Both loaders extract sequence names from the split file, then enumerate and sort **all directory files**, rather than enforcing individual listed frame names. Dataset directories must therefore contain the expected complete frames and no extras. |
| Training preprocessing | NUDT images and masks are resized to `256 × 256`; the source does not explicitly set PIL resampling. Mask values are divided by 255, then any positive value becomes 1. Images are normalized by mean `105.4025`, std `26.6452`. |
| Temporal candidate list | For each sequence, `i=4,...,N-1` when seqlen=40, taking frames `[max(0,i-40), i)`. Candidate sampling is with replacement and probability proportional to candidate length. The last directory frame is absent from the final candidate because the upper index is excluded. |
| Short windows | Windows with fewer than 40 frames are padded to 40 with zeros **after normalization**. Whether padding is before or after depends on dataset index parity. |
| Spatial crop | One common `128 × 128` crop over the sequence. With probability 0.75 when a target exists in the selected middle label, choose a target-centered crop with jitter; otherwise choose a random crop. Target/jitter selections use Python and Torch RNGs. |
| Other augmentation | `transform=None`; the transform invocation is commented out. No active flips, rotations, or extra temporal augmentation were found in this loader. |
| Effective epoch size | `int(number_of_candidate_windows × sample_rate)`. For 80 sequences of exactly 100 frames: `80 × 96 = 7,680` candidates, `int(7,680 × 0.1)=768` sampled examples/epoch. This matches the archived log. |

Sources: `data_utils/TrainDataLoader.py:21–23,37–60,64–107,124–169`; `data_utils/TestDataLoader.py:112–139`; archived training log `E/logs/DeepPro-Plus.txt:3–4`.

The 80/20 split has no training validation split in this entrypoint. NUDT training evaluates the **test split every epoch and selects the best checkpoint on it**. Reproducing this historical protocol must be distinguished from future development on a separate validation split.

RGB-T source lists contain 45 training and 15 test sequence names; their hashes are respectively `273735b9da40ac2c4a18828f1d3e848851a33339d23325e3f4564b0575b61b6d` and `102505309d793edc918fe3004a2d0de382634d83bcc1a9d6b7c0c14eff808bc4`. The bundled RGB-T Plus training log is an empty file, so it does not establish a complete historical RGB-T recipe.

## 4. Optimization and checkpoint selection

| Item | Current source / archived evidence |
|---|---|
| Model | `DeepPro-Plus`, one output channel, 40 input and output frames. |
| Batch / workers | Batch 4, shuffle true, 4 workers, pin memory true, drop last true. |
| Duration | 32 epochs; with the NUDT count above, 192 optimizer updates/epoch and 6,144 updates total for a fresh uninterrupted run. |
| Precision | Images/targets converted to float32. No autocast, GradScaler, gradient accumulation, clipping, or EMA in the entrypoint. |
| Optimizer | Adam, initial LR 0.001, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`. |
| LR schedule | `max(0.001 × 0.7**(zero_based_epoch // 10), 1e-5)`: epochs 1–10 `0.001`; 11–20 `0.0007`; 21–30 `0.00049`; 31–32 `0.000343`. |
| Loss | `SoftLoUloss`: sigmoid logits, sum intersection/union over time and both spatial axes for each sample, then `1 - mean(sample_IoU)`. `smooth=0.00`. |
| Scheduled checkpoints | Every 5 epochs plus final epoch. |
| Best checkpoint | Highest aggregate pixel intersection/union on the test windows, with ties replacing the prior best. This is not a Pd/Fa or AUC selection criterion. |
| Training-time evaluation | Original test loader, batch 1, eval mode; overlapping windows are each counted directly in IoU. It does not perform the unique-frame max fusion used in standalone `test.py`. |
| Resume | Automatically tries `best_model.pth` inside the output experiment, restores optimizer and starts at checkpoint epoch+1. `best_iou` resets to zero, RNG state is absent, and a broad exception silently falls back to fresh training. Use a new isolated output directory for a fresh reproduction. |

Sources: `train.py:43–64,115–148,150–171,181–218,221–276`; `networks/models/DeepPro-Plus.py:84–101`.

### Material archived/current recipe difference

`E/logs/DeepPro-Plus.txt:2` records `threshold_eval=0.3`; **current** `train.py:64` defaults to `0.5`. Since best-model selection uses this threshold, leaving the current default can choose a different checkpoint even with an otherwise identical training trajectory.

The archived namespace additionally contains obsolete `threshold=0.01`, absent from today's parser, and omits `sample_rate`. Its timestamp and save path refer to a November 2024 run with an older internal model name; the enclosing released folder is dated December 2024. However, archived epoch 31 IoU `0.871759` (`E/logs/DeepPro-Plus.txt:294–304`) agrees with checkpoint metadata `0.8717585863994493`. This supports association of the log with the checkpoint, while it does not establish a complete source/environment freeze.

## 5. Concrete source-supported command, for a later stage

The command below reconstructs all presently supported NUDT hyperparameters and the archived **0.3 selection threshold**. It is a specification, **not a command executed in this audit**, and requires complete training data, dependency fixes and the separately recorded deterministic adapter described above.

```bash
cd /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_deeppro
python3 train.py \
  --model DeepPro-Plus \
  --dataset NUDT-MIRSDT \
  --datapath /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_reproduction/data/NUDT-MIRSDT \
  --savepath /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/official_reproduction/training_outputs \
  --log_dir nudt_plus_archived_recipe_seed46_adapter \
  --gpu 0 --gpu_num 1 \
  --batch_size 4 --epoch 32 --seqlen 40 --patch_size 128 \
  --optimizer Adam --learning_rate 0.001 --decay_rate 0.0001 \
  --step_size 10 --lr_decay 0.7 --threshold_eval 0.3
```

The run label expresses the intended future adapted run; the unmodified command itself does **not** enforce seed46. `--sample_rate` is deliberately omitted: its source default is float `0.1`, but its parser type is incorrectly `int`, so `--sample_rate 0.1` is rejected. Do not silently pass 0 or round this value.

The original final `os.system('python test.py ...')` (`train.py:289–290`) does not forward `--savepath` as the test script's `--logpath`. A nondefault save directory therefore needs an explicit separate evaluation command; it must not be assumed that the training script's automatic test used the intended weights. Future isolated adapter should preserve and report this deviation.

## 6. Environment and executable compatibility

Official README lists only `Python 3`, `torch`, `tqdm` (`README.md:16–19`). No requirements, environment, setup, or pyproject manifest exists in the pinned repository. Therefore the author's historical library versions are not established.

CPU-only metadata check of unchanged image `ppg/ultralytics:8.3.203`:

| Distribution | Installed metadata |
|---|---|
| torch | 2.7.0+cu128 |
| numpy | 2.2.5 |
| tqdm | 4.67.1 |
| Pillow | 11.0.0 |
| opencv-python-headless | 4.12.0.88 |
| scipy | 1.16.2 |
| matplotlib | 3.11.2 |
| scikit-image | missing |
| scikit-learn | missing |
| thop | missing |
| pyparsing | missing |

Original test dependency evidence: `test.py:13–25` imports NumPy, Pillow, OpenCV, SciPy, sklearn and thop; `ShootingRules.py:6` requires skimage; `networks/layers/TPro.py:47–51` imports matplotlib/pyplot at module import time, even when no figure is generated. The existing Torch/CUDA image can be reused as the base for an isolated dependency layer, but is not sufficient as-is for the complete original test entrypoint. No packages were installed or changed in this audit.

Additional concrete compatibility issue for eventual training: `TrainDataLoader.py:65` calls `np.random.choice(self.samplelist, p=self.sample_p)` on a ragged nested list of windows. A CPU probe of this exact kind of argument in NumPy 2.2.5 raises `ValueError: ... inhomogeneous shape ...`. A compatibility adapter can sample an integer candidate index with the same probabilities, then index the Python list; record this fix, preserve the sampling distribution, and verify the loader before training. It does not affect the inference-only loader.

Other reproducibility boundaries:

- Current PyTorch checkpoint loading requires a narrow NumPy scalar/dtype safe-global compatibility allowlist for this checkpoint. Loading was read-only and strict; arbitrary unsafe deserialization was not used.
- Multi-GPU source behavior is DataParallel over hardcoded visible `0,1,2,3` whenever `gpu_num > 1` (`train.py:75–78,131–134`). It is not equivalent to a one-GPU recipe because local BatchNorm batch composition changes. Available four GPUs can instead serve independent inference shards while preserving the checkpoint recipe.
- PIL resize interpolation is unspecified in source; pin the installed Pillow version and image modes. Do not replace mask interpolation silently under a claim of exact historical reproduction.
- `torch.autograd.set_detect_anomaly = True` is an assignment, not a call (`train.py:183`); it does not enable anomaly detection as written. It is not part of the detector computation.

## 7. Decision for the present stage

**Proceed with full official checkpoint inference reproduction first.** This does not require training data, the training sampler, or resolving every historical training ambiguity. Keep the supplied checkpoint fixed and compare the complete original test protocol against released results. Label it checkpoint inference reproduction, not from-scratch training reproduction. The real-video motion-change comparison comes after this gate; no mechanism is selected by this audit.
