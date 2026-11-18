# MIST (TIP 2026)

Official implementation of the paper **"MIST: A Benchmark and Baseline for Multi-frame Infrared Small Target Detection in Complex Motion"** [![IEEE paper](https://img.shields.io/badge/IEEE-Paper-blue)](https://ieeexplore.ieee.org/document/11511399)

## News

- 2026-04-27: Congrats! Our paper has been accepted by **IEEE TIP**.
- 2026-03-08: Our dataset, code, and trained model weights are released.

## Overview

### 1. Highlights

- We release **MIST**, a challenging multi-frame infrared small target detection (MISTD) dataset built on a synthetic data engine. The dataset covers scenarios where targets have low signal-to-clutter ratios (SCR) and complex motion, featuring diverse target/background characteristics and precise annotations.
- Based on the Information Bottleneck (IB) theory, we develop a simple yet effective baseline called **MISTNet** for MISTD. Our method is robust in handling irregular and fast motion while resisting target-irrelevant interference.

### 2. MIST

MIST is a large-scale dataset for airborne infrared detection scenarios. The dataset is built on a **synthetic data engine** that models variations in pose, size, and intensity of moving targets while seamlessly blending them into real backgrounds for physical, geometric, and visual realism. Targets in MIST exhibit **low SCR** and **complex motion**, making it a promising yet challenging benchmark for developing algorithms focused on motion analysis.

![MIST](assets/MIST.png)

### 3. MISTNet

To tackle the challenges of MIST, we develop MISTNet, a robust baseline based on the IB theory. To handle irregular and fast motion, we propose a **Shifted Neighborhood Compensation Block (SNCB)** to efficiently model multi-scale correspondences for implicit motion compensation. To distill compact representations free from irrelevant cues, we design a **Progressive Distillation Decoder (PDD)** to hierarchically filter out redundancy while preserving target-relevant information.

![MISTNet](assets/MISTNet.png)

## Getting Started

### 1. Prerequisites

- Python 3.8
- [PyTorch](https://pytorch.org/) 1.10.1
- [NATTEN](https://natten.org/) 0.14.6

### 2. Installation

**Step 1.** Create a conda environment and activate it.

```bash
conda create -n Deep-MIST python=3.8
conda activate Deep-MIST
```

**Step 2.** Install PyTorch with CUDA support.

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**Step 3.** Install NATTEN using our downloaded pre-compiled wheel [[Google Drive]](https://drive.google.com/file/d/1kjhM7p4P2RZ1Wdus5E6sP4cIFzipBiGk/view?usp=sharing) (or you can refer to the [official install instructions](https://github.com/SHI-Labs/NATTEN/releases/tag/v0.14.6)).

```bash
pip install natten-0.14.6+torch1101cu113-cp38-cp38-linux_x86_64.whl
```

**Step 4.** Install other dependencies.

```bash
pip install -r requirements.txt
```

## Dataset Preparation

We conduct experiments on the proposed MIST dataset and the commonly used [NUDT-MIRSDT](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM) dataset. You can download the datasets from the following links and place them under the `data/` directory:

- MIST [[Google Drive]](https://drive.google.com/file/d/1jJabpOzmuLgxa52V1dtBuXv5-RpKNiSn/view?usp=sharing)
- NUDT-MIRSDT [[Baidu Netdisk]](https://pan.baidu.com/s/1pSN350eurMafLiHBQBnrPA?pwd=5whn)

The expected directory structure is as follows:

```bash
data/
├── MIST/
│   ├── bbox/
│   │   ├── 1/
│   │   │   ├── 0000.txt
│   │   │   └── ...
│   │   └── ...
│   ├── image/
│   │   ├── 1/
│   │   │   ├── 0000.png
│   │   │   └── ...
│   │   └── ...
│   ├── mask/
│   │   ├── 1/
│   │   │   ├── 0000.png
│   │   │   └── ...
│   │   └── ...
│   ├── point/
│   │   ├── 1/
│   │   │   ├── 0000.txt
│   │   │   └── ...
│   │   └── ...
│   ├── train.txt
│   ├── val_all.txt
│   └── val_hard.txt
│
└── NUDT-MIRSDT/
    ├── Sequence1/
    │   ├── images/
    │   │   ├── 00001.png
    │   │   └── ...
    │   ├── masks/
    │   │   ├── 00001.png
    │   │   └── ...
    │   ├── masks_centroid/
    │   │   ├── 00001.png
    │   │   └── ...
    │   └── Mix/
    │       ├── 00001.mat
    │       └── ...
    ├── ...
    ├── test.txt
    └── train.txt
```

**Note:** For the MIST dataset, we split it into a training set of 78 sequences and a test set of 42 sequences using a stratified sampling strategy. To rigorously assess model robustness against low SCR and complex motion, we also curate a more challenging subset (11 sequences) from the test set, named MIST-Hard. Targets in this subset are characterized by extremely low SCR (≤1), highly irregular trajectories, and ultra-high speeds (>7 pixels/frame).

## Basic Usage

### 1. Training

Please refer to the configuration files in `configs/` for detailed training settings. By default, the model is trained on two 24GB GPUs with a batch size of 16.

```bash
# Train on MIST
python train.py --config ./configs/train_MISTNet_MIST.yaml

# Train on NUDT-MIRSDT
python train.py --config ./configs/train_MISTNet_NUDTMIRSDT.yaml
```

### 2. Test

Our trained model weights are available at [[Google Drive]](https://drive.google.com/file/d/15Cu1G8Zga-tWszSByex5bb3x2VRfH2yW/view?usp=sharing). You can download the weights for quick evaluation.

```bash
# Test on MIST
python test.py --config ./configs/test_MISTNet_MIST.yaml

# Test on NUDT-MIRSDT
python test.py --config ./configs/test_MISTNet_NUDTMIRSDT.yaml
```

## Benchmark Results

We conduct a comprehensive comparison of MISTNet against 30 state-of-the-art methods on the MIST and NUDT-MIRSDT datasets. The results are summarized as follows:

![Benchmark Results](assets/benchmark_results.png)

## Acknowledgement
This repo is partially built upon [DTUM](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM), [RFR](https://github.com/XinyiYing/RFR), [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection), and [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD). We sincerely thank the authors for their excellent work.

## Contact
Feel free to open an issue or contact us at ruigao@shu.edu.cn for any questions.

## Citation
If you find this work useful for your research, please star this repo and cite our paper:

```bibtex
@article{gao2026mist,
  title={MIST: A Benchmark and Baseline for Multi-frame Infrared Small Target Detection in Complex Motion},
  author={Gao, Rui and Zhang, Meihong and Li, Gongyang and Li, Guanyi and Zhao, Kai and Zhang, Xianchao and Zeng, Dan},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  publisher={IEEE}
}
```