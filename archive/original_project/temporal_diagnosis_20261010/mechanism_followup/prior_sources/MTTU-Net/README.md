# Memory-based Temporal Transformer U-Net for Multi-frame Infrared Small Target Detection



# Introduction

We propose a memory-based temporal Transformer U-Net (MTTU-Net) to achieve segmentation-based MISTD. Experiments on IRDST and IDSMT demonstrate the effectiveness of our algorithm.

## Usage

#### 1. Data

* **Our dataset has the following structure:**
  ```
  ├──./datasets/
  │    ├── IRDST_real
  │    │    ├── train
  │    │    │    ├── images
  │    │    │    │    ├── 1
  │    │    │    │    │   ├── 1.bmp
  │    │    │    │    │   ├── 2.bmp
  │    │    │    │    │   ├── ...
  │    │    │    │    ├── 3
  │    │    │    │    ├── ...
  │    │    │    ├── masks
  │    │    │    ├── train_anno.json
  │    │    ├── test
  │    │    │    ├── images
  │    │    │    ├── masks
  │    │    │    ├── test_anno.json
  │    ├── IDSMT
  │    │    ├── train
  │    │    │    ├── images
  │    │    │    │    ├── 2
  │    │    │    │    │   ├── 0.png
  │    │    │    │    │   ├── 1.png
  │    │    │    │    │   ├── ...
  │    │    │    │    ├── 8
  │    │    │    │    ├── ...
  │    │    │    ├── labels
  │    │    │    │    ├── 2.json
  │    │    │    │    ├── 8.json
  │    │    │    │    ├── ...
  │    │    │    ├── masks
  │    │    ├── test
  │    │    │    ├── images
  │    │    │    ├── labels
  │    │    │    ├── masks
  ```


##### 2. Train.
```bash
python train.py
```

#### 3. Test.
```bash
python test.py
```

## Datasets and Trained Models

[[IRDST_real]](https://drive.google.com/drive/folders/1RILdC2bTlr1GiX7R5OY8RE5jaYm1x8w2?usp=sharing)
[[IDSMT]](https://drive.google.com/drive/folders/1V1sYoYr0jMYAThIcvT48NTPlg4c1wiIM?usp=sharing) 
|[[Weights]](https://drive.google.com/drive/folders/1rx44sn3uxi3ULDbUZON2kQeAJpsf2Z8J?usp=sharing)|











