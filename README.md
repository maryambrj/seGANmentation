# seGANmentation: Semantic Segmentation via GAN-Based Image-to-Image Translation

This project explores the use of state-of-the-art Generative Adversarial Networks (GANs) for semantic segmentation tasks. By formulating segmentation as an image-to-image translation problem (Image-to-Label), this approach aims to leverage generative priors and adversarial training to improve segmentation performance, especially in scenarios where traditional supervised methods might struggle or require extensive manual labeling.

The core implementation adapts [uvcgan2](https://github.com/LS4GAN/uvcgan2) for direct image-to-label translation, demonstrated initially on the Carvana car masking dataset. Additionally, standard segmentation baselines (U-Net and DeepLabV3+) are provided for direct comparison.

## Quick Start & Environment Setup

Clone the repository and set up the conda environment:

```bash
git clone https://github.com/maryambrj/seGANmentation.git
cd seGANmentation/uvcgan2
conda env create -f contrib/conda_env.yaml
conda activate uvcgan2
python3 setup.py develop --user
```

## Dataset Preparation

1. Download the Carvana dataset (`train.zip`) from the [Kaggle Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).
2. Extract and organize the data into the following structure:

```text
uvcgan2/
└── data/
    └── Carvana_resized/
        ├── train/
        │   ├── images/  (RGB photos)
        │   └── masks/   (Binary masks)
        └── val/
            ├── images/
            └── masks/
```

## 1. GAN-Based Segmentation (seGANmentation)

We use UVCGAN2 configured for Image-to-Label translation.

### Pre-training & Training

First, pre-train the generator on the image domain, then train the full translation model:

```bash
python3 scripts/Carvana/pretrain_generator.py
python3 scripts/Carvana/train_translation.py
```
*Models are saved to `outdir/Carvana_resized/` and `outdir/Carvana_resized/I2L`.*

### Inference & Evaluation

Translate validation images to segmentation masks and evaluate their performance (Accuracy, Dice, IoU):

```bash
python3 scripts/translate_images.py <PATH_TO_TRAINED_MODEL_in_outdir> --split val 
python3 scripts/evaluate_segmentation.py
```

## 2. Standard Baselines (U-Net & DeepLabV3+)

To provide a robust comparison against the GAN approach, we include standard supervised segmentation baselines. These scripts use the same dataset structure, evaluation metrics, and general hyperparameters.

### DeepLabV3+ (ResNet-34)

Train a DeepLabV3+ model using `segmentation_models_pytorch`:

```bash
# Requires: pip install segmentation_models_pytorch
python3 scripts/train_deeplabv3plus.py \
    --data-root data/Carvana_resized \
    --outdir outdir/deeplabv3plus \
    --batch-size 16 \
    --epochs 10 \
    --lr 1e-5 \
    --save-preds
```

*The scripts output standard logs (`training_log.csv`), model checkpoints, and optionally save predictions to evaluate alongside the GAN results.*
