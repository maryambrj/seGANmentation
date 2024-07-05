# seGANmentation: Semantic Segmentation via GAN-Based Image-to-Image Translation
This project leverages state-of-the-art Generative Adversarial Network (GAN) models for semantic segmentation tasks. Initially demonstrated with car images, the model is versatile and can be applied to other datasets. When used within a pipeline tailored for synthetic image generation in segmentation tasks, this approach aims to reduce manual labeling efforts and enhance segmentation model performance through augmented data.

seGANmentation adapts [uvcgan2](https://github.com/LS4GAN/uvcgan2) for direct image-to-label translation.

This repository houses the code and documentation necessary for further development and experimentation with image segmentation models.

## Cloning the Repository

```
git clone https://github.com/maryambrj/seGANmentation.git
cd seGANmentation/uvcgan2
```

## Environment Setup

The project environment can be constructed with `conda`:
```
conda env create -f contrib/conda_env.yaml
```
The created conda environment can be activated with:
```
conda activate uvcgan2
```
To install the uvcgan2 package, simply run the following command:
```
python3 setup.py develop --user
```

## Dataset Preperation:

### Download the Dataset
* The Carvana dataset is available on Kaggle at [Carvana Image Masking Challenge](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data).
* Download the `train.zip` file from the competition data section.

### Organize the Data:
* Create a folder named `data` in the `uvcgan2` project directory.
* Within the `data` folder, create another folder named `Carvana_resized`.
* Extract the contents of `train.zip` into the `Carvana_resized` folder. This will create a train folder containing the training images.

### Directory Structure:

After extraction, ensure that your directory structure looks like this:
```
uvcgan2/
│
├── data/
│   └── Carvana_resized/
│       └── train/
│           ├── 00087a6bd4dc_01.jpg
│           ├── ...
```
## Pre-Training the Generator

To pretrain the generator, execute:
```
python3 scripts/Carvana/pretrain_generator.py
```
When the pretraining is finished, the pre-trained model will be saved in the `outdir/Carvana_resized/` directory.

## Training for Image-to-Label Translation

To train the I2L translation, execute:
```
python3 scripts/Carvana/train_translation.py
```
When the training is finished, the trained model will be saved in the `outdir/Carvana_resized/I2L` directory.

## Translating Images 

After the training, to translate images to segmentation masks and see the results using model checkpoints saved in `outdir`, run:
```
python3 scripts/translate_images.py <PATH_TO_TRAINED_MODEL_in_outdir> --split val 
```
## Evaluating the Segmentation

After translating the images, the script below evaluates the segmentation performance of a model by comparing its predicted images with ground truth images saved in `outdir` using three metrics: Dice Coefficient, Intersection over Union (IoU), and accuracy:
```
python3 scripts/evaluate_segmentation.py
```
