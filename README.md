# seGANmentation: Semantic Segmentation via GAN-Based Image-to-Image Translation
This project leverages state-of-the-art Generative Adversarial Network (GAN) models, originally designed for image-to-image translation, for semantic segmentation tasks. By combining convolutional neural networks (CNNs) with vision transformers, the model generates semantic segmentation labels directly from input images. Initially demonstrated with car images and ground-truth segmentation labels, the model is versatile and can be applied to other datasets. This approach aims to reduce manual labeling efforts and enhance segmentation model performance through augmented data.

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

## Pre-Training the Model

To pretrain the model, execute:
```
python3 scripts/Carvana/pretrain_generator.py
```
When the pretraining is finished, the pre-trained model will be saved in the `outdir/Carvana_resized/` directory

## Training the Model

To train the model, execute:
```
python3 scripts/Carvana/train_translation.py
```
When the pretraining is finished, the trained model will be saved in the `outdir/Carvana_resized/I2L` directory

## Translating Images 

To translate images to segmentation masks and see the results, run:
```
python3 scripts/translate_images.py <PATH_TO_TRAINED_MODEL_in_outdir> --split val 
```
