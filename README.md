# Image Classification with ResNet-50 Fine-Tuning

This repository provides a Jupyter Notebook for fine-tuning the ResNet-50 model on the CIFAR-100 dataset using the Hugging Face Transformers library. The goal is to fine-tune a pre-trained model to classify images into 100 different categories.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [License](#license)

## Introduction

ResNet-50 is a deep residual network that has shown excellent performance in various image classification tasks. In this project, a pre-trained ResNet-50 model on the CIFAR-100 dataset is fine-tuned in order to improve its accuracy in classifying images into 100 categories.

## Dataset

The [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is used for training and evaluation. It consists of 60,000 color images in 100 classes, with 600 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Requirements

- Python 3.7 or higher
- streamlit
- torch
- numpy
- torchvision
- torchaudio
- transformers
- datasets
- accelerate

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/youzaina001/Fine_tuning_microsoft_resnet_50_using_hugging_face.git
    cd Fine_tuning_microsoft_resnet_50_using_hugging_face
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. Open the `vision.ipynb` notebook and run the cells in sequence to start the training process.

## Training

The training script is set up to:

1. **Load the CIFAR-100 dataset**:
    - Uses the `datasets` library to load the CIFAR-100 dataset.
   
2. **Preprocess the dataset**:
    - Converts the images to PyTorch tensors.

3. **Fine-tune the ResNet-50 model**:
    - The model is initialized with weights pre-trained on ImageNet.
    - The number of output labels is set to 100 to match the CIFAR-100 dataset.

4. **Configure Training Parameters**:
    - Uses the `Trainer` API from Hugging Face Transformers to define training arguments such as learning rate, batch size, and the number of epochs.

5. **Start Training**:
    - The model is trained on a GPU (if available) or a CPU.

## Evaluation

After training, the model is evaluated on the test dataset to calculate accuracy and loss. The results are printed in the notebook.

## Results

- The trained model is capable of classifying images into 100 different categories with improved accuracy over the baseline pre-trained ResNet-50.


