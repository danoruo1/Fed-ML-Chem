# Standard Centralized Neural Network for Wafer Dataset

## Overview
This report describes a centralized convolutional neural network developed for semiconductor wafer image classification.

## Model Architecture
The network has two components:
- Features: Two convolutional layers with ReLU and max pooling
- Classifier: A flatten layer, a dense hidden layer, and an output layer

The model accepts RGB images and outputs a classification based on manufacturing characteristics.

## Dataset
The dataset contains wafer images resized to 64×64 pixels. Standard data augmentation techniques are applied, and the dataset is split into training and validation sets using a single centralized client.

## Training Approach
Using the Flower framework, the model is trained with the Adam optimizer at a learning rate of 1e-3. CrossEntropyLoss is applied. Training is carried out for 25 epochs over a single round of centralized training.

## Key Parameters
- number_clients: 1
- rounds: 1
- max_epochs: 25
- batch_size: 64
- resize: 64
- lr: 1e-3

## Results and Visualization
The training produces confusion matrices, ROC curves, and accuracy/loss plots, all saved in 'results/CL_Wafer/'.
