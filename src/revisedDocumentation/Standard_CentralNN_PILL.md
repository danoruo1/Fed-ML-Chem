# Standard Centralized Neural Network for PILL Dataset

## Overview
This work implements a centralized neural network for pill classification using image data and a pretrained CNN backbone.

## Model Architecture
The model is based on VGG16 pretrained on ImageNet:
- The feature extractor's last layer is removed
- A custom head is added with pooling and fully connected layers
- The first 23 layers of VGG16 are frozen during training

## Dataset
The dataset contains pill images labeled as either good or bad. All images are resized to 256×256 pixels and normalized using ImageNet statistics. A 90:10 training-validation split is applied.

## Training Approach
Training is centralized and uses the Flower framework. The Adam optimizer with a learning rate of 2e-4 is used along with CrossEntropyLoss. Training spans 25 epochs and includes checkpointing and evaluation via a custom federated strategy.

## Key Parameters
- number_clients: 1
- max_epochs: 25
- batch_size: 32
- lr: 2e-4
- rounds: 1
- frac_fit: 1.0
- frac_eval: 1.0

## Results and Visualization
Confusion matrices, ROC curves, and performance plots are stored in 'results/CL_PILL/'.
