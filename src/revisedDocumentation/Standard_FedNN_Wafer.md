# Standard Federated Neural Network for Wafer Dataset

## Overview
This study explores federated learning for classifying semiconductor wafer images. The training is distributed across multiple clients, each managing a local dataset.

## Model Architecture
The network includes:
- A feature extraction block with two convolutional layers, ReLU activations, and max pooling
- A classifier with a flattening operation, a fully connected hidden layer, and an output layer

## Dataset
Wafer images are resized to 64×64 pixels. The dataset is partitioned across ten clients. Data augmentation is used to enhance model generalization.

## Training Approach
Training proceeds for 20 rounds, with each client training locally for 10 epochs per round. The Adam optimizer and CrossEntropyLoss are used. Clients apply varied learning rates through a custom aggregation strategy.

## Key Parameters
- number_clients: 10
- rounds: 20
- max_epochs: 10
- batch_size: 64
- resize: 64
- frac_fit: 1.0
- frac_eval: 0.5
- lr: 1e-3

## Results and Visualization
Training and evaluation outputs include confusion matrices, ROC curves, and performance plots saved in 'results/FL_Wafer/'.
