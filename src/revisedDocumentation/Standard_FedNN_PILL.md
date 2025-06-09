# Standard Federated Neural Network for PILL Dataset

## Overview
This project employs federated learning to classify pill images using a convolutional neural network based on VGG16. Training is distributed across clients who independently process their data.

## Model Architecture
- Based on a VGG16 backbone pretrained on ImageNet
- The final layer of the backbone is removed
- A new head with pooling and linear layers is appended
- The first 23 layers are frozen to retain pretrained weights

## Dataset
The pill dataset contains binary classifications. Images are resized to 224×224 pixels and normalized. Data is distributed across ten clients with a 90:10 train-validation split.

## Training Approach
Training runs over 20 federated rounds, with 10 local epochs per client per round. The Adam optimizer is used with CrossEntropyLoss. A custom federated strategy supports variable learning rates across clients.

## Key Parameters
- number_clients: 10
- max_epochs: 10
- batch_size: 32
- lr: 1e-3
- rounds: 20
- frac_fit: 1.0
- frac_eval: 0.5

## Results and Visualization
Each client’s performance is logged. Evaluation metrics and visualizations are saved in 'results/FL_PILL/'.
