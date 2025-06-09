# Standard Federated Neural Network for HIV Dataset

## Overview
This study applies federated learning to classify HIV molecular activity using graph-based representations. Clients independently train models using graph convolutional networks (GCNs) and synchronize updates through a centralized aggregation strategy.

## Model Architecture
The architecture includes:
- An initial GCN layer mapping 9-dimensional features to an embedding
- Three additional GCN layers with tanh activations
- Global pooling by combining max and mean features
- A final linear layer outputs binary classifications

## Dataset
The dataset contains molecular graphs labeled as either active or inactive. Data is distributed across ten clients. Each client processes molecular structures using PyTorch Geometric.

## Training Approach
Training occurs over 20 rounds, each with 10 local epochs per client. A custom RMSELoss function is applied alongside the Adam optimizer. Clients are divided such that half use the base learning rate and the rest a higher rate.

## Key Parameters
- number_clients: 10
- rounds: 20
- max_epochs: 10
- embedding_size = batch_size = 64
- frac_fit: 1.0
- frac_eval: 0.5
- lr: 1e-3

## Results and Visualization
Results include confusion matrices, ROC curves, and learning metrics saved in 'results/FL_HIV/'.
