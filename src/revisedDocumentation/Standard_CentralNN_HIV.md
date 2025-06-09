# Standard Centralized Neural Network for HIV Dataset

## Overview
This study presents a centralized graph-based neural network for classifying HIV inhibition activity using molecular graph data.

## Model Architecture
The model employs a Graph Convolutional Network (GCN) with:
- An initial GCN layer from 9 to an embedding dimension
- Three additional GCN layers with tanh activation
- Global pooling by combining max and mean features
- A final linear layer for binary classification

The input consists of molecular graphs where nodes denote atoms and edges represent chemical bonds.

## Dataset
The dataset contains molecular structures labeled as either active or inactive against HIV. It is sourced from MoleculeNet and processed using PyTorch Geometric. A 90:10 training-validation split is applied.

## Training Approach
The model is trained centrally using the Flower framework with a single client. It is optimized using the Adam algorithm with a learning rate of 1e-3 and a custom RMSELoss function. Training proceeds over 25 epochs, with performance assessed using specialized graph-based evaluation methods.

## Key Parameters
- number_clients: 1
- max_epochs: 25
- embedding_size: 64
- batch_size: 64
- lr: 1e-3
- rounds: 1
- frac_fit: 1.0
- frac_eval: 0.5

## Results and Visualization
The model's output includes confusion matrices, ROC curves, and learning curves, stored in 'results/CL_HIV/'.
