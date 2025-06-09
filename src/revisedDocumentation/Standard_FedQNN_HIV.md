# Standard Federated Quantum Neural Network for HIV Dataset

## Overview
This study integrates quantum computing with graph neural networks for classifying HIV molecular activity. Clients train on molecular graph data using a hybrid GCN and quantum circuit architecture in a federated setting.

## Model Architecture
The architecture includes:
- Classical GCN with four convolutional layers
- Quantum circuit with 2 qubits and 2 layers:
  - AngleEmbedding and entanglement layers
  - PauliZ expectation measurement

## Dataset
Molecular graphs labeled as either active or inactive are distributed across ten clients. The dataset is processed using PyTorch Geometric.

## Training Approach
Training spans 20 federated rounds with 10 local epochs per client. A custom RMSELoss is applied alongside the Adam optimizer. Clients are assigned standard or higher learning rates for experimentation.

## Key Parameters
- n_qubits: 2
- n_layers: 2
- number_clients: 10
- rounds: 20
- max_epochs: 10
- embedding_size = batch_size = 64
- frac_fit: 1.0
- frac_eval: 0.5
- lr: 1e-3

## Results and Visualization
Evaluation results include confusion matrices, ROC curves, and performance plots, saved in 'results/QFL_HIV/'.
