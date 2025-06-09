# Standard Federated Quantum Neural Network for Wafer Dataset

## Overview
This study employs federated learning with quantum-enhanced models for semiconductor wafer classification. Classical convolutional layers are augmented with a quantum layer and trained across distributed clients.

## Model Architecture
- CNN feature extractor with two convolutional layers and ReLU activations
- A classifier with one hidden layer
- A quantum circuit with 9 qubits and 9 layers using AngleEmbedding and BasicEntanglerLayers
- Outputs via PauliZ measurements

## Dataset
Wafer images resized to 64×64 pixels are distributed across ten clients. Data augmentation is used to enhance robustness.

## Training Approach
The model is trained over 20 rounds of federated learning with 10 local epochs per round. Learning rates vary across clients. Adam optimizer and CrossEntropyLoss are applied.

## Key Parameters
- n_qubits: 9
- n_layers: 9
- number_clients: 10
- rounds: 20
- max_epochs: 10
- batch_size: 64
- resize: 64
- frac_fit: 1.0
- frac_eval: 0.5
- lr: 1e-3

## Results and Visualization
Results include confusion matrices, ROC curves, and performance summaries saved in 'results/QFL_Wafer/'.
