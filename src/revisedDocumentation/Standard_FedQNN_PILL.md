# Standard Federated Quantum Neural Network for PILL Dataset

## Overview
This project introduces a federated quantum neural network for classifying pill images. The model leverages a pretrained VGG16 backbone and integrates quantum circuits for classification.

## Model Architecture
- Classical component based on VGG16 with frozen early layers
- A custom head with pooling and dense layers
- A quantum circuit with 2 qubits and 2 layers using AngleEmbedding and entanglement
- Outputs are derived via PauliZ measurements

## Dataset
Binary-labeled pill images are resized to 224×224 pixels. Data is normalized and distributed among ten clients with a 90:10 train-validation split.

## Training Approach
The model is trained over 20 federated rounds. Each client performs 10 local epochs. The Adam optimizer and CrossEntropyLoss are used, and a custom strategy varies learning rates among clients.

## Key Parameters
- number_clients: 10
- max_epochs: 10
- batch_size: 32
- lr: 1e-3
- rounds: 20
- frac_fit: 1.0
- frac_eval: 0.5
- n_qubits: 2
- n_layers: 2

## Results and Visualization
Visualizations and model outputs are stored in 'results/QFL_PILL/'.
