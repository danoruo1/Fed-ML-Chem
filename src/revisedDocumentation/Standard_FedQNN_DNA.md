# Standard Federated Quantum Neural Network for DNA Dataset

## Overview
This study presents a federated quantum neural network developed for classifying DNA sequences. It integrates quantum circuits with classical neural networks using PennyLane and distributes training across four clients through the Flower framework.

## Model Architecture
The hybrid model consists of:
- A classical MLP with five layers:
  - fc1: 384 to 1024 units
  - fc2: 1024 to 512 units
  - fc3: 512 to 256 units
  - fc4: 256 to 128 units
  - fc5: 128 to the output layer
- A quantum component with 7 qubits and 7 layers:
  - AngleEmbedding for classical input encoding
  - BasicEntanglerLayers for qubit entanglement
  - PauliZ measurements for output

## Dataset
The DNA dataset has seven categorical labels. It is divided among four clients, each having a 90:10 training-validation split. Data is processed as textual sequences.

## Training Approach
Federated training proceeds for 40 rounds, with each client performing 10 local epochs per round. The model is trained using the Adam optimizer with CrossEntropyLoss. Clients are assigned varying learning rates via a custom aggregation strategy.

## Key Parameters
- number_clients: 4
- max_epochs: 10
- batch_size: 32
- lr: 1e-3
- rounds: 40
- frac_fit: 1.0
- frac_eval: 0.5
- n_qubits: 7
- n_layers: 7

## Results and Visualization
Confusion matrices, ROC curves, and learning metrics are saved in the 'results/FL/' directory.
