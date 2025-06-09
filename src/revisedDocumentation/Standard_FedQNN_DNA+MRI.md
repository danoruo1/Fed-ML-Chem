# Standard Federated Quantum Neural Network for DNA+MRI Multimodal Dataset

## Overview
This project implements a hybrid federated quantum neural network for joint classification of DNA sequences and MRI images. Each client uses quantum-enhanced submodels for both modalities and contributes updates through a federated strategy.

## Model Architecture
The model includes:
- MRINet: A CNN with quantum integration using 4 qubits and 6 layers
- DNANet: A fully connected network with a quantum layer using 7 qubits and 6 layers
- Fusion Layer: Multihead attention and shared fully connected layers with modality-specific outputs

## Dataset
The dataset includes MRI images across four tumor categories and DNA sequences labeled into seven classes. Each client processes data split in a 90:10 training-validation ratio.

## Training Approach
Training involves 10 clients, 10 local epochs per round, and 20 federated rounds. Two CrossEntropyLoss functions are applied separately. A custom aggregation strategy assigns varied learning rates to clients.

## Key Parameters
- number_clients: 10
- max_epochs: 10
- batch_size: 32
- lr: 1e-3
- rounds: 20
- frac_fit: 1.0
- frac_eval: 0.5
- mri_n_qubits: 4
- dna_n_qubits: 7
- n_layers: 6

## Results and Visualization
Results include modality-specific confusion matrices, ROC curves, and training metrics saved in 'results/FL/'.
