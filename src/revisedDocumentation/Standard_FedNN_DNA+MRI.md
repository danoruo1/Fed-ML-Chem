# Standard Federated Neural Network for DNA+MRI Multimodal Dataset

## Overview
This project presents a federated learning system for a multimodal dataset incorporating DNA sequences and MRI images. Each client trains locally on a subset containing both modalities and contributes to a centralized model via aggregation.

## Model Architecture
The model is composed of:
- MRINet: A CNN handling MRI data
- DNANet: A fully connected network for DNA sequence classification
- Fusion Module: Multihead attention and gating mechanisms merge outputs into a unified representation with dual classification heads

## Dataset
The dataset includes four MRI image classes and seven DNA sequence classes. MRI images are resized to 224×224 pixels. The dataset is split across ten clients, and each local dataset follows a 90:10 training-validation split.

## Training Approach
The model is trained over 20 federated rounds. Each client performs 10 local epochs per round. Two separate CrossEntropyLoss functions are used for DNA and MRI predictions. A differential learning rate strategy is applied across clients.

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
- expert_vector: 6
- num_of_expert: 2

## Results and Visualization
Each modality's outputs are evaluated separately. Confusion matrices, ROC curves, and training metrics are saved in 'results/FL_DNA+MRI/'.
