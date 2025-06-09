# Standard Centralized Neural Network for DNA+MRI Multimodal Dataset

## Overview
This study introduces a centralized multimodal neural network for the joint classification of DNA sequences and MRI images.

## Model Architecture
The architecture includes two sub-networks:
- MRINet: A convolutional neural network for MRI image classification
- DNANet: A fully connected network for processing DNA sequences

The outputs from both networks are fused using multihead attention and gating mechanisms. The combined representation is used for dual output heads targeting each modality.

## Dataset
The dataset integrates MRI images labeled across four tumor categories and DNA sequences labeled across seven genetic classes. MRI images are resized to 224×224 pixels, and DNA sequences are processed through standard sequence handling. The data is split into training and validation sets at a 90:10 ratio.

## Training Approach
Training is executed centrally with one client using the Flower framework. The Adam optimizer is used with a learning rate of 1e-3. Separate CrossEntropyLoss functions are applied for each modality. The model is trained over 25 epochs with a custom federated strategy for simulation, checkpointing, and evaluation.

## Key Parameters
- number_clients: 1
- max_epochs: 25
- batch_size: 32
- lr: 1e-3
- rounds: 1
- frac_fit: 1.0
- frac_eval: 0.5
- mri_n_qubits: 4
- dna_n_qubits: 7
- expert_vector: 6
- num_of_expert: 2

## Results and Visualization
Confusion matrices, ROC curves, and accuracy/loss plots are generated for each modality and saved in 'results/CL_DNA+MRI/'.
