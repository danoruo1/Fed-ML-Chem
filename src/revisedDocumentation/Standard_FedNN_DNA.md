# Standard Federated Neural Network for DNA Dataset

## Overview
This study implements a federated learning framework for the DNA dataset using the Flower platform. The training is distributed across multiple clients, each responsible for learning from a distinct subset of the dataset. A global model is periodically aggregated from the local updates.

## Model Architecture
The architecture features a multilayer perceptron with five layers and ReLU activations:
- fc1: input_sp to 64 units
- fc2: 64 to 32 units
- fc3: 32 to 16 units
- fc4: 16 to 8 units
- fc5: 8 to the number of output classes

The input size is determined based on the characteristics of the dataset.

## Dataset
The DNA dataset includes seven categorical labels. Data is processed as sequences and is distributed across ten clients. Each client’s data is further divided into training and validation sets using a 90:10 ratio.

## Training Approach
The model is trained using a federated approach over 20 rounds, with each client performing 10 local epochs per round. The Adam optimizer and CrossEntropyLoss are used. A custom strategy allows differential learning rates, with half of the clients using a higher rate.

## Key Parameters
- number_clients: 10
- max_epochs: 10
- batch_size: 16
- lr: 1e-3
- rounds: 20
- frac_fit: 1.0
- frac_eval: 0.5

## Results and Visualization
Generated artifacts include confusion matrices, ROC curves, and performance metrics for each client, stored in 'results/FL_DNA/'.
