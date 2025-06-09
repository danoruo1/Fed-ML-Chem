# Standard Centralized Neural Network for DNA Dataset

## Overview
This study implements a centralized neural network designed for the DNA dataset. 

## Model Architecture
The model architecture consists of a multilayer perceptron with five fully connected layers and ReLU activations:
- fc1: input_sp to 64 units
- fc2: 64 to 32 units
- fc3: 32 to 16 units
- fc4: 16 to 8 units
- fc5: 8 to the number of classes

The input dimensionality is dynamically determined based on the dataset.

## Dataset
The dataset comprises DNA sequences categorized into seven classes ranging from 0 to 6. Data is processed using standard sequence handling methods and is split into training and validation sets with a 90:10 ratio.

## Training Approach
Training is conducted centrally using a single client through the Flower framework. The model is optimized using the Adam optimizer with a learning rate of 1e-3 and CrossEntropyLoss. It is trained over 25 epochs. A custom federated strategy is employed to simulate aggregation, save model checkpoints, and perform evaluation.

## Key Parameters
- number_clients: 1
- max_epochs: 25
- batch_size: 16
- lr: 1e-3
- rounds: 1
- frac_fit: 1.0
- frac_eval: 1.0

## Results and Visualization
Output includes confusion matrices, ROC curves, and training-validation metrics, all stored in the 'results/CL_DNA/' directory.
