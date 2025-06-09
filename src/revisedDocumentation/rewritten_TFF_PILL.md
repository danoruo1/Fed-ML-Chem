# TensorFlow Federated Neural Network for PILL Dataset

## Overview
This work implements a federated learning approach for pill classification using TensorFlow Federated. Unlike the Flower-based implementations in the rest of the project, this framework uses TFF’s native Federated Averaging algorithm to perform training across decentralized clients.

## Model Architecture
The model leverages transfer learning using VGG16:
- A VGG16 backbone pretrained on ImageNet
- The first 23 layers are frozen to retain learned weights
- An AveragePooling2D layer reduces spatial dimensions
- A Flatten layer prepares the feature map for classification
- A final Dense layer with softmax activation performs binary classification

## Dataset
The PILL dataset is used for binary classification between faulty and normal pills. Images are resized to 256×256 pixels and normalized. The dataset is distributed evenly across ten clients using TensorFlow's `image_dataset_from_directory` function, and all images are normalized by scaling pixel values to the [0, 1] range.

## Training Approach
Federated learning is performed using TFF's Federated Averaging:
- Ten clients participate in each round
- Each client trains locally for 10 epochs per round
- A total of 20 federated rounds are executed
- The Adam optimizer is used with a learning rate of 0.003
- CategoricalCrossentropy is used as the loss function
- Model evaluation is conducted after each round using data from all clients

## Key Parameters
- num_clients: 10
- image_shape: (256, 256, 3)
- num_categories: 2
- num_rounds: 20
- num_epochs: 10
- batch_size: 32
- learning_rate: 0.003

## Results and Visualization
Training metrics are logged after each round, and final accuracy is calculated by averaging results over all rounds. This implementation provides a comprehensive example of TensorFlow Federated in contrast to the PyTorch and Flower-based approaches used elsewhere in the project.

## Key Differences from Flower-based Implementations
- Utilizes TensorFlow Federated (TFF) instead of Flower
- Implements native FedAvg using TFF’s API
- Uses TensorFlow’s `image_dataset_from_directory` for data ingestion
- Employs a slightly different VGG16-based architecture and training procedure
