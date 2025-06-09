# Common Utilities Overview

## Overview
This module provides foundational utilities used throughout the neural network pipelines in this project. It includes file handling, device selection, data utilities, visualization, and parameter manipulation to support centralized, federated, and quantum models.

## File Operations
- `write_yaml`, `read_yaml`: Read and write configuration files in YAML format.
- `supp_ds_store`: Removes unnecessary `.DS_Store` files.
- `create_files_train_test`: Splits data into training and testing subsets.

## Device Selection
- `choice_device`: Selects the optimal compute device (CPU, GPU, MPS).

## Dataset Utilities
- `classes_string`: Retrieves class labels based on dataset type.
- Includes dataset-specific normalization values.

## Visualization
- `save_matrix`, `save_roc`, `save_graphs`, `save_graphs_multimodal`: Functions for saving model evaluation visuals including confusion matrices and ROC curves.

## Parameter Management
- `get_parameters2`, `set_parameters`: Extracts or sets model parameters for PyTorch models.

## Usage
This module is imported across all notebooks for streamlined functionality and consistent operations during training, evaluation, and reporting.
