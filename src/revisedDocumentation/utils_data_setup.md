# Data Setup Utilities Overview

## Overview
This module contains data handling and preprocessing functionality tailored to various datasets including images, sequences, molecular graphs, and multimodal formats. It facilitates consistent loading, normalization, augmentation, and client distribution.

## Dataset Normalization
- Provides predefined normalization constants for multiple datasets.

## Custom Dataset Classes
- `MultimodalDataset`: Manages multimodal input combinations such as DNA and MRI for federated learning.

## Preprocessing Functions
- `read_and_prepare_data`: Converts DNA sequences into numerical encodings.
- `preprocess_graph`: Prepares graph-based inputs like molecular structures.
- `preprocess_and_split_data`: Augments and partitions image/audio data.

## Client-Specific Splitting
- `split_data_client`: Distributes data to clients in a balanced manner.
- `load_datasets`: General-purpose loader for centralized or federated training, generating appropriate DataLoaders.

## Usage
Essential for preparing datasets before model training, this module ensures consistency in preprocessing across all experiments and learning paradigms.
