# Utils Package Initialization

## Overview
This file enables simplified imports from the `utils` module by aggregating all functionality from submodules: `common`, `data_setup`, and `engine`.

## Content
```python
from .common import *
from .data_setup import *
from .engine import *
```

## Purpose
Allows concise and readable imports in notebooks:
```python
from utils import load_datasets, train, choice_device
```

## Structure
- `common.py`: Shared utility functions and visualizations
- `data_setup.py`: Dataset preparation and distribution
- `engine.py`: Model training and evaluation logic

This structure ensures modularity and convenience in calling utility functions throughout the project.
