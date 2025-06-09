## Standard Centralized Neural Network for PILL Dataset

### Overview  
This notebook trains a centralized neural network on the PILL dataset using one client in the Flower framework — so it's technically centralized, even though federated tools are used.

### Model Architecture  
- The model uses a VGG16 backbone pretrained on ImageNet  
- Feature extractor comes from VGG16, with the last layer removed  
- A custom classifier is stacked on top with:  
  - MaxPool2d  
  - AvgPool2d  
  - Flatten  
  - Final Linear layer for output  
- First 23 layers of the backbone are frozen during training (weights stay fixed)  

### Dataset  
- Binary image classification: labels are either 'bad' or 'good'  
- Images are resized to 256x256  
- ImageNet normalization is applied to the input images  
- Training/validation split is 90% / 10%  

### Training Approach  
- All training is done centrally on one client with the Flower framework  
- Optimizer: Adam with a learning rate of 2e-4  
- Loss function: CrossEntropyLoss  
- Training runs for 25 epochs  
- A custom federated strategy (FedCustom) is used that:  
  - Aggregates weights after training (weighted average)  
  - Saves checkpoints at every round  
  - Evaluates the model on the test set after each round  

### Key Parameters  
- `number_clients`: 1  
- `max_epochs`: 25  
- `batch_size`: 32  
- `lr`: 2e-4  
- `rounds`: 1  
- `frac_fit`: 1.0  
- `frac_eval`: 1.0  

### Results and Visualization  
This notebook outputs:  
- Confusion matrix for predictions  
- ROC curves for performance evaluation  
- Training/validation accuracy and loss curves  
- All results are saved in `results/CL_PILL/`
