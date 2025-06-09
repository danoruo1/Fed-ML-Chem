## Standard Centralized Neural Network for DNA Dataset

### Overview  
This notebook trains a single, centralized neural network on the DNA dataset.

### Model Architecture  
- A basic multilayer perceptron (MLP) is used  
- The network has 5 fully connected layers with ReLU activations:  
  - fc1: input_sp → 64  
  - fc2: 64 → 32  
  - fc3: 32 → 16  
  - fc4: 16 → 8  
  - fc5: 8 → num_classes (final layer)  
- `input_sp` is automatically set based on the data’s shape  

### Dataset  
- Works with a DNA dataset that has 7 possible class labels ('0' through '6')  
- The data consists of DNA sequences, not images  
- Standard sequence/text processing methods are used  
- Dataset is split 90% training, 10% validation  

### Training Approach  
- Everything runs on one client (centralized) using the Flower framework  
- Optimizer: Adam, learning rate: 1e-3  
- Loss: CrossEntropyLoss  
- Trains for 25 epochs  
- Uses a custom federated strategy (FedCustom) that:  
  - Averages weights with respect to client contributions  
  - Saves the model after every round  
  - Tests the model on a separate test set  

### Key Parameters  
- `number_clients`: 1  
- `max_epochs`: 25  
- `batch_size`: 16  
- `lr`: 1e-3  
- `rounds`: 1  
- `frac_fit`: 1.0  
- `frac_eval`: 1.0  

### Results and Visualization  
This notebook outputs:  
- Confusion matrix for classification results  
- ROC curves to show model performance  
- Accuracy and loss curves for both training and validation  
- All results get saved in `results/CL_DNA/`
