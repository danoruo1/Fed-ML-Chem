## Standard Federated Neural Network for DNA Dataset

### Overview  
This notebook uses federated learning to train on the DNA dataset. Instead of using one central dataset, it splits the data across multiple clients. Each client trains on its own subset, and the server aggregates their models using the Flower framework.

### Model Architecture  
- A simple MLP with 5 fully connected layers and ReLU activations:  
  - fc1: input_sp → 64  
  - fc2: 64 → 32  
  - fc3: 32 → 16  
  - fc4: 16 → 8  
  - fc5: 8 → num_classes (final layer)  
- `input_sp` is dynamically determined based on the dataset  

### Dataset  
- DNA sequence classification with 7 total classes: '0' through '6'  
- Data is sequence-based, not image-based  
- Preprocessing is done using standard text/sequence methods  
- The dataset is divided across 10 clients  
- Each client's dataset is split 90% training, 10% validation  

### Training Approach  
- Fully federated setup with 10 clients  
- Optimizer: Adam with learning rate = 1e-3  
- Loss: CrossEntropyLoss  
- Each client trains for 10 epochs in each round  
- Runs for a total of 20 rounds  
- A custom federated strategy (FedCustom) is used that:  
  - Averages model weights across all clients  
  - Saves the model after each round  
  - Evaluates the model centrally  
  - Uses two learning rates:  
    - First half of the clients use standard `lr`  
    - Second half use a higher learning rate  

### Key Parameters  
- `number_clients`: 10  
- `max_epochs`: 10  
- `batch_size`: 16  
- `lr`: 1e-3  
- `rounds`: 20  
- `frac_fit`: 1.0  
- `frac_eval`: 0.5  

### Results and Visualization  
This notebook outputs:  
- Confusion matrix for final model evaluation  
- ROC curves for each class  
- Training and validation accuracy/loss plots per client  
- Results are saved in `results/FL_DNA/`
