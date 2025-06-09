## Standard Centralized Neural Network for HIV Dataset

### Overview  
This notebook trains a centralized neural network on the HIV dataset using a graph-based approach. It uses the Flower framework with one client, so it functions like a standard centralized setup. The model works with molecular graph data to predict whether a compound is active or inactive against HIV.

### Model Architecture  
- Graph Convolutional Network (GCN) setup  
- Structure:  
  - Starts with a GCNConv layer: 9 → `embedding_size`  
  - Followed by three more GCNConv layers with tanh activations  
  - Uses global pooling (max + mean) to combine graph-level features  
  - Final linear layer maps the pooled output to class predictions  
- Each graph:  
  - Nodes = atoms  
  - Edges = bonds  
  - Node features include chemical properties  
  - The graph layout represents molecule structure  

### Dataset  
- HIV dataset for binary classification:  
  - 'CI' = confirmed inactive  
  - 'CA' or 'CM' = confirmed (moderately) active  
- Comes from MoleculeNet  
- Graph data is handled with PyTorch Geometric  
- Dataset is split 90% training, 10% validation  

### Training Approach  
- Runs on one client using the Flower framework  
- Loss function: custom RMSELoss  
- Optimizer: Adam, learning rate = 1e-3  
- Training runs for 25 epochs  
- Custom train and test functions specific for graph data:  
  - `engine.train(task="Graph")`  
  - `engine.test_graph()`  
- Uses a custom federated strategy (FedCustom):  
  - Aggregates parameters with weighted averaging  
  - Saves a checkpoint after each round  
  - Evaluates using a test set  

### Key Parameters  
- `number_clients`: 1  
- `max_epochs`: 25  
- `embedding_size`: 64  
- `batch_size`: 64  
- `lr`: 1e-3  
- `rounds`: 1  
- `frac_fit`: 1.0  
- `frac_eval`: 0.5  

### Results and Visualization  
This notebook saves:  
- Confusion matrix for performance visualization  
- ROC curve for evaluating classification quality  
- Accuracy and loss curves over training and validation  
- Outputs go to `results/CL_HIV/`