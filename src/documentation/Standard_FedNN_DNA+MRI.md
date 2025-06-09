## Standard Federated Neural Network for DNA+MRI Multimodal Dataset

### Overview  
This notebook uses federated learning to train on a multimodal dataset that combines DNA sequences and MRI images. Each client trains locally on a portion of the data, and the server aggregates the models using the Flower framework. The model performs two classification tasks at once — one for DNA and one for MRI.

### Model Architecture  
The model is split into two branches — one for each modality — with a shared fusion layer:  
- **MRINet**: CNN used for MRI image classification  
  - Convolutional layers with ReLU and max pooling  
  - Followed by fully connected layers that produce a shared expert vector  
- **DNANet**: Fully connected network for DNA sequence classification  
  - 5 linear layers with ReLU  
  - Final output maps into the shared expert vector  
- **Fusion**:  
  - MultiheadAttention to combine features from both branches  
  - Gating mechanism to balance the two modalities  
  - Ends with two separate heads for final classification (DNA and MRI)  

### Dataset  
- Mixed dataset with both DNA and MRI data  
  - MRI classes: 'glioma', 'meningioma', 'notumor', 'pituitary'  
  - DNA classes: '0' through '6'  
- MRI images resized to 224×224  
- DNA processed with standard sequence techniques  
- Managed using a custom `MultimodalDataset` class  
- Data is split across 10 clients  
- Each client uses a 90/10 training-validation split  

### Training Approach  
- Federated setup with 10 clients  
- Optimizer: Adam with learning rate = 1e-3  
- Loss: two CrossEntropyLoss functions (one for each modality)  
- Each client trains for 10 epochs per round  
- Total: 20 rounds  
- Uses `engine.train_multimodal` and `engine.test_multimodal` for training and evaluation  
- Custom FedCustom strategy:  
  - Aggregates weights with weighted averaging  
  - Saves checkpoints after each round  
  - Evaluates after each round  
  - Half of the clients train with standard lr, others use a higher lr  

### Key Parameters  
- `number_clients`: 10  
- `max_epochs`: 10  
- `batch_size`: 32  
- `lr`: 1e-3  
- `rounds`: 20  
- `frac_fit`: 1.0  
- `frac_eval`: 0.5  
- `mri_n_qubits`: 4  
- `dna_n_qubits`: 7  
- `expert_vector`: 6  
- `num_of_expert`: 2  

### Results and Visualization  
This notebook generates:  
- Confusion matrices (separate for MRI and DNA)  
- ROC curves for both tasks  
- Accuracy/loss curves per client for each modality  
- All outputs saved in `results/FL_DNA+MRI/`
