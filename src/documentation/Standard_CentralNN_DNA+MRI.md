## Standard Centralized Neural Network for DNA+MRI Multimodal Dataset

### Overview  
This notebook trains one neural network using two types of data at the same time — DNA sequences and MRI images. The model tries to classify both using a single, centralized model.

### Model Architecture  
- A multimodal design with dedicated sub-networks for each data type:  
  - **CNN**: Used for image tasks by detecting patterns like edges, textures, and shapes  
  - **MRINet** (for MRI):  
    - Convolutional layers with ReLU and max pooling  
    - Followed by fully connected layers that reduce down to a shared expert vector  
  - **DNANet** (for DNA):  
    - A fully connected network with 5 linear layers and ReLU activations  
    - Final layer outputs to the same shared expert vector  
  - **Fusion Layer**:  
    - Uses MultiheadAttention to combine features from both DNA and MRI  
    - Includes a gating mechanism to balance the two inputs  
    - Two separate output heads handle classification for DNA and MRI tasks  

### Dataset  
- A combined dataset with both DNA and MRI data  
  - MRI data: 4 classes — 'glioma', 'meningioma', 'notumor', 'pituitary'  
  - DNA data: 7 classes — '0' through '6'  
- MRI images are resized to 224x224  
- DNA sequences are preprocessed using text-based techniques  
- Custom `MultimodalDataset` class handles both inputs together  
- Data is split 90% training and 10% validation  

### Training Approach  
- Everything is trained centrally on 1 client using the Flower framework  
- Optimizer: Adam with learning rate of 1e-3  
- Two separate loss functions (CrossEntropyLoss) for MRI and DNA  
- Trains for 25 epochs  
- Specialized functions are used for training and evaluation across both data types  
- Uses a custom federated learning strategy (FedCustom) that:  
  - Averages model weights after each round  
  - Saves model checkpoints each round  
  - Evaluates on the test set after training  

### Key Parameters  
- `number_clients`: 1  
- `max_epochs`: 25  
- `batch_size`: 32  
- `lr`: 1e-3  
- `rounds`: 1  
- `frac_fit`: 1.0  
- `frac_eval`: 0.5  
- `mri_n_qubits`: 4  
- `dna_n_qubits`: 7  
- `expert_vector`: 6  
- `num_of_expert`: 2  

### Results and Visualization  
This notebook outputs:  
- Confusion matrices for both DNA and MRI classification  
- ROC curves for each task  
- Ac
