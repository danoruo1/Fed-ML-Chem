## Standard Centralized Neural Network for Wafer Dataset

### Overview  
This notebook trains a CNN on wafer images using a centralized setup through the Flower framework. Only one client is used, so it's effectively standard centralized learning. The goal is to classify semiconductor wafers based on manufacturing quality or defects.

### Model Architecture  
The model is a simple CNN with two parts: feature extraction and classification.  
- **Feature Extractor**:  
  - Conv2d: 3 → 16, 3×3 kernel, padding=1  
  - ReLU  
  - MaxPool2d: 2×2  
  - Conv2d: 16 → 32, 3×3 kernel, padding=1  
  - ReLU  
  - MaxPool2d: 2×2  

- **Classifier**:  
  - Flatten layer  
  - Linear: 32×16×16 → 128  
  - ReLU  
  - Linear: 128 → num_classes  

```python
class Net(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
