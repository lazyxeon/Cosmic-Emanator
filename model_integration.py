# examples/integration/model_integration.py

import torch
import torch.nn as nn
from src.tfnp_layer import TFNPLayer

class CosmicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TFNPLayer(3, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layer1(x, t=1.0)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CosmicCNN()
input_tensor = torch.rand(8, 3, 32, 32)
output = model(input_tensor)
print("Output shape:", output.shape)


# examples/integration/model_integration.py

import torch
import torch.nn as nn
from src.tfnp_layer import TFNPLayer

class CosmicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TFNPLayer(3, 64)  # Cosmic layer for feature extraction
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        self.fc = nn.Linear(64, 10)  # Classifier for 10 classes (e.g., CIFAR-10)

    def forward(self, x, t=1.0):
        x = self.layer1(x, t=t)  # Apply cosmic transformation
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CosmicCNN().to(device)
input_tensor = torch.rand(8, 3, 32, 32).to(device)  # Batch of random images
output = model(input_tensor, t=1.0)
print("Output shape:", output.shape)  # Expected: torch.Size([8, 10])

# Simple inference demo (random input, argmax prediction)
predictions = output.argmax(dim=1)
print("Sample predictions:", predictions.tolist())