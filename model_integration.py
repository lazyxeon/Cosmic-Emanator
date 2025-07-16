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
