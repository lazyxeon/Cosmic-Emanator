# examples/basic/forward_pass_demo.py

import torch
from src.tfnp_layer import TFNPLayer

print("=== Forward Pass Demo ===")
model = TFNPLayer(3, 64)
input_tensor = torch.rand(1, 3, 32, 32)
output = model(input_tensor, t=1.0)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
