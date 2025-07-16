# examples/basic/forward_pass_demo.py

import torch
from src.tfnp_layer import TFNPLayer

print("=== Forward Pass Demo ===")
model = TFNPLayer(3, 64)
input_tensor = torch.rand(1, 3, 32, 32)
output = model(input_tensor, t=1.0)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)

# examples/basic/forward_pass_demo.py

import torch
from src.tfnp_layer import TFNPLayer
import matplotlib.pyplot as plt

print("=== Forward Pass Demo ===")
model = TFNPLayer(3, 64)  # Initialize with 3 input channels, 64 output

# Single pass
input_tensor = torch.rand(1, 3, 32, 32)  # Random input (e.g., image-like)
output = model(input_tensor, t=1.0)

print("Input shape:", input_tensor.shape)  # Expected: torch.Size([1, 3, 32, 32])
print("Output shape:", output.shape)  # Expected: torch.Size([1, 64, 32, 32])

# Assert shape preservation (spatial dims)
assert input_tensor.shape[2:] == output.shape[2:], "Spatial dimensions not preserved!"

# Visualize a channel slice of output
plt.imshow(output[0, 0].detach().numpy(), cmap='viridis')
plt.title("Output Channel 0 Visualization")
plt.colorbar()
plt.show()

# Multiple passes with varying t
for t in [0.5, 1.0, 1.5]:
    output_var = model(input_tensor, t=t)
    print(f"Output shape for t={t}:", output_var.shape)
