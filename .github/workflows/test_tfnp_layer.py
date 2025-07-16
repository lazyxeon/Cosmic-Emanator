import torch
from src.tfnp_layer import TFNPLayer

def test_tfnp_output_shape():
    model = TFNPLayer(3, 64)
    x = torch.rand(1, 3, 32, 32)
    y = model(x, t=1.0)
    assert y.shape == (1, 64, 32, 32), "Output shape mismatch"

def test_tfnp_no_nan():
    model = TFNPLayer(3, 64)
    x = torch.rand(1, 3, 32, 32)
    y = model(x, t=1.0)
    assert not torch.isnan(y).any(), "Output contains NaN"
