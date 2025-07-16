import unittest
import torch
from src.tfnp_layer import TFNPLayer

class TestTFNPLayer(unittest.TestCase):
    def test_output_shape(self):
        model = TFNPLayer(3, 64)
        input_tensor = torch.rand(1, 3, 32, 32)
        output = model(input_tensor, t=1.0)
        self.assertEqual(output.shape, torch.Size([1, 64, 32, 32]))