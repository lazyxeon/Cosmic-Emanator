import torch
import torch.nn as nn
import math

class TFNPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=3.5, phi=1.618, f=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.alpha = alpha
        self.phi = phi
        self.f = f

    def forward(self, x, t=0.0):
        batch, channels, height, width = x.shape
        phi_grid = torch.linspace(0, 2 * math.pi, height).unsqueeze(1).repeat(1, width)
        theta_grid = torch.linspace(0, 2 * math.pi, width).unsqueeze(0).repeat(height, 1)
        twist = theta_grid + self.alpha * phi_grid % (2 * math.pi)
        twist = twist.unsqueeze(0).unsqueeze(0).repeat(batch, channels, 1, 1)
        x_twisted = x * torch.cos(twist)
        x_scaled = x_twisted * self.phi
        sin_term = torch.sin(torch.tensor(2 * math.pi * self.f * t))
        x_activated = sin_term * self.conv(x_scaled)
        return x_activated