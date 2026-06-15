"""PyTorch compatibility helpers retained for public imports."""
from __future__ import annotations
import torch

class Identity(torch.nn.Module):
  def forward(self, *args, **kwargs):
    return args[0] if args else kwargs

def identity(x, *args, **kwargs):
  return x

LSTM = Identity
CategoricalActionModel = Identity
scale_by_adam_sg_denom = lambda *a, **k: None
