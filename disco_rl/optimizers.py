"""Optimizer helpers for the PyTorch implementation."""

from __future__ import annotations

import torch


def adam(params, learning_rate: float, **kwargs) -> torch.optim.Adam:
  return torch.optim.Adam(params, lr=learning_rate, **kwargs)
