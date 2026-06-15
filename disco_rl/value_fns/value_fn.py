"""Standalone PyTorch value-function trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ValueState:
  module: nn.Module
  optimizer: torch.optim.Optimizer


class ValueFunction:
  def __init__(self, module: nn.Module, learning_rate: float = 3e-4, max_grad_norm: float = 1.0):
    self.module = module
    self.optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    self.max_grad_norm = max_grad_norm

  def update(self, observations: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    values = self.module(observations).squeeze(-1)
    loss = 0.5 * (values - targets).pow(2).mean()
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return {"value_loss": loss.detach()}
