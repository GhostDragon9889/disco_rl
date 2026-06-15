"""Meta-networks used by the Disco update rule."""

from __future__ import annotations

import torch
from torch import nn


class LSTM(nn.Module):
  """Sequence model that predicts policy/prediction targets from RL features."""

  def __init__(self, input_size: int, hidden_size: int = 128, prediction_size: int = 32, num_actions: int = 3):
    super().__init__()
    self.prediction_size = prediction_size
    self.num_actions = num_actions
    self.lstm = nn.LSTM(input_size, hidden_size)
    self.pi = nn.Linear(hidden_size, num_actions)
    self.y = nn.Linear(hidden_size, prediction_size)
    self.z = nn.Linear(hidden_size, prediction_size)

  def forward(self, features: torch.Tensor, state=None):
    h, state = self.lstm(features, state)
    return {"pi": self.pi(h), "y": self.y(h), "z": self.z(h)}, state
