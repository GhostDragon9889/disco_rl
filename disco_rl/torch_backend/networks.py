# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Torch network modules."""

from collections.abc import Sequence

import torch
from torch import nn


class MLPActorCritic(nn.Module):
  """A compact MLP actor-critic network for discrete-action tasks."""

  def __init__(
      self,
      observation_shape: Sequence[int],
      num_actions: int,
      hidden_sizes: Sequence[int] = (256, 256),
  ):
    super().__init__()
    if num_actions < 1:
      raise ValueError('num_actions must be positive.')
    input_size = 1
    for dim in observation_shape:
      input_size *= int(dim)

    layers: list[nn.Module] = []
    last_size = input_size
    for hidden_size in hidden_sizes:
      layers.append(nn.Linear(last_size, int(hidden_size)))
      layers.append(nn.ReLU())
      last_size = int(hidden_size)
    self.torso = nn.Sequential(*layers)
    self.policy_head = nn.Linear(last_size, num_actions)
    self.value_head = nn.Linear(last_size, 1)

  def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = observations.to(dtype=torch.float32).flatten(start_dim=1)
    features = self.torso(x)
    logits = self.policy_head(features)
    values = self.value_head(features).squeeze(-1)
    return logits, values
