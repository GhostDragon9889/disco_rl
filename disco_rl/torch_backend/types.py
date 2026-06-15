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

"""Typed containers used by the PyTorch backend."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TimeStep:
  """Batched environment transition tensors.

  Attributes:
    observation: Observation tensor with shape `[B, ...]`.
    reward: Float reward tensor with shape `[B]`.
    done: Boolean terminal tensor with shape `[B]`.
  """

  observation: torch.Tensor
  reward: torch.Tensor
  done: torch.Tensor


@dataclass(frozen=True)
class Rollout:
  """Time-major rollout emitted by `ActorCriticAgent.collect_rollout`."""

  observations: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  dones: torch.Tensor
  logits: torch.Tensor
  values: torch.Tensor
  log_probs: torch.Tensor
  entropies: torch.Tensor
