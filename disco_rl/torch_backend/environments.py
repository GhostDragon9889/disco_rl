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

"""Torch-native environments.

The original repository keeps both CPU and JAX-jittable Catch environments.  The
classes in this module provide the same small Catch task using only PyTorch
operations so rollouts can live directly on CUDA devices.
"""

from dataclasses import dataclass

import torch

from disco_rl.torch_backend.types import TimeStep


@dataclass(frozen=True)
class CatchConfig:
  """Configuration for the Torch Catch environment."""

  rows: int = 8
  columns: int = 8
  batch_size: int = 1
  seed: int = 1
  device: str | torch.device = 'cpu'


class TorchCatchEnv:
  """Batched Catch environment implemented with PyTorch tensors.

  Actions are scalar integers: 0 moves the paddle left, 1 is no-op, and 2 moves
  the paddle right.  Episodes automatically reset on the step after a terminal
  transition, matching the original single-stream Catch behavior.
  """

  num_actions = 3

  def __init__(self, config: CatchConfig):
    if config.rows < 2:
      raise ValueError('CatchConfig.rows must be at least 2.')
    if config.columns < 1:
      raise ValueError('CatchConfig.columns must be at least 1.')
    if config.batch_size < 1:
      raise ValueError('CatchConfig.batch_size must be at least 1.')
    self.config = config
    self.device = torch.device(config.device)
    self.generator = torch.Generator(device=self.device)
    self.generator.manual_seed(config.seed)
    self.ball_y = torch.zeros(config.batch_size, dtype=torch.long, device=self.device)
    self.ball_x = torch.zeros(config.batch_size, dtype=torch.long, device=self.device)
    self.paddle_y = torch.full(
        (config.batch_size,), config.rows - 1, dtype=torch.long, device=self.device
    )
    self.paddle_x = torch.zeros(config.batch_size, dtype=torch.long, device=self.device)
    self._needs_reset = torch.ones(config.batch_size, dtype=torch.bool, device=self.device)

  @property
  def observation_shape(self) -> tuple[int, int, int]:
    return (self.config.rows, self.config.columns, 1)

  def reset(self) -> TimeStep:
    """Resets all batch elements and returns an initial `TimeStep`."""
    self._reset_indices(torch.ones_like(self._needs_reset))
    return TimeStep(
        observation=self._render(),
        reward=torch.zeros(self.config.batch_size, dtype=torch.float32, device=self.device),
        done=torch.zeros(self.config.batch_size, dtype=torch.bool, device=self.device),
    )

  def step(self, actions: torch.Tensor) -> TimeStep:
    """Applies one batched environment step."""
    actions = actions.to(device=self.device, dtype=torch.long).reshape(self.config.batch_size)
    self._reset_indices(self._needs_reset)

    self.paddle_x = torch.clamp(
        self.paddle_x + actions - 1, min=0, max=self.config.columns - 1
    )
    self.ball_y = self.ball_y + 1

    done = self.ball_y == self.paddle_y
    reward = torch.where(
        done,
        torch.where(self.paddle_x == self.ball_x, 1.0, -1.0),
        0.0,
    ).to(torch.float32)
    observation = self._render()
    self._needs_reset = done
    return TimeStep(observation=observation, reward=reward, done=done)

  def _reset_indices(self, mask: torch.Tensor) -> None:
    count = int(mask.sum().item())
    if count == 0:
      return
    self.ball_y[mask] = 0
    self.ball_x[mask] = torch.randint(
        low=0,
        high=self.config.columns,
        size=(count,),
        generator=self.generator,
        device=self.device,
    )
    self.paddle_y[mask] = self.config.rows - 1
    self.paddle_x[mask] = self.config.columns // 2
    self._needs_reset[mask] = False

  def _render(self) -> torch.Tensor:
    board = torch.zeros(
        (self.config.batch_size, self.config.rows, self.config.columns, 1),
        dtype=torch.float32,
        device=self.device,
    )
    batch = torch.arange(self.config.batch_size, device=self.device)
    board[batch, self.ball_y, self.ball_x, 0] = 1.0
    board[batch, self.paddle_y, self.paddle_x, 0] = 1.0
    return board
