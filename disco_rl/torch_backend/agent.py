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

"""Torch-native actor-critic agent."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from disco_rl.torch_backend.environments import TorchCatchEnv
from disco_rl.torch_backend.networks import MLPActorCritic
from disco_rl.torch_backend.types import Rollout
from disco_rl.torch_backend.types import TimeStep


@dataclass(frozen=True)
class ActorCriticConfig:
  """Training hyper-parameters for `ActorCriticAgent`."""

  learning_rate: float = 1e-3
  gamma: float = 0.99
  gae_lambda: float = 0.95
  value_cost: float = 0.5
  entropy_cost: float = 0.01
  max_grad_norm: float = 10.0


class ActorCriticAgent:
  """Minimal PyTorch actor-critic trainer for discrete-action environments."""

  def __init__(
      self,
      network: MLPActorCritic,
      config: ActorCriticConfig | None = None,
      optimizer: torch.optim.Optimizer | None = None,
  ):
    self.network = network
    self.config = config or ActorCriticConfig()
    self.optimizer = optimizer or torch.optim.Adam(
        self.network.parameters(), lr=self.config.learning_rate
    )

  @property
  def device(self) -> torch.device:
    return next(self.network.parameters()).device

  def act(
      self, timestep: TimeStep
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples actions and returns action/log-prob/entropy/value tensors."""
    logits, values = self.network(timestep.observation.to(self.device))
    distribution = Categorical(logits=logits)
    actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    entropies = distribution.entropy()
    return actions, log_probs, entropies, values

  @torch.no_grad()
  def collect_rollout(self, env: TorchCatchEnv, rollout_length: int) -> Rollout:
    """Collects a time-major rollout from `env`."""
    if rollout_length < 1:
      raise ValueError('rollout_length must be at least 1.')
    timestep = env.reset()
    observations = []
    actions = []
    rewards = []
    dones = []
    logits = []
    values = []
    log_probs = []
    entropies = []
    for _ in range(rollout_length):
      step_logits, step_values = self.network(timestep.observation.to(self.device))
      distribution = Categorical(logits=step_logits)
      step_actions = distribution.sample()
      next_timestep = env.step(step_actions)
      observations.append(timestep.observation)
      actions.append(step_actions)
      rewards.append(next_timestep.reward)
      dones.append(next_timestep.done)
      logits.append(step_logits)
      values.append(step_values)
      log_probs.append(distribution.log_prob(step_actions))
      entropies.append(distribution.entropy())
      timestep = next_timestep
    return Rollout(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dones=torch.stack(dones),
        logits=torch.stack(logits),
        values=torch.stack(values),
        log_probs=torch.stack(log_probs),
        entropies=torch.stack(entropies),
    )

  def update(
      self, rollout: Rollout, bootstrap_value: torch.Tensor | None = None
  ) -> dict[str, float]:
    """Runs one actor-critic update from a rollout."""
    rewards = rollout.rewards.to(self.device)
    dones = rollout.dones.to(self.device)
    old_actions = rollout.actions.to(self.device)
    observations = rollout.observations.to(self.device)

    time_steps, batch_size = rewards.shape[:2]
    flat_observations = observations.reshape(
        time_steps * batch_size, *observations.shape[2:]
    )
    logits, values = self.network(flat_observations)
    logits = logits.reshape(time_steps, batch_size, -1)
    values = values.reshape(time_steps, batch_size)
    distribution = Categorical(logits=logits)
    log_probs = distribution.log_prob(old_actions)
    entropies = distribution.entropy()

    if bootstrap_value is None:
      bootstrap_value = torch.zeros(batch_size, dtype=values.dtype, device=self.device)
    returns, advantages = self._generalized_advantage_estimate(
        rewards=rewards,
        dones=dones,
        values=values,
        bootstrap_value=bootstrap_value.to(self.device),
    )
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = nn.functional.mse_loss(values, returns.detach())
    entropy = entropies.mean()
    total_loss = (
        policy_loss
        + self.config.value_cost * value_loss
        - self.config.entropy_cost * entropy
    )

    self.optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(
        self.network.parameters(), self.config.max_grad_norm
    )
    self.optimizer.step()

    return {
        'total_loss': float(total_loss.detach().cpu()),
        'policy_loss': float(policy_loss.detach().cpu()),
        'value_loss': float(value_loss.detach().cpu()),
        'entropy': float(entropy.detach().cpu()),
        'grad_norm': float(grad_norm.detach().cpu()),
    }

  def _generalized_advantage_estimate(
      self,
      rewards: torch.Tensor,
      dones: torch.Tensor,
      values: torch.Tensor,
      bootstrap_value: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    next_value = bootstrap_value
    next_advantage = torch.zeros_like(bootstrap_value)
    for t in range(rewards.shape[0] - 1, -1, -1):
      not_done = (~dones[t]).to(values.dtype)
      delta = rewards[t] + self.config.gamma * next_value * not_done - values[t]
      next_advantage = (
          delta
          + self.config.gamma
          * self.config.gae_lambda
          * not_done
          * next_advantage
      )
      advantages[t] = next_advantage
      next_value = values[t]
    returns = advantages + values
    return returns, advantages
