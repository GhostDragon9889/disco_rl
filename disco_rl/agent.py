"""PyTorch DiscoRL agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from disco_rl import types
from disco_rl.networks import nets
from disco_rl.update_rules import actor_critic, disco, policy_gradient


@dataclass
class LearnerState:
  params: nn.Module
  optimizer: torch.optim.Optimizer
  meta_state: types.MetaState


class Agent:
  def __init__(self, *, single_observation_spec: Any, single_action_spec: types.ActionSpec, agent_settings: Any, batch_axis_name: str | None = None, device: str | torch.device | None = None):
    del batch_axis_name
    self.settings = agent_settings
    self.single_observation_spec = single_observation_spec
    self.single_action_spec = single_action_spec
    self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if getattr(agent_settings, "update_rule_name", "disco") == "disco":
      self.update_rule = disco.DiscoUpdateRule(**agent_settings.update_rule)
    elif agent_settings.update_rule_name == "actor_critic":
      self.update_rule = actor_critic.ActorCritic(**agent_settings.update_rule)
    elif agent_settings.update_rule_name == "policy_gradient":
      self.update_rule = policy_gradient.PolicyGradientUpdate(**agent_settings.update_rule)
    else:
      raise ValueError(f"Unsupported update rule: {agent_settings.update_rule_name}")
    self.network = nets.get_network(name=agent_settings.net_settings.name, observation_shape=single_observation_spec.shape, action_spec=single_action_spec, **agent_settings.net_settings.net_args).module.to(self.device)
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(agent_settings.learning_rate))

  def initial_learner_state(self) -> LearnerState:
    return LearnerState(self.network, self.optimizer, self.update_rule.init_meta_state(self.network))

  @torch.no_grad()
  def actor_step(self, observation: np.ndarray | torch.Tensor) -> types.ActorTimestep:
    obs = torch.as_tensor(observation, device=self.device).unsqueeze(0)
    out = self.network(obs)
    dist = torch.distributions.Categorical(logits=out["logits"])
    action = dist.sample()
    return types.ActorTimestep(obs, action, out, torch.zeros(1, device=self.device), torch.ones(1, device=self.device), None, out["logits"])

  def update(self, rollout: types.ActorRollout, state: LearnerState | None = None, is_meta_training: bool = False) -> tuple[LearnerState, dict[str, torch.Tensor]]:
    state = state or self.initial_learner_state()
    observations = rollout.observations.to(self.device)
    actions = rollout.actions.to(self.device)
    rewards = rollout.rewards.to(self.device)
    discounts = rollout.discounts.to(self.device)
    agent_out = state.params(observations)
    inputs = types.UpdateRuleInputs(observations, actions, rewards[1:] if rewards.shape[0] == observations.shape[0] else rewards, discounts[1:] == 0 if discounts.shape[0] == observations.shape[0] else discounts == 0, agent_out, rollout.agent_outs)
    hyper = self.settings.hyper_params.to_dict() if hasattr(self.settings.hyper_params, "to_dict") else dict(self.settings.hyper_params)
    if isinstance(self.update_rule, disco.DiscoUpdateRule):
      meta_out, meta_state = self.update_rule.unroll_meta_net(state.params, state.meta_state, inputs, hyper)
    else:
      meta_out, meta_state = {}, state.meta_state
    loss, log = self.update_rule.agent_loss(inputs, meta_out, hyper, backprop=is_meta_training)
    loss2, log2 = self.update_rule.agent_loss_no_meta(inputs, meta_out, hyper)
    total = (loss + loss2).mean()
    state.optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(state.params.parameters(), float(getattr(self.settings, "max_abs_update", 1.0)))
    state.optimizer.step()
    return LearnerState(state.params, state.optimizer, meta_state), {"loss": total.detach(), **log, **log2}
