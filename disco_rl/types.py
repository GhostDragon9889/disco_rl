"""PyTorch-native types used by DiscoRL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from dm_env import specs

Array = torch.Tensor
ArraySpec = specs.Array
ActionSpec = specs.BoundedArray
Specs = dict[str, specs.Array]
SpecsTree = ArraySpec | Sequence['SpecsTree'] | dict[str, 'SpecsTree']
MetaState = dict[str, Any]
UpdateRuleLog = dict[str, torch.Tensor]
AgentOuts = dict[str, Any]
UpdateRuleOuts = dict[str, Any]
HyperParams = dict[str, torch.Tensor | float]
OptState = Any
AgentParams = Any
MetaParams = Any
MetaParamsEMA = dict[float, MetaParams]
LogDict = dict[str, torch.Tensor]
RNNState = Any


@dataclass
class ValueFnConfig:
  net: str
  net_args: dict[str, Any]
  learning_rate: float
  max_abs_update: float
  discount_factor: float
  td_lambda: float
  outer_value_cost: float
  ema_decay: float = 0.99
  ema_eps: float = 1e-6


@dataclass
class EmaState:
  moment1: Any
  moment2: Any
  decay_product: torch.Tensor


@dataclass
class ValueState:
  params: Any
  state: Any
  opt_state: Any
  adv_ema_state: EmaState
  td_ema_state: EmaState


@dataclass
class TransformConfig:
  source: str
  transforms: Sequence[str | Callable[[Any], torch.Tensor]]


@dataclass
class MetaNetInputOption:
  base: Sequence[TransformConfig]
  action_conditional: Sequence[TransformConfig]


@dataclass(frozen=True)
class PolicyNetwork:
  init: Callable[..., Any]
  one_step: Callable[..., Any]
  unroll: Callable[..., Any]


@dataclass
class EnvironmentTimestep:
  step_type: torch.Tensor
  reward: torch.Tensor
  discount: torch.Tensor
  observation: Any


@dataclass
class ActorTimestep:
  observations: Any
  actions: torch.Tensor
  agent_outs: AgentOuts
  rewards: torch.Tensor
  discounts: torch.Tensor
  states: Any
  logits: torch.Tensor


@dataclass
class ActorRollout:
  observations: Any
  actions: torch.Tensor
  agent_outs: AgentOuts
  rewards: torch.Tensor
  discounts: torch.Tensor
  states: Any | None = None
  logits: torch.Tensor | None = None


@dataclass
class UpdateRuleInputs:
  observations: Any
  actions: torch.Tensor
  rewards: torch.Tensor
  is_terminal: torch.Tensor
  agent_out: AgentOuts
  behaviour_agent_out: AgentOuts | None = None
  value_out: Any | None = None
  extra_from_rule: dict[str, Any] | None = None
  should_reset_mask_fwd: torch.Tensor | None = None


@dataclass
class ValueOuts:
  value: torch.Tensor
  target_value: torch.Tensor
  rho: torch.Tensor
  adv: torch.Tensor
  normalized_adv: torch.Tensor
  td: torch.Tensor
  normalized_td: torch.Tensor
  value_target: torch.Tensor
  qv_adv: Any
  normalized_qv_adv: Any
  q_target: Any
  q_value: Any
  target_q_value: Any
  q_td: Any
  normalized_q_td: Any
