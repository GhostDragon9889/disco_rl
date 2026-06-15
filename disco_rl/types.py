"""PyTorch type definitions for DiscoRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from dm_env import specs

Tensor = torch.Tensor
TensorTree = Tensor | Mapping[str, "TensorTree"] | Sequence["TensorTree"]
ActionSpec = specs.BoundedArray
Specs = dict[str, tuple[int, ...]]
MetaState = dict[str, Any]
UpdateRuleLog = dict[str, Tensor]
UpdateRuleOuts = dict[str, Tensor]
HyperParams = dict[str, float | Tensor]


@dataclass
class EmaState:
  mean: Tensor
  variance: Tensor
  count: Tensor


@dataclass
class ActorRollout:
  observations: Tensor
  actions: Tensor
  rewards: Tensor
  discounts: Tensor
  agent_outs: dict[str, Tensor]
  logits: Tensor | None = None


@dataclass
class UpdateRuleInputs:
  observations: Tensor
  actions: Tensor
  rewards: Tensor
  is_terminal: Tensor
  agent_out: dict[str, Tensor]
  behaviour_agent_out: dict[str, Tensor] | None = None
  value_out: Any | None = None
  extra_from_rule: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValueOuts:
  value: Tensor
  target_value: Tensor
  rho: Tensor
  adv: Tensor
  normalized_adv: Tensor
  td: Tensor
  normalized_td: Tensor
  value_target: Tensor
  qv_adv: Tensor
  normalized_qv_adv: Tensor
  q_target: Tensor
  q_value: Tensor
  target_q_value: Tensor
  q_td: Tensor
  normalized_q_td: Tensor


@dataclass
class ActorTimestep:
  observations: Tensor
  actions: Tensor
  agent_outs: dict[str, Tensor]
  rewards: Tensor
  discounts: Tensor
  states: Any | None
  logits: Tensor


@dataclass
class LearnerState:
  params: Any
  opt_state: Any
  meta_state: MetaState


@dataclass
class PolicyNetwork:
  module: torch.nn.Module
  one_step: Callable[[Tensor], dict[str, Tensor]]
  unroll: Callable[[Tensor], dict[str, Tensor]]


@dataclass(frozen=True)
class TransformConfig:
  source: str
  transforms: Sequence[str | Callable[[Any], Tensor]]


@dataclass(frozen=True)
class MetaNetInputOption:
  base: Sequence[TransformConfig]
  action_conditional: Sequence[TransformConfig]
