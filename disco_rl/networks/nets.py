"""PyTorch policy/value networks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from disco_rl import types, utils


class MLP(nn.Module):
  """Shared torso with policy, categorical prediction and Q heads."""

  def __init__(
      self,
      observation_shape: Iterable[int],
      action_spec: types.ActionSpec,
      dense: Iterable[int] = (64, 64),
      prediction_size: int = 32,
      num_bins: int = 51,
      head_w_init_std: float | None = 0.01,
      **_: Any,
  ) -> None:
    super().__init__()
    self.num_actions = utils.get_num_actions_from_spec(action_spec)
    self.prediction_size = prediction_size
    self.num_bins = num_bins
    in_dim = int(np.prod(tuple(observation_shape)))
    layers: list[nn.Module] = [nn.Flatten()]
    last = in_dim
    for width in dense:
      layers += [nn.Linear(last, int(width)), nn.ReLU()]
      last = int(width)
    self.torso = nn.Sequential(*layers)
    self.logits = nn.Linear(last, self.num_actions)
    self.y = nn.Linear(last, prediction_size)
    self.z = nn.Linear(last, self.num_actions * prediction_size)
    self.aux_pi = nn.Linear(last, self.num_actions * self.num_actions)
    self.q = nn.Linear(last, self.num_actions * num_bins)
    if head_w_init_std:
      for head in (self.logits, self.y, self.z, self.aux_pi, self.q):
        nn.init.trunc_normal_(head.weight, std=head_w_init_std)
        nn.init.zeros_(head.bias)

  def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
    orig_shape = observations.shape[:-len(observations.shape[-2:])] if observations.ndim > 2 else observations.shape[:-1]
    flat = observations.reshape(-1, *observations.shape[-2:]) if observations.ndim > 2 else observations.reshape(-1, observations.shape[-1])
    h = self.torso(flat.float())
    out = {
        "logits": self.logits(h),
        "y": self.y(h),
        "z": self.z(h).reshape(-1, self.num_actions, self.prediction_size),
        "aux_pi": self.aux_pi(h).reshape(-1, self.num_actions, self.num_actions),
        "q": self.q(h).reshape(-1, self.num_actions, self.num_bins),
    }
    return {k: v.reshape(*orig_shape, *v.shape[1:]) for k, v in out.items()}


def get_network(
    name: str,
    *,
    observation_shape: Iterable[int],
    action_spec: types.ActionSpec,
    out_spec: Mapping[str, Any] | None = None,
    model_out_spec: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> types.PolicyNetwork:
  del out_spec, model_out_spec
  if name != "mlp":
    raise ValueError(f"Unknown network: {name}")
  module = MLP(observation_shape=observation_shape, action_spec=action_spec, **kwargs)
  return types.PolicyNetwork(module=module, one_step=module, unroll=module)
