"""PyTorch utility functions for DiscoRL."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from disco_rl import types


def get_num_actions_from_spec(action_spec: types.ActionSpec) -> int:
  return int(action_spec.maximum - action_spec.minimum + 1)


def get_logits_specs(action_spec: types.ActionSpec) -> dict[str, tuple[int, ...]]:
  return {"logits": (get_num_actions_from_spec(action_spec),)}


def batch_lookup(values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
  """Gather action-conditioned values with leading [T, B] or [B] dimensions."""
  actions = actions.long()
  if values.ndim == actions.ndim + 1:
    return values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
  index = actions.unsqueeze(-1).unsqueeze(-1).expand(*actions.shape, 1, *values.shape[actions.ndim + 1:])
  return values.gather(actions.ndim, index).squeeze(actions.ndim)


def categorical_kl(target_logits: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
  target = F.softmax(target_logits, dim=-1)
  return (target * (F.log_softmax(target_logits, dim=-1) - F.log_softmax(logits, dim=-1))).sum(dim=-1)


class MovingAverage:
  """Exponential moving average normalizer."""

  def __init__(self, decay: float = 0.99, eps: float = 1e-6):
    self.decay = decay
    self.eps = eps

  def init_state(self, device: torch.device | None = None) -> types.EmaState:
    return types.EmaState(
        mean=torch.zeros((), device=device),
        variance=torch.ones((), device=device),
        count=torch.zeros((), device=device),
    )

  def update_state(self, x: torch.Tensor, state: types.EmaState) -> types.EmaState:
    mean = x.detach().mean()
    var = x.detach().var(unbiased=False).clamp_min(self.eps)
    if state.count.item() == 0:
      return types.EmaState(mean, var, torch.ones_like(state.count))
    return types.EmaState(
        self.decay * state.mean + (1.0 - self.decay) * mean,
        self.decay * state.variance + (1.0 - self.decay) * var,
        state.count + 1,
    )

  def normalize(self, x: torch.Tensor, state: types.EmaState) -> torch.Tensor:
    return (x - state.mean) / torch.sqrt(state.variance + self.eps)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, coeff: float) -> None:
  with torch.no_grad():
    for tgt, src in zip(target.parameters(), source.parameters()):
      tgt.mul_(coeff).add_(src, alpha=1.0 - coeff)
