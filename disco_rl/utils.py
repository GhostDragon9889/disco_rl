"""PyTorch utility functions for DiscoRL."""
from __future__ import annotations

from typing import Any, Sequence, TypeVar
import numpy as np
import torch
import torch.nn.functional as F
from dm_env import specs
from disco_rl import types

_T = TypeVar('_T')


def tree_map(fn, tree, *rest):
  if isinstance(tree, dict):
    return {k: tree_map(fn, tree[k], *(r[k] for r in rest)) for k in tree}
  if isinstance(tree, (list, tuple)):
    return type(tree)(tree_map(fn, x, *(r[i] for r in rest)) for i, x in enumerate(tree))
  return fn(tree, *rest)


def shard_across_devices(data: _T, devices: Sequence[torch.device]) -> _T:
  del devices
  return data


def gather_from_devices(data: _T) -> _T:
  return data


def batch_lookup(table: torch.Tensor, index: torch.Tensor, num_dims: int = 2) -> torch.Tensor:
  del num_dims
  index = index.long().unsqueeze(-1)
  if table.ndim == index.ndim:
    return table.gather(-1, index).squeeze(-1)
  return table.gather(-2, index.unsqueeze(-1).expand(*index.shape, table.shape[-1])).squeeze(-2)


def broadcast_specs(spec_tree, n: int, replace: bool = False):
  def prep(s):
    shape = (n,) + tuple(s.shape[1:] if replace else s.shape)
    return type(s)(shape=shape, dtype=s.dtype, name=getattr(s, 'name', None))
  return tree_map(prep, spec_tree)


def tree_stack(elems: Sequence[Any], axis: int = 0) -> Any:
  return tree_map(lambda *xs: torch.stack([torch.as_tensor(x) for x in xs], dim=axis), elems[0], *elems[1:])


def cast_to_single_precision(tree_like: _T, cast_ints: bool = True, host_data: bool = False) -> _T:
  del host_data
  def cast(x):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
      return x.float()
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.floating):
      return x.astype(np.float32)
    if cast_ints and isinstance(x, np.ndarray) and x.dtype == np.int64:
      return x.astype(np.int32)
    return x
  return tree_map(cast, tree_like)


def get_num_actions_from_spec(spec: types.ActionSpec) -> int:
  return int(spec.maximum - spec.minimum + 1)


def get_logits_specs(spec: types.ActionSpec, with_batch_dim: bool = False) -> types.ArraySpec:
  del with_batch_dim
  return specs.Array((get_num_actions_from_spec(spec),), np.float32)


def zeros_like_spec(spec_tree: Any, prepend_shape: tuple[int, ...] = ()): 
  return tree_map(lambda s: torch.zeros(prepend_shape + tuple(s.shape), dtype=torch.float32), spec_tree)


def differentiable_policy_gradient_loss(logits_t: torch.Tensor, a_t: torch.Tensor, adv_t: torch.Tensor, backprop: bool) -> torch.Tensor:
  log_pi_a = F.log_softmax(logits_t, dim=-1).gather(-1, a_t.long().unsqueeze(-1)).squeeze(-1)
  return -log_pi_a * (adv_t if backprop else adv_t.detach())


class MovingAverage:
  def __init__(self, example_tree: Any, decay: float = 0.999, eps: float = 1e-6):
    self._example_tree = example_tree; self._decay = decay; self._eps = eps
  def init_state(self) -> types.EmaState:
    zeros = tree_map(lambda _: torch.tensor(0., dtype=torch.float32), self._example_tree)
    return types.EmaState(zeros, zeros, torch.tensor(1., dtype=torch.float32))
  def update_state(self, tree_like: Any, state: types.EmaState, pmean_axis_name: str | None = None) -> types.EmaState:
    del pmean_axis_name
    m1 = tree_map(lambda m, x: self._decay*m + (1-self._decay)*torch.as_tensor(x).float().mean(), state.moment1, tree_like)
    m2 = tree_map(lambda m, x: self._decay*m + (1-self._decay)*torch.square(torch.as_tensor(x).float()).mean(), state.moment2, tree_like)
    return types.EmaState(m1, m2, state.decay_product * self._decay)
  def normalize(self, tree_like: Any, state: types.EmaState) -> Any:
    def norm(x, m1, m2):
      bias = 1 - state.decay_product
      mean = m1 / bias.clamp_min(self._eps); second = m2 / bias.clamp_min(self._eps)
      return (x - mean) / torch.sqrt((second - mean.square()).clamp_min(0) + self._eps)
    return tree_map(norm, tree_like, state.moment1, state.moment2)
