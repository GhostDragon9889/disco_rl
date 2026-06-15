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

"""Utility functions."""

import functools
from typing import Any, Sequence, TypeVar

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import rlax
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from disco_rl import types

_T = TypeVar('_T')
_SpecsT = TypeVar('_SpecsT')


def shard_across_devices(data: _T, devices: Sequence[jax.Device]) -> _T:
  """Splits a leading batch axis and places shards on `devices`.

  This is the project-local replacement for the removed
  `jax.device_put_sharded` API.  It keeps the previous host-facing contract:
  the input tree is split along axis 0 into one shard per device, and the
  returned tree has a sharded leading axis that can be consumed by `jax.pmap`.
  """
  num_shards = len(devices)
  leaves, treedef = jax.tree_util.tree_flatten(data)
  split_leaves = [np.split(leaf, num_shards, axis=0) for leaf in leaves]
  flat_shards = ((leaf[i] for leaf in split_leaves) for i in range(num_shards))
  data_shards = [jax.tree_util.tree_unflatten(treedef, shard) for shard in flat_shards]
  mesh = Mesh(np.asarray(devices), ('devices',))
  sharding = NamedSharding(mesh, P('devices'))
  return jax.tree_util.tree_map(
      lambda *xs: jax.device_put(jnp.stack(xs), sharding), *data_shards
  )


def replicate_across_devices(data: _T, devices: Sequence[jax.Device]) -> _T:
  """Replicates a pytree on `devices` without `jax.device_put_replicated`.

  The returned tree has a sharded leading axis of length `len(devices)`, matching
  the legacy API's shape contract for inputs to `jax.pmap`.
  """
  mesh = Mesh(np.asarray(devices), ('devices',))
  sharding = NamedSharding(mesh, P('devices'))
  return jax.tree_util.tree_map(
      lambda x: jax.device_put(jnp.stack([x] * len(devices)), sharding), data
  )


def gather_from_devices(data: _T) -> _T:
  return jax.tree_util.tree_map(
      lambda x: x.reshape((-1, *x.shape[2:])), jax.device_get(data)
  )


def batch_lookup(
    table: chex.Array, index: chex.Array, num_dims: int = 2
) -> chex.Array:

  def _lookup(table: chex.Array, index: chex.Array) -> chex.Array:
    return jax.vmap(lambda x, i: x[i])(table, index)

  if index is not None:
    index = index.astype(jnp.int32)
  return hk.BatchApply(_lookup, num_dims=num_dims)(table, index)


def broadcast_specs(specs: _SpecsT, n: int, replace: bool = False) -> _SpecsT:
  """Prepends `n` to the specs' shapes.

  Args:
    specs: specs to broadcast.
    n: a value to prepend to shapes.
    replace: whether to replace or prepend the first dimension.

  Returns:
    Broadcasted specs.
  """
  f_i = 1 if replace else 0

  def _prepend(s: types.ArraySpec | types.specs.Array):
    if isinstance(s, types.specs.Array):
      return s.replace(shape=(n,) + s.shape[f_i:])
    elif isinstance(s, types.ArraySpec):
      return types.ArraySpec(shape=(n,) + s.shape[f_i:], dtype=s.dtype)
    else:
      raise ValueError(f'Unsupported spec type: {type(s)}')

  return jax.tree_util.tree_map(_prepend, specs)


def tree_stack(
    elems: Sequence[chex.ArrayTree], axis: int = 0
) -> chex.ArrayTree:
  """Stacks a sequence of trees into a single tree."""
  return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *elems)


def cast_to_single_precision(
    tree_like: _T, cast_ints: bool = True, host_data: bool = False
) -> _T:
  """Casts the data to full precision."""
  if host_data:

    def conditional_cast(x):
      if isinstance(x, (np.ndarray, jnp.ndarray)):
        if np.issubdtype(x.dtype, np.floating) or jnp.issubdtype(
            x.dtype, jnp.floating
        ):
          if x.dtype != np.float32:
            x = x.astype(np.float32)
        elif cast_ints and x.dtype == np.int64:
          x = x.astype(np.int32)
      return x

    return jax.tree_util.tree_map(conditional_cast, tree_like)
  else:
    return jmp.cast_to_full(tree_like)


def get_num_actions_from_spec(spec: types.ActionSpec) -> int:
  """Returns the number of actions from the action spec."""
  return spec.maximum - spec.minimum + 1


def get_logits_specs(
    spec: types.ActionSpec, with_batch_dim: bool = False
) -> types.ArraySpec:
  """Extracts a tree of shapes for logits for the provided spec."""
  if with_batch_dim:
    spec = spec.replace(shape=spec.shape[1:])

  return types.ArraySpec((get_num_actions_from_spec(spec),), np.float32)


def zeros_like_spec(spec: Any, prepend_shape: tuple[int, ...] = ()):
  """Returns a tree of zeros from `spec`.

  Args:
    spec: a tree of `array_like`s or specs.
    prepend_shape: a tuple of integers to prepend to the shapes.

  Returns:
    A tree of zero arrays.
  """
  return jax.tree_util.tree_map(
      lambda spec: np.zeros(shape=prepend_shape + spec.shape, dtype=spec.dtype),
      spec,
  )


def differentiable_policy_gradient_loss(
    logits_t: chex.Array, a_t: chex.Array, adv_t: chex.Array, backprop: bool
) -> chex.Array:
  """Calculates the policy gradient loss with differentiable advantage.

  An optimised version of `rlax.policy_gradient_loss()`.

  Args:
    logits_t: a sequence of unnormalized action preferences (shape: [..., |A|]).
    a_t: a sequence of actions sampled from the preferences `logits_t`.
    adv_t: the observed or estimated advantages from executing actions `a_t`.
    backprop: whether to make the loss differentiable.

  Returns:
    Loss (per step) whose gradient corresponds to a policy gradient update.
  """

  chex.assert_type([logits_t, a_t, adv_t], [float, int, float])

  log_pi_a = rlax.batched_index(jax.nn.log_softmax(logits_t), a_t)
  if backprop:
    loss_per_step = -log_pi_a * adv_t
  else:
    loss_per_step = -log_pi_a * jax.lax.stop_gradient(adv_t)
  return loss_per_step


class MovingAverage:
  """Functions to track EMAs and use them for normalization."""

  def __init__(
      self,
      example_tree: chex.ArrayTree,
      decay: float = 0.999,
      eps: float = 1e-6,
  ):
    """Initialize moving average parameters.

    Args:
      example_tree: An example of the structure later passed to `update_state`.
      decay: The decay of the moments. I.e., the learning rate is `1 - decay`.
      eps: Epsilon used for normalization.
    """
    self._example_tree = example_tree
    self._decay = decay
    self._eps = eps

  def init_state(self) -> types.EmaState:
    zeros = jax.tree_util.tree_map(
        lambda x: jnp.zeros((), jnp.float32), self._example_tree
    )
    return types.EmaState(  # pytype: disable=wrong-arg-types  # jnp-type
        moment1=zeros,
        moment2=zeros,
        decay_product=jnp.ones([], dtype=jnp.float32),
    )

  def update_state(
      self,
      tree_like: chex.ArrayTree,
      state: types.EmaState,
      pmean_axis_name: str | None,
  ) -> types.EmaState:
    """Update moving average stats."""
    squared_tree = jax.tree_util.tree_map(jnp.square, tree_like)

    def _update(
        moment: chex.Array,
        value: chex.Array,
        pmean_axis_name: str | None = None,
    ) -> chex.Array:
      mean = jnp.mean(value)
      # Compute the mean across all learner devices involved in the `pmap`.
      if pmean_axis_name is not None:
        mean = jax.lax.pmean(mean, axis_name=pmean_axis_name)
      return self._decay * moment + (1.0 - self._decay) * mean

    update_fn = functools.partial(_update, pmean_axis_name=pmean_axis_name)
    moment1 = jax.tree_util.tree_map(update_fn, state.moment1, tree_like)
    moment2 = jax.tree_util.tree_map(update_fn, state.moment2, squared_tree)
    return types.EmaState(
        moment1=moment1,
        moment2=moment2,
        decay_product=state.decay_product * self._decay,
    )

  def _compute_moments(
      self, state: types.EmaState
  ) -> tuple[chex.ArrayTree, chex.ArrayTree]:
    """Computes moments, applying 0-debiasing as in the Adam optimizer."""

    # Factor to account for initializing moments with 0s.
    debias = 1.0 / (1 - state.decay_product)

    # Debias mean.
    mean = jax.tree_util.tree_map(lambda m1: m1 * debias, state.moment1)

    # Estimate zero-centered debiased variance; clip negative values to
    # safeguard against numerical errors.
    variance = jax.tree_util.tree_map(
        lambda m2, m: jnp.maximum(0.0, m2 * debias - jnp.square(m)),
        state.moment2,
        mean,
    )
    return mean, variance

  def normalize(
      self,
      value: chex.ArrayTree,
      state: types.EmaState,
      subtract_mean: bool = True,
      root_eps: float = 1e-12,
  ) -> chex.ArrayTree:
    """Normalize by dividing by second moment and subtracting by mean."""

    def _normalize(mean, var, val):
      # Two epsilons, instead of one, are used for numerical stability when
      # backpropagating through the normalization (as in optax.scale_by_adam).
      if subtract_mean:
        return (val - mean) / (jnp.sqrt(var + root_eps) + self._eps)
      else:
        return val / (jnp.sqrt(var + root_eps) + self._eps)

    mean, variance = self._compute_moments(state)
    return jax.tree_util.tree_map(_normalize, mean, variance, value)
