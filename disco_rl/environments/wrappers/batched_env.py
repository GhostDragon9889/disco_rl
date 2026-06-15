"""Numpy batched wrapper for non-jittable environments."""

from __future__ import annotations

import numpy as np
import dm_env


class BatchedSingleStreamEnvironment:
  def __init__(self, env_ctor, batch_size: int, env_settings):
    self._envs = [env_ctor(env_settings) for _ in range(batch_size)]
    self.batch_size = batch_size
  def reset(self, seed: int | None = None):
    del seed
    steps = [env.reset() for env in self._envs]
    return self._stack(steps)
  def step(self, actions):
    steps = [env.step(int(a)) for env, a in zip(self._envs, np.asarray(actions))]
    return self._stack(steps)
  def observation_spec(self):
    return self._envs[0].observation_spec()
  def action_spec(self):
    return self._envs[0].action_spec()
  def _stack(self, steps):
    return dm_env.TimeStep(
      step_type=np.asarray([s.step_type for s in steps]),
      reward=np.asarray([s.reward for s in steps], dtype=np.float32),
      discount=np.asarray([s.discount for s in steps], dtype=np.float32),
      observation=np.stack([s.observation for s in steps]),
    )
