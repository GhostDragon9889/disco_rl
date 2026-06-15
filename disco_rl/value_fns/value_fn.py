"""Small PyTorch value-function trainer."""
from __future__ import annotations
import torch
from disco_rl.value_fns import value_utils

class ValueFn:
  def __init__(self, config, axis_name=None): self.config=config; self.axis_name=axis_name
  def init_state(self, rng, dummy_observation): return None
  def learner_step(self, value_state, rollout, value_net_out):
    outs,_,_=value_utils.get_value_outs(value_net_out=value_net_out, rollout=rollout, discount=self.config.discount_factor)
    return value_state, {'value_loss': torch.square(outs.td).mean()}, outs
