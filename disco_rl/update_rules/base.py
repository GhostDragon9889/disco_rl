"""PyTorch base classes for update rules."""
from __future__ import annotations
import numpy as np
import torch
from dm_env import specs as dm_env_specs
from disco_rl import types, utils

ArraySpec = types.ArraySpec

def get_agent_out_spec(action_spec: types.ActionSpec, flat_out_spec: types.Specs, model_out_spec: types.Specs) -> types.Specs:
  if set(flat_out_spec).intersection(model_out_spec):
    raise ValueError('Keys overlap between flat_out_spec and model_out_spec.')
  n = utils.get_num_actions_from_spec(action_spec)
  out = dict(flat_out_spec)
  for k, v in model_out_spec.items():
    out[k] = dm_env_specs.Array((n, *v.shape), v.dtype)
  return out

class UpdateRule(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def _get_dummy_input(self, include_behaviour_out=True, include_value_out=False, include_agent_adv=False):
    b,t=1,2; action_spec=dm_env_specs.BoundedArray(shape=(), dtype=np.int32, minimum=0, maximum=3)
    agent_out = utils.tree_map(lambda s: torch.zeros((t+1,b,*s.shape), dtype=torch.float32), self.agent_output_spec(action_spec))
    dummy = types.UpdateRuleInputs(torch.zeros(t+1,b), torch.zeros(t+1,b,dtype=torch.long), torch.zeros(t,b), torch.ones(t,b,dtype=torch.bool), agent_out)
    if include_behaviour_out: dummy.behaviour_agent_out = dict(agent_out)
    dummy.extra_from_rule = {'target_out': agent_out}
    return dummy
  def init_params(self, rng=None):
    return self.state_dict(), None
  def flat_output_spec(self, action_spec): return {}
  def model_output_spec(self, action_spec): return {}
  def agent_output_spec(self, action_spec): return get_agent_out_spec(action_spec, self.flat_output_spec(action_spec), self.model_output_spec(action_spec))
  def init_meta_state(self, rng=None, params=None): return {}
  def unroll_meta_net(self, meta_params, params, state, meta_state, rollout, hyper_params, unroll_policy_fn, rng=None, axis_name=None):
    return {}, meta_state
  def agent_loss(self, rollout, meta_out, hyper_params, backprop: bool):
    raise NotImplementedError
  def agent_loss_no_meta(self, rollout, meta_out, hyper_params):
    logits=rollout.agent_out['logits'][:-1]
    entropy=torch.distributions.Categorical(logits=logits).entropy()
    coef=float(hyper_params.get('entropy_cost', 0.0))
    return -coef*entropy, {'entropy': entropy}
