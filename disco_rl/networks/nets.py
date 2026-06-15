"""PyTorch agent networks."""
from __future__ import annotations
import numpy as np, torch
from disco_rl import types, utils

class MLPAgentNet(torch.nn.Module):
  def __init__(self, observation_shape, action_spec, out_spec, model_out_spec=None, hidden_sizes=(64,64)):
    super().__init__(); self.out_spec=out_spec; self.model_out_spec=model_out_spec or {}; self.num_actions=utils.get_num_actions_from_spec(action_spec)
    obs_dim=int(np.prod(observation_shape or (1,))); layers=[]; last=obs_dim
    for h in hidden_sizes: layers += [torch.nn.Linear(last,h), torch.nn.Tanh()]; last=h
    self.body=torch.nn.Sequential(*layers); self.heads=torch.nn.ModuleDict()
    for k,s in {**out_spec, **{k: type('S',(),{'shape':(self.num_actions,*v.shape)}) for k,v in self.model_out_spec.items()}}.items():
      self.heads[k]=torch.nn.Linear(last, int(np.prod(s.shape)))
  def _flat_obs(self, obs):
    if isinstance(obs, dict): obs=torch.cat([torch.as_tensor(v).float().reshape(*v.shape[:1], -1) for v in obs.values()], -1)
    else: obs=torch.as_tensor(obs).float().reshape(obs.shape[0], -1)
    return obs
  def forward(self, obs):
    h=self.body(self._flat_obs(obs)); out={}
    for k,head in self.heads.items():
      spec = self.out_spec.get(k) or type('S',(),{'shape':(self.num_actions,*self.model_out_spec[k].shape)})
      out[k]=head(h).reshape(h.shape[0], *spec.shape)
    return out
  def init(self, rng, obs, should_reset=None): return self, None
  def one_step(self, params, state, obs, should_reset=None): return self.forward(obs), state
  def unroll(self, params, state, observations, should_reset=None):
    outs=[]
    for t in range(next(iter(observations.values())).shape[0] if isinstance(observations,dict) else observations.shape[0]):
      o={k:v[t] for k,v in observations.items()} if isinstance(observations,dict) else observations[t]
      outs.append(self.forward(o))
    return utils.tree_stack(outs, axis=0), state

def get_network(name, action_spec, out_spec, model_out_spec=None, **kwargs):
  obs_shape=kwargs.pop('observation_shape', kwargs.pop('input_shape', (1,)))
  hidden_sizes=kwargs.pop('hidden_sizes', kwargs.pop('layers', (64,64)))
  return MLPAgentNet(obs_shape, action_spec, out_spec, model_out_spec, hidden_sizes)
