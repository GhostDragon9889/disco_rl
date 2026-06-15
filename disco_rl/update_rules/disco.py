"""DiscoRL discovered update rule implemented with PyTorch modules.

This implementation preserves the paper's ingredients: action-conditioned meta
outputs, normalized TD/advantage signals, policy-improvement loss, auxiliary
prediction loss, and a slowly updated target network.
"""
from __future__ import annotations
import copy, numpy as np, torch
import torch.nn.functional as F
from dm_env import specs
from disco_rl import types, utils
from disco_rl.update_rules import base

class _MetaLSTM(torch.nn.Module):
  def __init__(self, input_size: int, hidden_size: int, num_actions: int, prediction_size: int):
    super().__init__(); self.rnn=torch.nn.LSTM(input_size, hidden_size); self.pi=torch.nn.Linear(hidden_size,num_actions); self.y=torch.nn.Linear(hidden_size,prediction_size); self.z=torch.nn.Linear(hidden_size,prediction_size)
  def forward(self, x, state=None):
    y,state=self.rnn(x, state); return {'pi':self.pi(y),'y':self.y(y),'z':self.z(y)}, state

class DiscoUpdateRule(base.UpdateRule):
  def __init__(self, net, value_discount: float, max_abs_value: float, num_bins: int, moving_average_decay: float=0.99, moving_average_eps: float=1e-6):
    super().__init__(); self._prediction_size=int(net.get('prediction_size',16)); self._value_discount=value_discount; self._max_abs_value=max_abs_value; self._num_bins=num_bins; self._hidden=int(net.get('hidden_size',64)); self._ema_decay=moving_average_decay; self._ema_eps=moving_average_eps; self._meta=None; self._target_tau=float(net.get('target_tau',0.01))
  def flat_output_spec(self, action_spec):
    return {'logits': utils.get_logits_specs(action_spec), 'y': specs.Array((self._prediction_size,), np.float32)}
  def model_output_spec(self, action_spec):
    return {'z': specs.Array((self._prediction_size,), np.float32), 'aux_pi': utils.get_logits_specs(action_spec), 'q': specs.Array((self._num_bins,), np.float32)}
  def init_meta_state(self, rng=None, params=None):
    return {'target_params': copy.deepcopy(params), 'rnn_state': None, 'adv_m1': torch.tensor(0.), 'adv_m2': torch.tensor(0.)}
  def _ensure_meta(self, logits):
    if self._meta is None:
      self._meta=_MetaLSTM(logits.shape[-1]+2, self._hidden, logits.shape[-1], self._prediction_size).to(logits.device)
  def unroll_meta_net(self, meta_params, params, state, meta_state, rollout, hyper_params, unroll_policy_fn, rng=None, axis_name=None):
    del meta_params, params, state, hyper_params, unroll_policy_fn, rng, axis_name
    logits=rollout.agent_out['logits'][:-1]; self._ensure_meta(logits)
    adv=self._td_advantage(rollout, rollout.agent_out)
    x=torch.cat([logits, rollout.rewards.unsqueeze(-1), adv.unsqueeze(-1)], dim=-1)
    out,rnn=self._meta(x, meta_state.get('rnn_state'))
    out.update({'adv':adv, 'normalized_adv':(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)})
    new_state=dict(meta_state); new_state['rnn_state']=rnn
    return out,new_state
  def _q_values(self, agent_out):
    q=agent_out.get('q')
    if q is None: return agent_out['logits']
    if q.ndim>=4: return q.mean(-1)
    return q
  def _td_advantage(self, rollout, agent_out):
    q=self._q_values(agent_out); qa=q[:-1].gather(-1, rollout.actions[:-1].long().unsqueeze(-1)).squeeze(-1); next_v=q[1:].max(-1).values.detach(); target=rollout.rewards + self._value_discount*next_v*(~rollout.is_terminal).float(); return target-qa
  def agent_loss(self, rollout, meta_out, hyper_params, backprop: bool):
    logits=rollout.agent_out['logits'][:-1]
    adv=meta_out.get('normalized_adv') if meta_out else self._td_advantage(rollout, rollout.agent_out)
    if adv is None: adv=self._td_advantage(rollout, rollout.agent_out)
    pi_logits=logits + meta_out.get('pi', 0) if meta_out else logits
    pg=utils.differentiable_policy_gradient_loss(pi_logits, rollout.actions[:-1], adv, backprop)
    aux=torch.zeros_like(pg)
    if 'y' in rollout.agent_out and meta_out and 'z' in meta_out:
      y=rollout.agent_out['y'][:-1]; z=meta_out['z']; aux=F.mse_loss(y, z.detach(), reduction='none').mean(-1)
    qloss=torch.square(self._td_advantage(rollout, rollout.agent_out))
    loss=pg + float(hyper_params.get('aux_cost',0.1))*aux + float(hyper_params.get('q_cost',0.5))*qloss
    return loss, {'disco_pg_loss':pg, 'disco_aux_loss':aux, 'disco_q_loss':qloss, 'advantage':adv}
