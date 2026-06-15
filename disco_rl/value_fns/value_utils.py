"""PyTorch value-target utilities."""
from __future__ import annotations
import torch
from disco_rl import types

def discounted_returns(rewards, terminals, discount=0.99):
  g=torch.zeros_like(rewards[-1]); outs=[]
  for r,d in zip(reversed(rewards), reversed(terminals)):
    g=r+discount*g*(~d).float(); outs.append(g)
  return torch.stack(list(reversed(outs)))

def get_value_outs(value_net_out=None, target_value_net_out=None, q_net_out=None, target_q_net_out=None, rollout=None, pi_logits=None, discount=0.99, lambda_=1.0, **kwargs):
  del target_value_net_out, target_q_net_out, pi_logits, lambda_, kwargs
  value = value_net_out if value_net_out is not None else q_net_out.mean(-1).mean(-1) if q_net_out is not None and q_net_out.ndim>=4 else torch.zeros_like(rollout.rewards)
  if value.shape[0] == rollout.rewards.shape[0]: boot=torch.cat([value, value[-1:]],0)
  else: boot=value
  target=discounted_returns(rollout.rewards, rollout.is_terminal, float(discount))
  adv=target - boot[:-1]
  norm=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
  q=q_net_out if q_net_out is not None else boot
  return types.ValueOuts(boot, boot, torch.ones_like(rollout.rewards), adv, norm, adv, norm, target, adv, norm, q, q, q, adv, norm), None, None
