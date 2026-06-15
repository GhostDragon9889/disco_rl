"""REINFORCE / policy-gradient update rule in PyTorch."""
from __future__ import annotations
import torch
from disco_rl import types, utils
from disco_rl.update_rules import base

class PolicyGradientUpdate(base.UpdateRule):
  def __init__(self, normalize_returns: bool = True, discount: float = 0.99):
    super().__init__(); self.normalize_returns=normalize_returns; self.discount=discount
  def flat_output_spec(self, action_spec):
    return {'logits': utils.get_logits_specs(action_spec)}
  def _returns(self, rewards, terminals):
    out=[]; g=torch.zeros_like(rewards[-1])
    for r, done in zip(reversed(rewards), reversed(terminals)):
      g = r + self.discount * g * (~done).float(); out.append(g)
    ret=torch.stack(list(reversed(out)))
    if self.normalize_returns: ret=(ret-ret.mean())/(ret.std(unbiased=False)+1e-8)
    return ret
  def agent_loss(self, rollout, meta_out, hyper_params, backprop: bool):
    del meta_out, hyper_params
    adv=self._returns(rollout.rewards, rollout.is_terminal)
    loss=utils.differentiable_policy_gradient_loss(rollout.agent_out['logits'][:-1], rollout.actions[:-1], adv, backprop)
    return loss, {'pg_loss': loss, 'advantage': adv}
