"""Actor-critic update rule in PyTorch."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from dm_env import specs
from disco_rl import utils
from disco_rl.update_rules.policy_gradient import PolicyGradientUpdate

class ActorCritic(PolicyGradientUpdate):
  def __init__(self, discount: float = 0.99, value_cost: float = 0.5, normalize_returns: bool = False):
    super().__init__(normalize_returns=normalize_returns, discount=discount); self.value_cost=value_cost
  def flat_output_spec(self, action_spec):
    return {'logits': utils.get_logits_specs(action_spec), 'value': specs.Array((), np.float32)}
  def agent_loss(self, rollout, meta_out, hyper_params, backprop: bool):
    del meta_out
    values=rollout.agent_out['value'].squeeze(-1) if rollout.agent_out['value'].ndim==3 else rollout.agent_out['value']
    returns=self._returns(rollout.rewards, rollout.is_terminal)
    adv=returns - values[:-1]
    pg=utils.differentiable_policy_gradient_loss(rollout.agent_out['logits'][:-1], rollout.actions[:-1], adv, backprop)
    vloss=F.mse_loss(values[:-1], returns.detach(), reduction='none')
    loss=pg + float(hyper_params.get('value_cost', self.value_cost))*vloss
    return loss, {'pg_loss': pg, 'value_loss': vloss, 'advantage': adv}
