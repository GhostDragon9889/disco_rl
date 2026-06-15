"""Policy-gradient update rule."""

import torch
import torch.nn.functional as F

from disco_rl import types
from disco_rl.update_rules import base


class PolicyGradientUpdate(base.UpdateRule):
  def init_meta_state(self, params: torch.nn.Module) -> types.MetaState:
    del params
    return {}

  def agent_loss(self, rollout: types.UpdateRuleInputs, meta_out, hyper_params: types.HyperParams, backprop: bool = True):
    del meta_out, backprop
    logits = rollout.agent_out["logits"][:-1]
    logp = F.log_softmax(logits, dim=-1).gather(-1, rollout.actions[:-1].long().unsqueeze(-1)).squeeze(-1)
    returns = torch.flip(torch.cumsum(torch.flip(rollout.rewards, [0]), dim=0), [0])
    loss = -logp * returns.detach() * float(hyper_params.get("pg_cost", 1.0))
    return loss, {"pg_loss": loss.mean().detach()}

  def agent_loss_no_meta(self, rollout, meta_out, hyper_params):
    return torch.zeros_like(rollout.rewards), {}
