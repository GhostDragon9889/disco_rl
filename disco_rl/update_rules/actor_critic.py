"""Actor-critic update rule."""

import torch
import torch.nn.functional as F

from disco_rl import types
from disco_rl.update_rules import base


class ActorCritic(base.UpdateRule):
  def flat_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    return super().flat_output_spec(action_spec) | {"value": (1,)}

  def init_meta_state(self, params: torch.nn.Module) -> types.MetaState:
    del params
    return {}

  def agent_loss(self, rollout: types.UpdateRuleInputs, meta_out, hyper_params: types.HyperParams, backprop: bool = True):
    del meta_out, backprop
    logits = rollout.agent_out["logits"][:-1]
    values = rollout.agent_out.get("value", torch.zeros_like(rollout.rewards).unsqueeze(-1))[:-1].squeeze(-1)
    returns = torch.flip(torch.cumsum(torch.flip(rollout.rewards, [0]), dim=0), [0])
    adv = returns - values
    logp = F.log_softmax(logits, dim=-1).gather(-1, rollout.actions[:-1].long().unsqueeze(-1)).squeeze(-1)
    policy_loss = -logp * adv.detach()
    value_loss = 0.5 * adv.pow(2)
    loss = policy_loss + float(hyper_params.get("value_cost", 0.5)) * value_loss
    return loss, {"policy_loss": policy_loss.mean().detach(), "value_loss": value_loss.mean().detach()}

  def agent_loss_no_meta(self, rollout, meta_out, hyper_params):
    return torch.zeros_like(rollout.rewards), {}
