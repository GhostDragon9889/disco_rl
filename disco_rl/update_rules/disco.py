"""PyTorch implementation of the DiscoRL discovered update rule."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from disco_rl import types, utils
from disco_rl.networks import meta_nets
from disco_rl.update_rules import base
from disco_rl.value_fns import value_utils


class DiscoUpdateRule(base.UpdateRule):
  """Discovered update rule with policy, representation, auxiliary policy and Q losses."""

  def __init__(self, net: dict[str, Any], value_discount: float, max_abs_value: float, num_bins: int, moving_average_decay: float = 0.99, moving_average_eps: float = 1e-6) -> None:
    self.prediction_size = int(net.get("prediction_size", 32))
    self.value_discount = value_discount
    self.max_abs_value = max_abs_value
    self.num_bins = num_bins
    self.adv_ema = utils.MovingAverage(moving_average_decay, moving_average_eps)
    self.td_ema = utils.MovingAverage(moving_average_decay, moving_average_eps)
    self.net_config = dict(net)
    self.meta_net: meta_nets.LSTM | None = None

  def build_meta_net(self, num_actions: int, input_size: int, device: torch.device) -> torch.nn.Module:
    if self.meta_net is None:
      if self.net_config.get("name", "lstm") != "lstm":
        raise ValueError(f"Invalid model network name: {self.net_config.get('name')}")
      self.meta_net = meta_nets.LSTM(input_size=input_size, hidden_size=int(self.net_config.get("hidden_size", 128)), prediction_size=self.prediction_size, num_actions=num_actions).to(device)
    return self.meta_net

  def flat_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    return {"logits": (utils.get_num_actions_from_spec(action_spec),), "y": (self.prediction_size,)}

  def model_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    return {"z": (self.prediction_size,), "aux_pi": (utils.get_num_actions_from_spec(action_spec),), "q": (self.num_bins,)}

  def init_meta_state(self, params: torch.nn.Module) -> types.MetaState:
    return {"rnn_state": None, "adv_ema_state": self.adv_ema.init_state(next(params.parameters()).device), "td_ema_state": self.td_ema.init_state(next(params.parameters()).device), "target_params": deepcopy(params).eval()}

  def _features(self, rollout: types.UpdateRuleInputs, values: types.ValueOuts) -> torch.Tensor:
    actions = rollout.actions[:-1].long()
    logits = rollout.agent_out["logits"][:-1]
    behaviour = (rollout.behaviour_agent_out or rollout.agent_out)["logits"][:-1]
    pi_a = utils.batch_lookup(F.softmax(logits, dim=-1), actions).unsqueeze(-1)
    mu_a = utils.batch_lookup(F.softmax(behaviour, dim=-1), actions).unsqueeze(-1)
    reward = torch.sign(rollout.rewards) * torch.log1p(rollout.rewards.abs())
    done = rollout.is_terminal.float()
    return torch.cat([pi_a, mu_a, reward.unsqueeze(-1), done.unsqueeze(-1), values.value[:-1].unsqueeze(-1), values.adv.unsqueeze(-1), values.normalized_adv.unsqueeze(-1)], dim=-1)

  def unroll_meta_net(self, params: torch.nn.Module, meta_state: types.MetaState, rollout: types.UpdateRuleInputs, hyper_params: types.HyperParams) -> tuple[types.UpdateRuleOuts, types.MetaState]:
    del hyper_params
    target_params = meta_state["target_params"]
    with torch.no_grad():
      target_out = target_params(rollout.observations)
    values, adv_state, td_state = value_utils.get_value_outs(None, rollout.agent_out["q"], None, target_out["q"], rollout, rollout.agent_out["logits"], discount=self.value_discount, categorical_value=True, max_abs_value=self.max_abs_value, drop_last=False, adv_ema_state=meta_state["adv_ema_state"], adv_ema_fn=self.adv_ema, td_ema_state=meta_state["td_ema_state"], td_ema_fn=self.td_ema)
    features = self._features(rollout, values)
    net = self.build_meta_net(rollout.agent_out["logits"].shape[-1], features.shape[-1], features.device)
    meta_out, rnn_state = net(features, meta_state.get("rnn_state"))
    meta_out.update({"q_target": values.q_target, "adv": values.adv, "normalized_adv": values.normalized_adv, "qv_adv": values.qv_adv, "normalized_qv_adv": values.normalized_qv_adv, "q_value": values.q_value, "q_td": values.q_td, "normalized_q_td": values.normalized_q_td, "target_out": target_out})
    new_state = dict(meta_state, rnn_state=rnn_state, adv_ema_state=adv_state, td_ema_state=td_state)
    coeff = float(hyper_params.get("target_params_coeff", 0.995)) if hyper_params else 0.995
    utils.soft_update(new_state["target_params"], params, coeff)
    return meta_out, new_state

  def agent_loss(self, rollout: types.UpdateRuleInputs, meta_out: types.UpdateRuleOuts, hyper_params: types.HyperParams, backprop: bool = True) -> tuple[torch.Tensor, types.UpdateRuleLog]:
    agent_out = {k: v[:-1] for k, v in rollout.agent_out.items()}
    actions = rollout.actions[:-1].long()
    pi_hat, y_hat, z_hat = meta_out["pi"], meta_out["y"], meta_out["z"]
    if not backprop:
      pi_hat, y_hat, z_hat = pi_hat.detach(), y_hat.detach(), z_hat.detach()
    z_a = utils.batch_lookup(agent_out["z"], actions)
    aux_pi_a = utils.batch_lookup(agent_out["aux_pi"], actions)
    aux_target = rollout.agent_out["logits"][1:].detach()
    losses = {
        "pi_loss": utils.categorical_kl(pi_hat, agent_out["logits"]),
        "y_loss": utils.categorical_kl(y_hat, agent_out["y"]),
        "z_loss": utils.categorical_kl(z_hat, z_a),
        "aux_policy_loss": utils.categorical_kl(aux_target, aux_pi_a) * (1.0 - rollout.is_terminal.float()),
    }
    total = float(hyper_params.get("pi_cost", 1.0)) * losses["pi_loss"] + float(hyper_params.get("y_cost", 1.0)) * losses["y_loss"] + float(hyper_params.get("z_cost", 1.0)) * losses["z_loss"] + float(hyper_params.get("aux_policy_cost", 1.0)) * losses["aux_policy_loss"]
    log = {k: v.mean().detach() for k, v in losses.items()}
    return total, log

  def agent_loss_no_meta(self, rollout: types.UpdateRuleInputs, meta_out: types.UpdateRuleOuts, hyper_params: types.HyperParams) -> tuple[torch.Tensor, types.UpdateRuleLog]:
    q_a = utils.batch_lookup(rollout.agent_out["q"], rollout.actions)[:-1]
    loss = value_utils.value_loss_from_td(q_a, meta_out["q_td"].detach()) * float(hyper_params.get("value_cost", 1.0))
    return loss, {"q_loss": loss.mean().detach(), "td": meta_out["q_td"].mean().detach()}


def get_input_option() -> types.MetaNetInputOption:
  return types.MetaNetInputOption(base=(types.TransformConfig("agent_out/logits", ("softmax", "select_a")), types.TransformConfig("rewards", ("sign_log",)), types.TransformConfig("is_terminal", ("masks_to_discounts",))), action_conditional=())
