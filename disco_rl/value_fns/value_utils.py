"""PyTorch value-target utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from disco_rl import types, utils

DEFAULT_DISCOUNT = 0.995
DEFAULT_TD_LAMBDA = 0.95


def categorical_values(logits: torch.Tensor, max_abs_value: float) -> torch.Tensor:
  bins = torch.linspace(-max_abs_value, max_abs_value, logits.shape[-1], device=logits.device)
  return (F.softmax(logits, dim=-1) * bins).sum(dim=-1)


def lambda_returns(rewards: torch.Tensor, discounts: torch.Tensor, values: torch.Tensor, gamma: float, lambda_: float) -> torch.Tensor:
  returns = torch.empty_like(rewards)
  g = values[-1]
  for t in range(rewards.shape[0] - 1, -1, -1):
    bootstrap = (1.0 - lambda_) * values[t + 1] + lambda_ * g
    g = rewards[t] + gamma * discounts[t] * bootstrap
    returns[t] = g
  return returns


def get_value_outs(
    value_net_out: torch.Tensor | None,
    q_net_out: torch.Tensor | None,
    target_value_net_out: torch.Tensor | None,
    target_q_net_out: torch.Tensor | None,
    rollout: types.ActorRollout | types.UpdateRuleInputs,
    pi_logits: torch.Tensor,
    discount: float = DEFAULT_DISCOUNT,
    lambda_: float = DEFAULT_TD_LAMBDA,
    nonlinear_transform: bool = False,
    categorical_value: bool = False,
    max_abs_value: float | None = None,
    drop_last: bool = True,
    adv_ema_state: types.EmaState | None = None,
    adv_ema_fn: utils.MovingAverage | None = None,
    td_ema_state: types.EmaState | None = None,
    td_ema_fn: utils.MovingAverage | None = None,
    axis_name: str | None = None,
) -> tuple[types.ValueOuts, types.EmaState | None, types.EmaState | None]:
  del nonlinear_transform, axis_name
  rewards = rollout.rewards[:-1] if drop_last else rollout.rewards
  actions = rollout.actions[:-1]
  env_discounts = rollout.discounts[:-1] if isinstance(rollout, types.ActorRollout) else (1.0 - rollout.is_terminal.float())
  if drop_last and not isinstance(rollout, types.ActorRollout):
    env_discounts = env_discounts[:-1]

  if categorical_value:
    if max_abs_value is None:
      raise ValueError("max_abs_value is required for categorical values")
    q_values = categorical_values(q_net_out, max_abs_value) if q_net_out is not None else None
    target_q_values = categorical_values(target_q_net_out if target_q_net_out is not None else q_net_out, max_abs_value) if q_net_out is not None else None
    values = categorical_values(value_net_out, max_abs_value) if value_net_out is not None else None
  else:
    q_values = q_net_out.squeeze(-1) if q_net_out is not None and q_net_out.shape[-1] == 1 else q_net_out
    target_q_values = target_q_net_out if target_q_net_out is not None else q_values
    values = value_net_out.squeeze(-1) if value_net_out is not None else None

  if q_values is not None:
    policy = F.softmax(pi_logits, dim=-1)
    state_values = (policy * q_values).sum(dim=-1)
    target_state_values = (policy * target_q_values).sum(dim=-1)
    targets = lambda_returns(rewards, env_discounts, target_state_values, float(discount), float(lambda_))
    q_a = utils.batch_lookup(q_values, actions)
    q_td = targets - q_a
    adv = targets - state_values[:-1]
    qv_adv = target_q_values[:-1] - state_values[:-1].unsqueeze(-1)
    rho = torch.ones_like(rewards)
    value = state_values
    target_value = target_state_values
    q_target = targets.unsqueeze(-1).expand_as(q_values[:-1])
    q_value = q_values[:-1]
    target_q_value = target_q_values[:-1]
    td = adv
  elif values is not None:
    targets = lambda_returns(rewards, env_discounts, values, float(discount), float(lambda_))
    adv = targets - values[:-1]
    qv_adv = adv.unsqueeze(-1)
    q_td = adv
    rho = torch.ones_like(rewards)
    value = values
    target_value = values
    q_target = targets
    q_value = values[:-1]
    target_q_value = values[:-1]
    td = adv
  else:
    raise ValueError("value_net_out or q_net_out must be provided")

  new_adv = adv_ema_fn.update_state(adv, adv_ema_state) if adv_ema_fn and adv_ema_state else None
  normalized_adv = adv_ema_fn.normalize(adv, new_adv) if adv_ema_fn and new_adv else adv
  new_td = td_ema_fn.update_state(q_td, td_ema_state) if td_ema_fn and td_ema_state else None
  normalized_td = td_ema_fn.normalize(q_td, new_td) if td_ema_fn and new_td else q_td
  normalized_qv_adv = adv_ema_fn.normalize(qv_adv, new_adv) if adv_ema_fn and new_adv else qv_adv
  return types.ValueOuts(value, target_value, rho, adv, normalized_adv, td, normalized_td, targets, qv_adv, normalized_qv_adv, q_target, q_value, target_q_value, q_td, normalized_td), new_adv, new_td


def value_loss_from_td(value_net_out: torch.Tensor, td: torch.Tensor, **_: object) -> torch.Tensor:
  pred = utils.batch_lookup(value_net_out, torch.zeros(td.shape, dtype=torch.long, device=td.device)) if value_net_out.ndim > td.ndim else value_net_out
  return 0.5 * (pred.float() - td.float()).pow(2)
