"""Base update-rule interfaces for PyTorch DiscoRL."""

from __future__ import annotations

from disco_rl import types, utils


def get_agent_out_spec(action_spec: types.ActionSpec, flat_out_spec: types.Specs, model_out_spec: types.Specs) -> types.Specs:
  num_actions = utils.get_num_actions_from_spec(action_spec)
  out = dict(flat_out_spec)
  for key, shape in model_out_spec.items():
    out[key] = (num_actions, *shape)
  return out


class UpdateRule:
  def flat_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    return {"logits": (utils.get_num_actions_from_spec(action_spec),)}

  def model_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    del action_spec
    return {}

  def agent_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    return get_agent_out_spec(action_spec, self.flat_output_spec(action_spec), self.model_output_spec(action_spec))
