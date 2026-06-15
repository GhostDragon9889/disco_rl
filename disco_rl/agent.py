"""PyTorch DiscoRL agent."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, torch
import dm_env
from disco_rl import types
from disco_rl.networks import nets
from disco_rl.update_rules import actor_critic, base as update_rules_base, disco, policy_gradient

@dataclass(frozen=True)
class LearnerState:
  params: torch.nn.Module
  opt_state: torch.optim.Optimizer
  meta_state: types.MetaState

class Agent:
  update_rule: update_rules_base.UpdateRule
  def __init__(self, *, single_observation_spec, single_action_spec, agent_settings, batch_axis_name=None):
    self.settings=agent_settings; self.single_observation_spec=single_observation_spec; self.single_action_spec=single_action_spec; self._batch_axis_name=batch_axis_name
    if agent_settings.update_rule_name=='disco': self.update_rule=disco.DiscoUpdateRule(**agent_settings.update_rule)
    elif agent_settings.update_rule_name=='actor_critic': self.update_rule=actor_critic.ActorCritic(**agent_settings.update_rule)
    elif agent_settings.update_rule_name=='policy_gradient': self.update_rule=policy_gradient.PolicyGradientUpdate(**agent_settings.update_rule)
    else: raise ValueError(f'Unsupported update rule: {agent_settings.update_rule_name}')
    flat=self.update_rule.flat_output_spec(single_action_spec); model=self.update_rule.model_output_spec(single_action_spec)
    obs_shape = next(iter(single_observation_spec.values())).shape if isinstance(single_observation_spec,dict) else single_observation_spec.shape
    net_args=dict(agent_settings.net_settings.get('net_args', {})); net_args.setdefault('observation_shape', obs_shape)
    self._network=nets.get_network(agent_settings.net_settings.name, single_action_spec, flat, model, **net_args)
    self._optimizer=torch.optim.Adam(self._network.parameters(), lr=float(agent_settings.learning_rate))
  def _dummy_obs(self,batch_size):
    def z(s): return torch.zeros((batch_size,*s.shape), dtype=torch.float32)
    return {k:z(v) for k,v in self.single_observation_spec.items()} if isinstance(self.single_observation_spec,dict) else z(self.single_observation_spec)
  def _dummy_should_reset(self,batch_size): return torch.zeros(batch_size,dtype=torch.bool)
  def initial_actor_state(self, rng=None): return None
  def initial_learner_state(self, rng_key=None): return LearnerState(self._network, self._optimizer, self.update_rule.init_meta_state(params=self._network))
  def actor_step(self, actor_params, rng, timestep, actor_state):
    agent_outs,_=self._network.one_step(actor_params, actor_state, timestep.observation, timestep.step_type == dm_env.StepType.LAST)
    actions=torch.distributions.Categorical(logits=agent_outs['logits']).sample()
    at=types.ActorTimestep(timestep.observation, actions, agent_outs, timestep.reward, (timestep.step_type != dm_env.StepType.LAST).float(), actor_state, agent_outs['logits'])
    return at, actor_state
  def unroll_net(self, agent_net_params, agent_net_state, rollout): return self._network.unroll(agent_net_params, agent_net_state, rollout.observations)
  def learner_step(self, rng, rollout, learner_state, agent_net_state, update_rule_params=None, is_meta_training=False):
    agent_out,_=self.unroll_net(learner_state.params, agent_net_state, rollout)
    inputs=types.UpdateRuleInputs(rollout.observations, rollout.actions, rollout.rewards[1:] if rollout.rewards.shape[0]==agent_out['logits'].shape[0] else rollout.rewards, (rollout.discounts[1:]==0) if rollout.discounts.shape[0]==agent_out['logits'].shape[0] else (rollout.discounts==0), agent_out, rollout.agent_outs)
    meta_out, meta_state=self.update_rule.unroll_meta_net(update_rule_params, learner_state.params, agent_net_state, learner_state.meta_state, inputs, getattr(self.settings,'hyper_params',{}), self.unroll_net)
    loss_steps, log=self.update_rule.agent_loss(inputs, meta_out, getattr(self.settings,'hyper_params',{}), backprop=is_meta_training)
    loss=loss_steps.mean(); learner_state.opt_state.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(learner_state.params.parameters(), float(self.settings.max_abs_update)); learner_state.opt_state.step()
    log={k:v.detach().mean() for k,v in log.items()}; log['total_loss']=loss.detach()
    return LearnerState(learner_state.params, learner_state.opt_state, meta_state), agent_net_state, log
