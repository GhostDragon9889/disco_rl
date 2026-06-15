"""Single-stream environment wrapper."""

class SingleStreamEnv:
  def __init__(self, env):
    self.env = env
  def reset(self, seed=None):
    del seed
    return self.env.reset()
  def step(self, action):
    return self.env.step(action)
  def observation_spec(self):
    return self.env.observation_spec()
  def action_spec(self):
    return self.env.action_spec()
