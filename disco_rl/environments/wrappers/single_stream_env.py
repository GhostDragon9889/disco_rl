class SingleStreamEnv:
  def __init__(self, env): self.env=env
  def reset(self): return self.env.reset()
  def step(self, action): return self.env.step(action)
