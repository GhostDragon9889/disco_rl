"""Simple Python batched environment wrapper."""
class BatchedEnv:
  def __init__(self, envs): self.envs=list(envs)
  def reset(self): return [e.reset() for e in self.envs]
  def step(self, actions): return [e.step(a) for e,a in zip(self.envs, actions)]
