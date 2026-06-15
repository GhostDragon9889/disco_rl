# DiscoRL PyTorch CUDA 12.9 Implementation

This branch keeps a PyTorch-only reinforcement-learning implementation. The current implementation
provides a CUDA-capable batched `Catch` environment, an MLP actor-critic network,
and an on-policy actor-critic/GAE training loop.

## Installation

CUDA 12.9 PyTorch wheels are distributed from the PyTorch nightly `cu129` index.
Install the project and CUDA 12.9 runtime dependencies with:

```bash
python -m venv disco_rl_venv
source disco_rl_venv/bin/activate
pip install -e .
pip install --pre -r requirements.txt
```

If CUDA is unavailable, the same code runs on CPU for development and tests.

## Quick smoke test

```python
from disco_rl import ActorCriticAgent, CatchConfig, MLPActorCritic, TorchCatchEnv
from disco_rl import seed_all, select_device

seed_all(1)
device = select_device()
env = TorchCatchEnv(CatchConfig(batch_size=8, device=device))
network = MLPActorCritic(env.observation_shape, env.num_actions).to(device)
agent = ActorCriticAgent(network)
rollout = agent.collect_rollout(env, rollout_length=16)
metrics = agent.update(rollout)
print(metrics)
```

## Package layout

* `disco_rl.agent` exposes the PyTorch `ActorCriticAgent` and config.
* `disco_rl.torch_backend.environments` implements batched Torch `Catch`.
* `disco_rl.torch_backend.networks` implements `MLPActorCritic`.
* `disco_rl.torch_backend.types` defines Torch `TimeStep` and `Rollout` data
  containers.
* `disco_rl.utils` contains Torch-oriented device and seeding helpers.

## Citation

Please cite the original Nature paper:

```bibtex
@Article{DiscoRL2025,
  author  = {Oh, Junhyuk and Farquhar, Greg and Kemaev, Iurii and Calian, Dan A. and Hessel, Matteo and Zintgraf, Luisa and Singh, Satinder and van Hasselt, Hado and Silver, David},
  journal = {Nature},
  title   = {Discovering State-of-the-art Reinforcement Learning Algorithms},
  year    = {2025},
  doi     = {10.1038/s41586-025-09761-x}
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0).
This is not an official Google product.
