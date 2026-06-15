# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""PyTorch backend for DiscoRL experiments."""

from disco_rl.torch_backend.agent import ActorCriticAgent
from disco_rl.torch_backend.environments import CatchConfig
from disco_rl.torch_backend.environments import TorchCatchEnv
from disco_rl.torch_backend.networks import MLPActorCritic
from disco_rl.torch_backend.types import Rollout
from disco_rl.torch_backend.types import TimeStep

__all__ = [
    'ActorCriticAgent',
    'CatchConfig',
    'MLPActorCritic',
    'Rollout',
    'TimeStep',
    'TorchCatchEnv',
]
