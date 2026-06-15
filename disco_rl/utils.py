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

"""PyTorch utility helpers."""

import random

import numpy as np
import torch


def select_device(prefer_cuda: bool = True) -> torch.device:
  """Returns a CUDA device when available, otherwise CPU."""
  if prefer_cuda and torch.cuda.is_available():
    return torch.device('cuda')
  return torch.device('cpu')


def seed_all(seed: int) -> None:
  """Seeds Python, NumPy, CPU Torch, and all visible CUDA devices."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
