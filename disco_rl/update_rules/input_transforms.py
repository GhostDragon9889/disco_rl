"""Small PyTorch input transforms used by the Disco meta-network."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor) -> torch.Tensor:
  return F.softmax(x, dim=-1)


def stop_grad(x: torch.Tensor) -> torch.Tensor:
  return x.detach()


def sign_log(x: torch.Tensor) -> torch.Tensor:
  return torch.sign(x) * torch.log1p(x.abs())


def masks_to_discounts(x: torch.Tensor) -> torch.Tensor:
  return 1.0 - x.float()
