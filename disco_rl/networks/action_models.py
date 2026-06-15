"""Action-conditioned PyTorch heads are implemented directly in networks.nets."""

from __future__ import annotations

from typing import Any


def get_action_model(*_: Any, **__: Any):
  raise NotImplementedError("Action models are folded into disco_rl.networks.nets.MLP in the PyTorch refactor.")
