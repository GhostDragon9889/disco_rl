"""Environment protocol definitions."""

from __future__ import annotations

from typing import Protocol, Any


class Environment(Protocol):
  def reset(self, seed: int | None = None) -> Any: ...
  def step(self, actions: Any) -> Any: ...
