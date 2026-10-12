"""Small shared protocols that avoid runtime import cycles."""

from typing import Protocol


class BudgetView(Protocol):
    max_models: int
    max_epochs: int
    seed: int
