"""Facade/strategy pattern example built on Registrable (pykmc._core.registrable).

This file is both a test and a recipe.  It shows the complete facade/strategy
pattern as used throughout pyKMC:

- A facade holds data and delegates to a strategy.
- Algorithm hierarchies use the ``XxxStrategy`` naming convention and inherit
  from ``Registrable`` with ``root=True``.  The "Strategy" suffix is a naming
  convention, not a base class.
- Concrete strategies declare a unique ``name`` and register themselves at
  import time.

For tests of the underlying registry mechanism see test_registrable.py.
"""

import pytest
from abc import abstractmethod
from pykmc._core import Registrable


# Strategies
class BaseOperationStrategy(Registrable, root=True):
    @abstractmethod
    def compute(self, a: float, b: float) -> float: ...


class Addition(BaseOperationStrategy):
    name = "addition"

    def compute(self, a, b):
        return a + b


class Multiplication(BaseOperationStrategy):
    name = "multiplication"

    def compute(self, a, b):
        return a * b


class Division(BaseOperationStrategy):
    name = "division"

    def compute(self, a, b):
        return a / b


# Facade
class ComputeOperation:
    """Stable user-facing API, delegates computation to the active strategy."""

    def __init__(self, strategy: BaseOperationStrategy) -> None:
        self._strategy = strategy

    def compute(self, a: float, b: float) -> float:
        return self._strategy.compute(a, b)

    @classmethod
    def create(cls, strategy_name: str) -> "ComputeOperation":
        return cls(strategy=BaseOperationStrategy.create(strategy_name))


# Tests

class TestFacadePattern:
    def test_addition(self):
        assert ComputeOperation.create("addition").compute(3, 2) == 5

    def test_multiplication(self):
        assert ComputeOperation.create("multiplication").compute(3, 2) == 6

    def test_division(self):
        assert ComputeOperation.create("division").compute(6, 2) == 3.0

    def test_unknown_strategy_raises_value_error(self):
        with pytest.raises(ValueError):
            ComputeOperation.create("modulo")

    def test_all_strategies_auto_registered(self):
        assert set(BaseOperationStrategy._registry) >= {
            "addition",
            "multiplication",
            "division",
        }
