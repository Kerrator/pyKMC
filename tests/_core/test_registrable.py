"""Unit tests for the Registrable mechanism (pykmc._core.registrable).

Tests the registry plumbing in isolation: auto-registration, validation,
collision detection, and registry isolation. See test_strategy.py for an
end-to-end example of the facade/strategy pattern built on top of Registrable.
"""

import pytest
from abc import abstractmethod
from pykmc._core import Registrable


class FakeRoot(Registrable, root=True):
    @abstractmethod
    def run(self) -> int: ...


class Alpha(FakeRoot):
    name = "alpha"

    def run(self):
        return 1


class Beta(FakeRoot):
    name = "beta"

    def run(self):
        return 2


class TestRegistration:
    def test_concrete_classes_are_registered(self):
        assert "alpha" in FakeRoot._registry
        assert "beta" in FakeRoot._registry
        assert FakeRoot._registry["alpha"] is Alpha
        assert FakeRoot._registry["beta"] is Beta

    def test_create_returns_correct_type(self):
        assert isinstance(FakeRoot.create("alpha"), Alpha)
        assert isinstance(FakeRoot.create("beta"), Beta)

    def test_missing_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty 'name'"):

            class NoName(FakeRoot):
                def run(self):
                    return 0

    def test_empty_name_raises_type_error(self):
        with pytest.raises(TypeError, match="non-empty 'name'"):

            class EmptyName(FakeRoot):
                name = ""

                def run(self):
                    return 0

    def test_name_collision_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Name collision"):

            class DuplicateAlpha(FakeRoot):
                name = "alpha"

                def run(self):
                    return 99

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown"):
            FakeRoot.create("gamma")

    def test_error_message_lists_available_names(self):
        with pytest.raises(ValueError, match="alpha"):
            FakeRoot.create("nonexistent")

    def test_create_on_registrable_base_raises_type_error(self):
        """Registrable itself has no _registry, create() must reject it."""
        with pytest.raises(TypeError):
            Registrable.create("alpha")

    def test_abstract_with_name_raises_type_error(self):
        """Declaring name on an abstract class (e.g. typo in override) is caught early."""
        with pytest.raises(TypeError, match="still abstract"):

            class TypoRun(FakeRoot):
                name = "typo"

                def runn(self):  # typo: run → runn
                    return 0

    def test_inherited_name_not_accepted(self):
        """A subclass must declare its own name, inheriting a parent's is rejected."""
        with pytest.raises(TypeError, match="non-empty 'name'"):

            class Child(Alpha):
                def run(self):
                    return -1

    def test_intermediate_abstract_base_not_registered(self):
        class IntermediateFake(FakeRoot):
            def _helper(self): ...

        class Gamma(IntermediateFake):
            name = "gamma"

            def run(self):
                return 3

        assert "gamma" in FakeRoot._registry
        assert FakeRoot._registry["gamma"] is Gamma
        assert isinstance(FakeRoot.create("gamma"), Gamma)

    def test_registries_are_isolated_between_roots(self):
        class OtherRoot(Registrable, root=True):
            @abstractmethod
            def run(self): ...

        class OtherAlpha(OtherRoot):
            name = "alpha"

            def run(self):
                return 42

        assert OtherRoot._registry is not FakeRoot._registry
        assert OtherRoot._registry["alpha"] is OtherAlpha
        assert FakeRoot._registry["alpha"] is Alpha
