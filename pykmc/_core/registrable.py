"""Shared plumbing for the pluggable registry pattern.

This module provides :class:`Registrable`, the domain-free base class that
powers auto-registration and lookup across pyKMC for algorithm strategies,
engine backends, and any other pluggable hierarchy.

Concrete implementations live in their respective modules. See the
`architecture documentation <../docs/dev_strategy_pattern.md>`_ for the strategy
design usecase and :func:`pykmc._core.discovery.autodiscover` for how implementation
modules are loaded.
"""

from __future__ import annotations

import inspect
from abc import ABC
from typing import ClassVar


class Registrable(ABC):
    """Abstract base for any pluggable, auto-registrable component in pyKMC.

    Provides the registry mechanism shared by all pluggable hierarchies
    (algorithm strategies, engine backends, etc.). Each hierarchy defines its
    own root by subclassing with ``root=True``, concrete implementations then
    subclass the root, declare a unique ``name``, and are registered
    automatically at class-definition time.

    Attributes
    ----------
    name : str
        Unique identifier within a root's registry.
    _registry : dict[str, type[Registrable]]
        Maps names to classes. Fresh per root.
    _root : type[Registrable]
        The root that owns the registry.
    """

    name: ClassVar[str]
    _registry: ClassVar[dict[str, type["Registrable"]]]
    _root: ClassVar[type["Registrable"]]

    def __init_subclass__(cls, root: bool = False, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        if root:
            cls._registry = {}
            cls._root = cls
            return

        if inspect.isabstract(cls):
            if "name" in cls.__dict__:
                unimplemented = sorted(
                    attr
                    for attr in dir(cls)
                    if getattr(getattr(cls, attr, None), "__isabstractmethod__", False)
                )
                raise TypeError(
                    f"{cls.__name__} defines 'name' but is still abstract "
                    f"(unimplemented: {unimplemented})"
                )
            return

        name = cls.__dict__.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError(f"{cls.__name__} must define a non-empty 'name' str")

        registry = cls._root._registry
        if name in registry:
            raise RuntimeError(
                f"Name collision '{name}': {registry[name].__name__} vs {cls.__name__}"
            )
        registry[name] = cls

    @classmethod
    def create(cls, name: str, **kwargs) -> "Registrable":
        """Instantiate the registered component identified by ``name``.

        Parameters
        ----------
        name : str
            Name of a component registered under this root.
        **kwargs
            Forwarded to the selected component's constructor.

        Raises
        ------
        TypeError
            If called on ``Registrable`` directly rather than on a root.
        ValueError
            If ``name`` is not registered under this root.
        """
        if not hasattr(cls, "_registry"):
            raise TypeError(
                "create() must be called on a root class, not on Registrable itself"
            )
        if name not in cls._registry:
            raise ValueError(f"'{name}' unknown. Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)
