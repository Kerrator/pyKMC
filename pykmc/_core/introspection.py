"""Static attribute inspection shared by the engine and manager registries.

Both ``pykmc.engine.base`` (extension discovery) and ``pykmc.manager.worker``
(operation registries) must decide whether a statically looked-up class
attribute denotes a callable operation without evaluating it. Keeping the
single rule here, in the domain-free core, avoids an engine -> manager import.
"""

from __future__ import annotations

import types
from typing import Any

# Routine types that are safe to bind with ``getattr``: binding applies the
# descriptor protocol but computes nothing. Deliberately narrower than
# ``inspect.isroutine``, which also accepts every ``__get__``-only descriptor
# (``functools.cached_property``, ``partialmethod``, ``singledispatchmethod``,
# custom lazy descriptors) and would let discovery evaluate them.
_ROUTINE_TYPES: tuple[type, ...] = (
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
    types.WrapperDescriptorType,
    types.MethodWrapperType,
)


def is_static_callable(raw: Any) -> bool:
    """Return True if a statically looked-up attribute denotes a callable operation.

    ``raw`` is the object returned by :func:`inspect.getattr_static`, i.e. the
    attribute exactly as stored on the class or instance, with no descriptor
    protocol applied. Only an explicit allow-list qualifies: plain and builtin
    functions, bound methods, the C-level method/wrapper descriptors, and
    ``classmethod``/``staticmethod`` wrappers, plus callable objects whose type
    has no ``__get__``. Every other descriptor (``property``,
    ``functools.cached_property``, ``partialmethod``, custom ``__get__``
    objects) computes a value on access and is rejected, so it is never
    evaluated during discovery. ``inspect.isroutine`` is not used because it
    accepts any ``__get__``-only descriptor.

    Parameters
    ----------
    raw : Any
        Attribute as returned by ``inspect.getattr_static``.

    Returns
    -------
    bool

    """
    if isinstance(raw, (classmethod, staticmethod)):
        return True
    if isinstance(raw, _ROUTINE_TYPES):
        return True
    return callable(raw) and not hasattr(type(raw), "__get__")
