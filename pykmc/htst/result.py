"""Error codes for HTST prefactors."""

from enum import auto

from pykmc._core.result import ErrorCode


class HTSTError(ErrorCode):
    """Reasons an event gets no nu0 and falls back to `k0`."""

    ENGINE_FAILURE = auto()
    NO_FROZEN_BOUNDARY = auto()
    SADDLE_NOT_FIRST_ORDER = auto()
    UNSTABLE_MINIMUM = auto()
    NONFINITE_PREFACTOR = auto()
    FROZEN_OVERLAP = auto()
    WRAPPED_OPEN_AXIS = auto()
