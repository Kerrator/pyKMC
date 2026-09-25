from .base import PrefactorBackend
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from concurrent.futures import Future


class ConstantBackendConfig(Protocol):
    """Configuration interface for [`ConstantBackend`][pykmc.rate_constant.backends.constant.ConstantBackend].

    Attributes
    ----------
    k0 : float
        Constant prefactor value in ps^-1.
    """

    k0: float


class ConstantBackend(PrefactorBackend):
    """Backend using a constant prefactor.

    Parameters
    ----------
    config : ConstantBackendConfig
        Configuration object exposing a ``k0`` attribute.
        Compatible with the Pydantic ``RateConstantConfig`` used by pykmc.
    """

    name = "constant"

    def __init__(self, config: ConstantBackendConfig, manager: object = None) -> None:
        self.config = config  # ``manager`` is accepted for ctor uniformity; unused

    def compute(self, **kwargs) -> float:
        """Return the constant prefactor.

        Returns
        -------
        float
            Constant prefactor in ps^-1.
        """
        return self.config.k0

    def compute_prefactors_batch(
        self, payloads: "list[dict[str, object]]", config: object
    ) -> "list[Future]":
        """Constant-style batch: immediately-resolved futures, no per-event nu0.

        Uniform contract with the htst/rpa backends: every returned future
        resolves to an ``EventPrefactors``; callers read ``.nu0_forward`` /
        ``.nu0_backward`` (both ``None`` here -> the caller keeps its
        ``k0``-based values). ``config`` (the full pykmc ``Config``) is unused.
        """
        from concurrent.futures import Future

        from pykmc.rate_constant.prefactor import EventPrefactors

        futures: "list[Future]" = []
        for _ in payloads:
            f: "Future" = Future()
            f.set_result(
                EventPrefactors(
                    nu0_forward=None,
                    nu0_backward=None,
                    n_free=0,
                    n_neg_saddle=0,
                    ok_forward=False,
                    ok_backward=False,
                    reason="constant style: no per-event nu0",
                )
            )
            futures.append(f)
        return futures
