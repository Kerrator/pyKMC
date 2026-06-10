from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import Future


class PrefactorBackend(ABC):
    """Abstract base class for prefactor backends.

    Subclasses must define:

    - a ``name`` class attribute (``str``) : unique key used by the factory registry.
    - a ``compute(**kwargs) -> float`` method : returns the prefactor in ps^-1.

    A ``TypeError`` is raised at class definition time if ``name`` is missing.

    Backends may optionally hold an injected engine ``manager`` (a live object,
    never part of the pydantic config); per-event batch prefactor computation
    goes through :meth:`compute_prefactors_batch`.
    """
    name: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define a 'name' class attribute")

    @abstractmethod
    def compute(self, **kwargs: object) -> float:
        """Compute the rate constant prefactor."""
        pass

    def compute_prefactors_batch(
        self, payloads: "list[dict[str, object]]", config: object
    ) -> "list[Future]":
        """Default (constant-style) batch: immediately-resolved, no per-event nu0.

        Uniform contract: every returned future resolves to an
        ``EventPrefactors``; callers read ``.nu0_forward`` / ``.nu0_backward``
        (both ``None`` here -> the caller keeps its ``k0``-based values).
        Backends that compute per-event prefactors (htst/rpa) override this to
        fan the payloads out over the engine manager.

        Parameters
        ----------
        payloads : list[dict]
            One dict per event: ``central_atom_idx``, ``min1_positions``,
            ``saddle_positions``, ``min2_positions``, ``types``, ``cell``.
        config : object
            The FULL pykmc ``Config`` (the engine op reads ``config.rateconstant``);
            unused by this default.

        Returns
        -------
        list[Future]
            One resolved future per payload.

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