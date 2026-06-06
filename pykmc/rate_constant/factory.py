from .backends import PrefactorBackend
from .rate_constant import RateConstant


def _get_registry() -> dict[str, type[PrefactorBackend]]:
    """Return all available prefactor backends indexed by name.

    Returns
    -------
    dict[str, type[PrefactorBackend]]
        Mapping of backend name to backend class.
    """
    return {cls.name: cls for cls in PrefactorBackend.__subclasses__()}


def create_rate_constant(
    T: float, prefactor_backend_name: str, **kwargs: object
) -> RateConstant:
    """Instantiate a [`RateConstant`][pykmc.rate_constant.RateConstant] from a backend name and temperature.

    Parameters
    ----------
    T : float
        Temperature in K.
    prefactor_backend_name : str
        Name of the prefactor backend, as defined by its ``name`` class attribute.
        Available backends are discovered automatically from `pykmc/rate_constant/backends/`.
    **kwargs
        Arguments forwarded to the backend constructor (e.g. ``config``).

    Returns
    -------
    RateConstant
        Instantiated [`RateConstant`][pykmc.rate_constant.RateConstant] using the selected backend.

    Raises
    ------
    ValueError
        If ``prefactor_backend_name`` does not match any registered backend.

    Examples
    --------
    >>> rc = create_rate_constant(T=300.0, prefactor_backend_name="constant", config=MyConfig(k0=10))
    """
    registry = _get_registry()
    if prefactor_backend_name not in registry:
        raise ValueError(f"Backend '{prefactor_backend_name}' unknown. Available: {list(registry.keys())}")
    return RateConstant(T=T, prefactor_backend=registry[prefactor_backend_name](**kwargs))