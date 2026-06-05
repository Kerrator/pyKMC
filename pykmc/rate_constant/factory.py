from .backends import PrefactorBackend
from .rate_constant import RateConstant


def _get_registry() -> dict[str, type[PrefactorBackend]]:
    return {cls.name: cls for cls in PrefactorBackend.__subclasses__()}


def create_rate_constant(T: float, prefactor_backend_name: str, **kwargs) -> RateConstant:
    """
    Examples
    --------
    >>> rc = create_rate_constant(T=300.0, prefactor_backend_name="constant", config=MyConfig(k0=10))
    """
    registry = _get_registry()
    if prefactor_backend_name not in registry:
        raise ValueError(f"Backend '{prefactor_backend_name}' unknown. Available: {list(registry.keys())}")
    return RateConstant(T=T, prefactor_backend=registry[prefactor_backend_name](**kwargs))