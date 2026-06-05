from abc import ABC, abstractmethod


class PrefactorBackend(ABC):
    """Abstract base class for prefactor backends.

    Subclasses must define:

    - a ``name`` class attribute (``str``) : unique key used by the factory registry.
    - a ``compute(**kwargs) -> float`` method : returns the prefactor in ps^-1.

    A ``TypeError`` is raised at class definition time if ``name`` is missing.
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