from abc import ABC, abstractmethod


class PrefactorBackend(ABC):
    name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define a 'name' class attribute")

    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the rate constant prefactor."""
        pass