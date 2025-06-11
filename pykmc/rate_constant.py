"""Module defining function used to compute the rate constant."""

from .config import PhysicalConstants, Config
import math as m


def compute_rate_Eyring(dE: float, config: Config) -> float:
    r"""Compute the rate constant based on the energy barrier and parameters in the configuration.

    It uses the following equation : 
    $$
    k0*\frac{k_{b}T}{h}e^{-\frac{dE}{k_{b}T}}
    $$

    Parameters
    ----------
    dE : float
        The energy barrier.
    config : Config
        The configuration of the simulation.

    Returns
    -------
    float
        the rate constant.

    """
    p = PhysicalConstants()
    T = config.rateconstant.T
    k0 = config.rateconstant.k0
    return k0 * ((p.kb * T) / p.h) * m.exp(-dE / (p.kb * T))


def compute_htst() -> None:
    """Define a future operation to be implemented."""
    pass
