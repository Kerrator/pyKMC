"""Dealloying dissolution events for the off-lattice KMC engine.

Physics (user-specified): an atom of a configured dissolvable element whose
first-shell (rnei) coordination is ``<= coord_max`` can dissolve, competing in
the BKL selection with the Erlebacher bond-counting rate

    k_diss(n) = nu_d * exp(-n * E_b / (kb * T))

where ``n`` is the atom's current coordination, ``E_b`` the effective bond
energy (eV) and ``nu_d`` the attempt frequency (ps^-1). Selecting the event
deletes the atom (the surface recedes as less-noble atoms dissolve).

This module holds only the pure scan/rate helpers and the selection record; the
KMC integration (BKL competition, deletion, engine rebuild) lives in
:mod:`pykmc.kmc`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PhysicalConstants


@dataclass(frozen=True)
class DissolutionSelection:
    """A dissolution event chosen by the rejection-free (BKL) selection.

    Attributes
    ----------
    atom_index : int
        Index (current system numbering) of the atom that dissolves.
    coordination : int
        First-shell (rnei) coordination ``n`` of the atom at selection time.
    rate : float
        Bond-counting rate ``k_diss(n)`` of the selected event (ps^-1).

    """

    atom_index: int
    coordination: int
    rate: float


def eligible_atoms(
    types: list[str] | np.ndarray,
    coordination: np.ndarray,
    elements: list[str],
    coord_max: int,
) -> np.ndarray:
    """Return the indices of atoms eligible to dissolve.

    An atom is eligible iff its chemical species is in the dissolvable
    ``elements`` set AND its first-shell coordination is ``<= coord_max``.

    Parameters
    ----------
    types : list[str] or np.ndarray of str
        Per-atom chemical symbols of the system.
    coordination : np.ndarray of int
        Per-atom first-shell (rnei) coordination count.
    elements : list[str]
        Chemical symbols of the dissolvable (less-noble) species.
    coord_max : int
        Maximum coordination for which an atom may dissolve.

    Returns
    -------
    np.ndarray of int
        Indices of the eligible atoms, in ascending order.

    """
    dissolvable = set(elements)
    is_dissolvable = np.array([t in dissolvable for t in types], dtype=bool)
    under_coordinated = np.asarray(coordination) <= coord_max
    return np.flatnonzero(is_dissolvable & under_coordinated)


def dissolution_rates(
    coordination: np.ndarray,
    nu_d: float,
    E_b: float,
    T: float,
) -> np.ndarray:
    """Compute the Erlebacher bond-counting dissolution rates.

    ``k_diss(n) = nu_d * exp(-n * E_b / (kb * T))`` elementwise over the
    coordination array; fewer bonds dissolve faster (monotone decreasing in n).

    Parameters
    ----------
    coordination : np.ndarray of int
        Per-atom first-shell coordination ``n``.
    nu_d : float
        Attempt frequency (ps^-1).
    E_b : float
        Effective bond energy (eV).
    T : float
        Temperature (K).

    Returns
    -------
    np.ndarray of float
        Dissolution rate of each atom (ps^-1), aligned with ``coordination``.

    """
    kb = PhysicalConstants().kb
    n = np.asarray(coordination, dtype=float)
    return nu_d * np.exp(-n * E_b / (kb * T))
