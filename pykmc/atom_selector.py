"""Atom selection utilities for inactive and frozen atom constraints."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pykmc.config import AtomSelector


def resolve_atom_selector(
    selector: "AtomSelector",
    positions: np.ndarray,
    types: list[str],
) -> frozenset[int]:
    """Resolve an AtomSelector to a frozenset of 0-based atom indices.

    Combines all criteria (types, indices, region) with union semantics.
    Called at runtime so region-based selection tracks atom movement.

    Parameters
    ----------
    selector : AtomSelector
        Selector configuration.
    positions : np.ndarray, shape (N, 3)
        Current atom Cartesian positions.
    types : list[str]
        Chemical symbol for each atom (e.g. ['Fe', 'Fe', 'O']).

    Returns
    -------
    frozenset[int]
        0-based atom indices matching any criterion.
    """
    selected: set[int] = set()

    if selector.types:
        type_set = set(selector.types)
        selected.update(i for i, t in enumerate(types) if t in type_set)

    if selector.indices:
        selected.update(selector.indices)

    if selector.region is not None:
        selected.update(_resolve_region(selector.region, positions))

    return frozenset(selected)


def _resolve_region(region, positions: np.ndarray) -> set[int]:
    n = len(positions)

    if region.region_type == "sphere":
        c = np.array(region.center)
        r = region.radius
        dists = np.linalg.norm(positions - c, axis=1)
        base = set(np.where(dists <= r)[0].tolist())

    elif region.region_type == "shell":
        c = np.array(region.center)
        dists = np.linalg.norm(positions - c, axis=1)
        base = set(np.where((dists >= region.inner_radius) & (dists <= region.radius))[0].tolist())

    elif region.region_type == "box":
        mask = (
            (positions[:, 0] >= region.xlo) & (positions[:, 0] <= region.xhi) &
            (positions[:, 1] >= region.ylo) & (positions[:, 1] <= region.yhi) &
            (positions[:, 2] >= region.zlo) & (positions[:, 2] <= region.zhi)
        )
        base = set(np.where(mask)[0].tolist())

    elif region.region_type == "plane":
        ax = {"x": 0, "y": 1, "z": 2}[region.normal]
        if region.side == "above":
            base = set(np.where(positions[:, ax] >= region.threshold)[0].tolist())
        else:  # "below"
            base = set(np.where(positions[:, ax] <= region.threshold)[0].tolist())
        return base

    if region.side == "outside":
        return set(range(n)) - base
    return base
