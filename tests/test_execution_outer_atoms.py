"""An executed event moves the atoms its saddle displaces outside rcut.

The refined saddle covers the whole system, and an active row keeps the rcut
neighbours of the central atom. The atoms outside rcut that the saddle displaces
by more than ``Reconstruction.saddle_disp_thr`` are kept beside them, as
displacements from the minimum. The tests run ``Refinement.execute`` ->
``KMC.add_active_events`` -> ``KMC._reconstruction_active_event`` with the PSR
match and the engine stubbed, and read the systems sent to the minimizer.
"""

from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import pykmc.refinement as refinement_module
from pykmc import System
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.psr import PSROutput
from pykmc.refinement import Refinement
from pykmc.result import EventRefinementOutput, Ok

_CELL = np.diag([20.0, 20.0, 20.0])
# Atom 0 is the central atom and atom 1 hops. Atoms 0-4 are the rcut neighbours.
# Atom 5 sits at the cell boundary and atom 6 is far away; both are outside rcut.
_MIN = np.array(
    [
        [10.0, 10.0, 10.0],
        [12.5, 10.0, 10.0],
        [10.0, 12.5, 10.0],
        [10.0, 10.0, 12.5],
        [7.5, 10.0, 10.0],
        [19.99, 10.0, 10.0],
        [10.0, 17.0, 10.0],
    ]
)
_RCUT = [0, 1, 2, 3, 4]
_SADDLE = _MIN.copy()
_SADDLE[1] = [13.75, 10.0, 10.0]
_SADDLE[5] = [0.04, 10.0, 10.0]  # +0.05 along x, across the boundary
_SADDLE[6] = [10.0, 17.001, 10.0]  # below the threshold
_FINAL = _MIN.copy()
_FINAL[1] = [15.0, 10.0, 10.0]
_OUTER_DISPLACEMENT = np.array([0.05, 0.0, 0.0])
_TOTAL_ENERGY = -10.0
_BARRIER = 0.5
_PUSH = 0.15


class _NeighborsList:
    """Neighbour list returning the fixed `rcut` list of the central atom."""

    def get_neighbors(self, cutoff_type: str, idx: int) -> np.ndarray:
        """Return the `rcut` list of the central atom."""
        return np.array(_RCUT)


class _IdentityPSR:
    """Point-set registration mapping the reference event onto itself."""

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "_IdentityPSR":
        """Mirror the strategy factory of `PointSetRegistration`."""
        return cls()

    def match_and_check(self, *args: Any, **kwargs: Any) -> Ok[PSROutput]:
        """Return the identity transformation."""
        return Ok(
            PSROutput(
                matching_score=0.0,
                rotation_matrix=np.eye(3),
                translation_matrix=np.zeros(3),
                permutation_matrix=np.arange(len(_RCUT)),
            )
        )


class _Manager:
    """Engine pool returning the refined saddle and the two minima."""

    def __init__(self) -> None:
        self.minimized: list[np.ndarray] = []

    def partn_refine(self, **kwargs: Any) -> concurrent.futures.Future:
        """Return the saddle of the whole system."""
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(
            Ok(
                EventRefinementOutput(
                    central_atom_index=kwargs["central_atom_idx"],
                    saddle_positions=_SADDLE.copy(),
                    E_saddle=_TOTAL_ENERGY + _BARRIER,
                    refined="T",
                )
            )
        )
        return future

    def group_minimize_with_results(
        self, config: Any, positions: np.ndarray, types: Any
    ) -> tuple[np.ndarray, float]:
        """Record the system and return the first minimum, then the second."""
        self.minimized.append(np.array(positions))
        minimum = _MIN if len(self.minimized) == 1 else _FINAL
        return minimum.copy(), _TOTAL_ENERGY


def _config(saddle_disp_thr: float) -> SimpleNamespace:
    return SimpleNamespace(
        # refine_thr above 1 puts the only reference event under the threshold
        control=SimpleNamespace(
            recycle=False, active_volume=False, refine_thr=2.0, seed=None
        ),
        eventsearch=SimpleNamespace(refined_energy_thr=0.1, refine_style="partn"),
        psr=SimpleNamespace(style="ira", matching_score_thr=0.4),
        ira=SimpleNamespace(),
        atomicenvironment=SimpleNamespace(atom_coloring_mode="full"),
        rateconstant=SimpleNamespace(style="constant", k0=10.0, T=300.0),
        reconstruction=SimpleNamespace(
            push_fraction=_PUSH, saddle_disp_thr=saddle_disp_thr
        ),
    )


def _reference_events() -> pd.DataFrame:
    """One reference event on the rcut neighbours."""
    return pd.DataFrame(
        [
            {
                "event_id": "env",
                "idx_ref": 0,
                "initial_positions": _MIN[_RCUT].copy(),
                "types": ["Ni"] * len(_RCUT),
                "move_atom_idx": 1,
                "saddle_positions": _SADDLE[_RCUT].copy(),
                "final_positions": _FINAL[_RCUT].copy(),
                "sym_matrix": [np.eye(3)],
                "sym_perm": [np.arange(len(_RCUT))],
                "energy_barrier": _BARRIER,
                "k": 1.0,
                "k_pref": 10.0,
            }
        ]
    )


def _refined_kmc(
    monkeypatch: pytest.MonkeyPatch, saddle_disp_thr: float = 0.01
) -> tuple[KMC, _Manager]:
    """Refine the event on central atom 0 and store its active row."""
    monkeypatch.setattr(refinement_module, "PointSetRegistration", _IdentityPSR)
    config = _config(saddle_disp_thr)
    system = System(
        types=np.array(["Ni"] * len(_MIN)),
        positions=_MIN.copy(),
        cell=_CELL.copy(),
        pbc=np.array([True, True, True]),
        index=np.arange(len(_MIN)),
    )
    atomic_environment = Mock()
    atomic_environment.get_atoms_with_id.return_value = [0]
    manager = _Manager()
    neighbors_list = _NeighborsList()

    refinement = Refinement(
        config, Mock(), system, neighbors_list, atomic_environment, manager
    )
    refinement.execute(_reference_events(), _TOTAL_ENERGY)
    kmc = KMC(config)
    kmc.system = system
    kmc.manager = manager
    kmc.neighbors_list = neighbors_list
    kmc.reference_table = SimpleNamespace(table=_reference_events())
    kmc.active_table = ActiveEventTable(config)
    kmc.add_active_events(refinement.get_successes_results())
    assert len(kmc.active_table.table) == 1
    return kmc, manager


def test_refined_row_keeps_displaced_outer_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row keeps atom 5 as a minimum-image displacement, and not atom 6."""
    kmc, _ = _refined_kmc(monkeypatch)
    row = kmc.active_table.table.loc[0]

    np.testing.assert_array_equal(row.at["outer_atoms"], [5])
    np.testing.assert_allclose(
        row.at["outer_saddle_displacements"], [_OUTER_DISPLACEMENT], atol=1e-12
    )
    np.testing.assert_array_equal(row.at["neighbors"], _RCUT)
    np.testing.assert_allclose(row.at["saddle_positions"], _SADDLE[_RCUT])


def test_threshold_decides_which_outer_atoms_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lower threshold keeps atom 6, a higher one keeps no outer atom."""
    kmc, _ = _refined_kmc(monkeypatch, saddle_disp_thr=0.0005)
    np.testing.assert_array_equal(kmc.active_table.table.at[0, "outer_atoms"], [5, 6])

    kmc, _ = _refined_kmc(monkeypatch, saddle_disp_thr=0.1)
    assert len(kmc.active_table.table.at[0, "outer_atoms"]) == 0


def test_execution_moves_outer_atoms_to_the_saddle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both minimizations start from a system with atom 5 moved, then it is put back."""
    kmc, manager = _refined_kmc(monkeypatch)

    result = kmc._reconstruction_active_event(0, kmc.active_table)

    assert result.is_ok()
    toward_min1, toward_min2 = manager.minimized
    # toward the first minimum: the hopping atom and atom 5 are pushed back
    np.testing.assert_allclose(toward_min1[1], [13.75 - _PUSH * 1.25, 10.0, 10.0])
    np.testing.assert_allclose(toward_min1[5], [0.04 - _PUSH * 0.05, 10.0, 10.0])
    # toward the second minimum: atom 5 has no target, so it stays at the saddle
    np.testing.assert_allclose(toward_min2[1], [13.75 + _PUSH * 1.25, 10.0, 10.0])
    np.testing.assert_allclose(toward_min2[5], _SADDLE[5])
    # atom 6 is under the threshold and never moves
    np.testing.assert_allclose(toward_min1[6], _MIN[6])
    np.testing.assert_allclose(toward_min2[6], _MIN[6])
    np.testing.assert_allclose(kmc.system.positions, _MIN)


def test_carried_row_displaces_outer_atoms_from_where_they_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outer atom that moved since the refinement is displaced from its new place."""
    kmc, manager = _refined_kmc(monkeypatch)
    moved = _MIN.copy()
    moved[5] = [19.99, 11.0, 10.0]
    kmc.system.update_positions(moved)

    kmc._reconstruction_active_event(0, kmc.active_table)

    toward_min1, toward_min2 = manager.minimized
    np.testing.assert_allclose(toward_min1[5], [0.04 - _PUSH * 0.05, 11.0, 10.0])
    np.testing.assert_allclose(toward_min2[5], [0.04, 11.0, 10.0])
    np.testing.assert_allclose(kmc.system.positions, moved)
