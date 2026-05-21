"""Tests for the v2 event-recycling feature (displacement + distance).
Goal with this test is to have 3 vacancies with 2 close together with a 3rd far away, we should see
that when event around vacancy A or B is excuted then the events around vacancy C will be recycled.
a test should also have two close events be on other sides of PBC to check that this distance is PBC aware

"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from pykmc import Config, System
from pykmc.config import EventRecyclingConfig
from pykmc.event_recycling import select_recyclable_events
from pykmc.event_table import ActiveEventTable


_BASE_INI = (
    "[Control]\n"
    "initial_config = ./initial_config.xyz\n"
    "n_steps = 1\n"
    "engine = lammps\n"
    "n_sessions = 1\n"
    "engine_use_rank_0 = True\n"
    "[Lammps]\n"
    "pair_style = eam/alloy\n"
    "pair_coeff = * * dummy.eam Ni\n"
    "[AtomicEnvironment]\n"
    "style = cna/graph\n"
    "rnei = 2.8\n"
    "rcut = 6.5\n"
    "[EventSearch]\n"
    "style = partn\n"
    "nsearch = 1\n"
    "[pARTn]\n"
    "path_artnso = /dummy/libartn.so\n"
    "[RateConstant]\n"
    "style = constant\n"
    "k0 = 1e12\n"
    "T = 300.0\n"
    "[PSR]\n"
    "style = ira\n"
    "[IRA]\n"
)


def _make_active_table(rows: list[dict]) -> ActiveEventTable:
    """Build an ActiveEventTable directly from a list of row dicts."""
    table = pd.DataFrame(rows)
    return ActiveEventTable(Mock(), event_dataframe=table)


def _make_ni_fcc_with_vacancies(
    vacancy_targets: list[tuple[float, float, float]],
) -> tuple[System, list[int], list[int]]:
    """Build a 10x10x10 Ni FCC supercell and remove atoms at the given target positions.

    Each target is snapped to the closest FCC site (PBC-aware). For each removed
    atom, the closest surviving FCC neighbor is recorded as the "central atom"
    of a candidate event at that vacancy.

    Returns
    -------
    (system, vacancy_post_indices, central_atom_indices)
        - system : 4000 - len(vacancy_targets) atoms, box L = 35.2 Å, cubic.
        - vacancy_post_indices : the post-removal indices of the removed atoms
          (always -1 since they no longer exist; included for symmetry of API).
        - central_atom_indices : post-removal indices of one surviving neighbor
          per vacancy, suitable as the central_atom_index of a candidate event.

    """
    a = 3.52
    repeats = 10
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    ) * a

    positions = []
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                shift = np.array([i, j, k]) * a
                for atom in basis:
                    positions.append(atom + shift)
    positions = np.array(positions)
    L = repeats * a

    # Snap each target to the closest FCC site (PBC-aware).
    removed_pre: list[int] = []
    for target in vacancy_targets:
        diff = positions - np.asarray(target)
        diff -= L * np.round(diff / L)
        idx = int(np.argmin(np.linalg.norm(diff, axis=1)))
        removed_pre.append(idx)

    # Pick one nearest neighbor per vacancy (must not itself be a vacancy or already chosen).
    central_pre: list[int] = []
    for vac_idx in removed_pre:
        diff = positions - positions[vac_idx]
        diff -= L * np.round(diff / L)
        order = np.argsort(np.linalg.norm(diff, axis=1))
        for j in order:
            j_int = int(j)
            if j_int == vac_idx or j_int in removed_pre or j_int in central_pre:
                continue
            central_pre.append(j_int)
            break

    # Remove vacancies and compute pre→post index map.
    keep_mask = np.ones(len(positions), dtype=bool)
    for idx in removed_pre:
        keep_mask[idx] = False
    survivor_positions = positions[keep_mask]
    pre_to_post = {int(pre): post for post, pre in enumerate(np.where(keep_mask)[0])}

    system = System()
    system.positions = survivor_positions
    system.types = ["Ni"] * len(survivor_positions)
    system.cell = np.diag([L, L, L])
    system.pbc = np.array([True, True, True])
    system.index = np.arange(len(survivor_positions))

    central_post = [pre_to_post[c] for c in central_pre]
    vacancy_post = [pre_to_post.get(r, -1) for r in removed_pre]
    return system, vacancy_post, central_post


class TestEventRecyclingConfig:
    def test_defaults(self) -> None:
        cfg = EventRecyclingConfig()
        assert cfg.enabled is False
        assert cfg.movement_thr == 0.02
        assert cfg.distance_thr == 10.0

    def test_thresholds_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            EventRecyclingConfig(movement_thr=0.0)
        with pytest.raises(ValidationError):
            EventRecyclingConfig(distance_thr=-1.0)

    def test_ini_parsing(self, tmp_path) -> None:
        ini = tmp_path / "input.in"
        ini.write_text(
            _BASE_INI
            + "[EventRecycling]\n"
            + "enabled = True\n"
            + "movement_thr = 0.05\n"
            + "distance_thr = 8.0\n"
        )
        config = Config.from_ini_file(str(ini))
        assert config.eventrecycling.enabled is True
        assert config.eventrecycling.movement_thr == 0.05
        assert config.eventrecycling.distance_thr == 8.0

    def test_section_optional(self, tmp_path) -> None:
        ini = tmp_path / "input.in"
        ini.write_text(_BASE_INI)
        config = Config.from_ini_file(str(ini))
        assert config.eventrecycling.enabled is False
        assert config.eventrecycling.movement_thr == 0.02
        assert config.eventrecycling.distance_thr == 10.0


class TestRealisticVacancyScenario:
    """10x10x10 Ni FCC (~4000 atoms) with vacancies at 3 sites.

    Vacancy A at box center; vacancy B ~8 Å from A (close); vacancy C ~20 Å from A
    on a diagonal (box L = 35.2 Å limits single-axis PBC distance to L/2 = 17.6 Å,
    so the "far" vacancy must use a diagonal placement).
    """

    A_target = (17.6, 17.6, 17.6)
    B_target = (25.6, 17.6, 17.6)            # A + (8, 0, 0)
    C_target = (31.6, 31.6, 22.6)            # A + (14, 14, 5), ~20.4 Å

    @staticmethod
    def _build() -> tuple[System, list[int], list[int]]:
        return _make_ni_fcc_with_vacancies(
            [
                TestRealisticVacancyScenario.A_target,
                TestRealisticVacancyScenario.B_target,
                TestRealisticVacancyScenario.C_target,
            ]
        )

    @staticmethod
    def _row(atom_index: int) -> dict:
        return {
            "atom_index": atom_index,
            "saddle_positions": np.zeros((1, 3)),
            "final_positions": np.zeros((1, 3)),
            "energy_barrier": 0.5,
            "k": 1.0,
            "num_reference_event": 0,
            "refined": "T",
        }

    def test_close_discarded_far_recycled(self) -> None:
        """8-Å vacancy must NOT be recycled; 20-Å vacancy MUST be recycled."""
        system, _vac, central = self._build()
        # Sanity: we should have ~3997 atoms.
        assert len(system.positions) == 4000 - 3
        positions_pre = system.positions.copy()
        # Simulate execution at vacancy A: shift its central atom by 0.3 Å.
        system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
        active = _make_active_table([self._row(c) for c in central])
        recycled = select_recyclable_events(
            active, executed_idx=0, system=system,
            positions_pre=positions_pre,
            movement_thr=0.02, distance_thr=10.0,
        )
        assert len(recycled) == 1
        assert int(recycled.iloc[0]["atom_index"]) == central[2]

    def test_distance_threshold_boundary(self) -> None:
        """Widen distance_thr past C's ~20 Å → C is no longer 'far' → 0 recycled."""
        system, _vac, central = self._build()
        positions_pre = system.positions.copy()
        system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
        active = _make_active_table([self._row(c) for c in central])
        recycled = select_recyclable_events(
            active, executed_idx=0, system=system,
            positions_pre=positions_pre,
            movement_thr=0.02, distance_thr=25.0,
        )
        assert len(recycled) == 0

    def test_movement_check_overrides_distance(self) -> None:
        """Move C's central atom by 0.05 Å (> movement_thr) → not recycled despite being far."""
        system, _vac, central = self._build()
        positions_pre = system.positions.copy()
        system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
        system.positions[central[2]] = positions_pre[central[2]] + np.array([0.05, 0.0, 0.0])
        active = _make_active_table([self._row(c) for c in central])
        recycled = select_recyclable_events(
            active, executed_idx=0, system=system,
            positions_pre=positions_pre,
            movement_thr=0.02, distance_thr=10.0,
        )
        assert len(recycled) == 0

    def test_self_excluded(self) -> None:
        """The executed-event row never appears in the recycled output."""
        system, _vac, central = self._build()
        positions_pre = system.positions.copy()
        system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
        active = _make_active_table([self._row(c) for c in central])
        recycled = select_recyclable_events(
            active, executed_idx=0, system=system,
            positions_pre=positions_pre,
            movement_thr=0.02, distance_thr=10.0,
        )
        assert central[0] not in [int(a) for a in recycled["atom_index"].tolist()]

    def test_pbc_wrap_close(self) -> None:
        """A 4th vacancy across the periodic wrap must be identified as CLOSE.

        With L=35.2, D at A + (33,0,0) wraps to A − (2.2,0,0) → PBC distance ≈ 2.2 Å.
        Only the 20-Å vacancy C should be recycled.
        """
        D_target = (TestRealisticVacancyScenario.A_target[0] + 33.0, 17.6, 17.6)
        system, _vac, central = _make_ni_fcc_with_vacancies(
            [self.A_target, self.B_target, self.C_target, D_target]
        )
        positions_pre = system.positions.copy()
        system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
        active = _make_active_table([self._row(c) for c in central])
        recycled = select_recyclable_events(
            active, executed_idx=0, system=system,
            positions_pre=positions_pre,
            movement_thr=0.02, distance_thr=10.0,
        )
        recycled_idx = [int(a) for a in recycled["atom_index"].tolist()]
        assert recycled_idx == [central[2]]
