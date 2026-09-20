"""Whole-event witness proposed on the mover's rcut crop (contracts 7f policy 6).

The IRA proposal of ``_common_map`` runs on the atoms within ``rcut`` of the
moving atom; the proposed rigid map is then extended to, and verified on, the
full coordinates. Spectator pairs far outside ``rcut`` make the full system
larger than the crop while keeping the whole event symmetric, so a recorded
IRA size tells the two proposals apart. No LAMMPS, no MPI.
"""

from __future__ import annotations

import numpy as np
import pytest

import pykmc.htst.event_identity as identity
from pykmc.htst.event_identity import calculations_equivalent

from . import test_whole_event_identity as base

SPECTATOR_RADII = (9.0, 11.0, 13.0, 15.0)
"""All beyond the test input's ``rcut`` (6.5 A) and its default free radius."""


def _with_spectators(core: np.ndarray) -> np.ndarray:
    """Append pairs mapped onto each other by ``base.ROTATION`` (x, y -> -x, -y)."""
    extra = []
    for r in SPECTATOR_RADII:
        extra += [[r, 0.0, 0.0], [-r, 0.0, 0.0], [0.0, r, 0.0], [0.0, -r, 0.0]]
    return np.vstack([core, np.array(extra)])


TYPES = base.SYMMETRIC_TYPES + ["C", "C", "Ge", "Ge"] * len(SPECTATOR_RADII)
INITIAL = _with_spectators(base.SYMMETRIC_INITIAL)
SADDLE = _with_spectators(base.SYMMETRIC_SADDLE)
# The core permutation swaps 1<->2 and 3<->4; every spectator pair swaps too.
PERM = np.concatenate(
    [base.SYMMETRIC_PERM]
    + [5 + 4 * k + np.array([1, 0, 3, 2]) for k in range(len(SPECTATOR_RADII))]
)
FINAL = (INITIAL @ base.ROTATION)[PERM]
N_ATOMS = len(TYPES)
N_CORE = len(base.SYMMETRIC_TYPES)


@pytest.fixture
def ira_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the atom count of every IRA proposal made by the identity module."""
    sizes: list[int] = []
    original = identity.simple_ira

    def recording(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor):
        sizes.append(int(nat1))
        return original(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)

    monkeypatch.setattr(identity, "simple_ira", recording)
    return sizes


def test_fixture_is_symmetric_and_needs_a_rotation() -> None:
    assert np.array_equal((SADDLE @ base.ROTATION)[PERM], SADDLE)
    assert np.array_equal((FINAL @ base.ROTATION)[PERM], INITIAL)
    assert N_ATOMS == N_CORE + 4 * len(SPECTATOR_RADII)
    # The identity-translation fast path cannot map min1 onto min2 within the
    # matching tolerance, so a proposal is needed.
    tolerance = base.configuration().psr.matching_score_thr
    assert np.max(np.linalg.norm(INITIAL - FINAL, axis=1)) > tolerance


def test_crop_proposal_is_verified_on_the_full_coordinates(ira_sizes: list[int]):
    """With ``crop_radius`` the witness is proposed on the crop, not on N atoms."""
    cfg = base.configuration()
    table = base.admitted(SADDLE, initial=INITIAL, final=FINAL, types=TYPES)
    forward = table.prefactor_archive.calculations[
        table.prefactor_archive.references[0].calculation_id
    ]
    backward = next(
        c
        for c in table.prefactor_archive.calculations.values()
        if c.direction == "backward"
    )
    ira_sizes.clear()
    assert calculations_equivalent(
        forward,
        backward,
        tolerance=cfg.psr.matching_score_thr,
        kmax_factor=cfg.ira.kmax_factor,
        crop_radius=cfg.atomicenvironment.rcut,
    )
    assert ira_sizes and set(ira_sizes) == {N_CORE}, ira_sizes
    ira_sizes.clear()
    # Without a crop radius the proposal still runs on the whole system.
    assert calculations_equivalent(
        forward,
        backward,
        tolerance=cfg.psr.matching_score_thr,
        kmax_factor=cfg.ira.kmax_factor,
    )
    assert set(ira_sizes) == {N_ATOMS}


def test_catalogue_collapse_proposes_on_the_rcut_crop(ira_sizes: list[int]) -> None:
    """The production self-reverse collapse never hands N atoms to IRA."""
    table = base.admitted(SADDLE, initial=INITIAL, final=FINAL, types=TYPES)
    assert len(table.table) == 1, "the symmetric control must still collapse"
    assert int(table.table.iloc[0].idx_backward) == int(table.table.iloc[0].idx_ref)
    assert ira_sizes, "the rotation-mapped event needs an IRA proposal"
    assert max(ira_sizes) == N_CORE
    assert N_ATOMS not in ira_sizes


def test_asymmetric_saddle_still_keeps_two_rows_with_the_crop_proposal(
    ira_sizes: list[int],
) -> None:
    """The policy-4 counterexample is unchanged by the crop-first proposal."""
    table = base.admitted(base.ASYMMETRIC_SADDLE)
    assert len(table.table) == 2
    assert set(
        zip(table.table.idx_ref.astype(int), table.table.idx_backward.astype(int))
    ) == {(0, 1), (1, 0)}


def test_admission_builds_no_request_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, htst_config, system_single_type_fcc
) -> None:
    """Admission is geometric: the full source is built once, for the worker."""
    from pykmc.rate_constant.prefactors import PrefactorService
    from tests.lifecycle.conftest import accepted

    from .test_identity_gate import _hop_event, _table_with_service

    calls: list[tuple] = []
    original = PrefactorService.build_request

    def counting(self, **kwargs):
        calls.append(kwargs["event_key"])
        return original(self, **kwargs)

    monkeypatch.setattr(PrefactorService, "build_request", counting)
    table, _ = _table_with_service(htst_config, accepted(5.0e12), accepted(5.0e12))
    system = system_single_type_fcc
    hop = _hop_event(system, 0, 0.5, 0.5, np.array([0.6, 0.0, 0.0]))
    table.add_events([hop], pbc=system.pbc)
    assert calls == [(0, 1)], calls  # only the dispatched request is built
    calls.clear()
    duplicate = table.add_events([hop], pbc=system.pbc)
    assert not duplicate[0].is_ok()
    assert calls == [], "a geometric duplicate never builds a full request"
