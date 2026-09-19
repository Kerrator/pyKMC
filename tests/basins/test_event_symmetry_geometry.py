"""Six independent basin geometry cases; no native engine, IRA or MPI launch.

Literal expected coordinates distinguish displacement symmetry about the
unchanged reference initial from rotating absolute coordinates around zero.
PSR, minimization and native refinement are named substituted boundaries;
actual basin callers, System and resolved-constraint validation run.
"""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc.basins.basin as basin_module
from pykmc.physics import ResolvedConstraints
from pykmc.result import Ok
from pykmc.system import System


ATOL = 2e-12
INITIAL = np.array([[5.0, 4.0, 3.0], [7.0, 4.0, 3.0], [6.0, 5.0, 3.0]])
DISPLACEMENT = np.array([[0.1, 0.2, 0.0], [0.3, -0.1, 0.05], [0.0, 0.1, 0.1]])
SOURCE = np.array([[10.0, 16.0, 6.0], [11.0, 15.0, 6.0], [11.0, 17.0, 6.0]])
# A 90-degree PSR rotation and nonidentity permutation applied after either
# identity or x-reflection symmetry. These values are independently tabulated.
EXPECTED_DELTA = (
    np.array([[-0.1, 0.0, 0.1], [-0.2, 0.1, 0.0], [0.1, 0.3, 0.05]]),
    np.array([[-0.1, 0.0, 0.1], [0.1, -0.3, 0.05], [-0.2, -0.1, 0.0]]),
)
IDS = (41, 7, 99)


def fixture(monkeypatch, style):
    source = System(
        types=np.array(["Cu"] * 3),
        positions=SOURCE.copy(),
        cell=np.diag([40.0, 40.0, 40.0]),
        pbc=[True, True, True],
        index=np.array(IDS),
    )
    neighbors = SimpleNamespace(get_neighbors=lambda *_: np.arange(3))
    state = SimpleNamespace(
        system=source,
        neighbors_list=neighbors,
        ensure_full_state=lambda *_: None,
        release_heavy_objects=lambda: None,
    )
    reference = pd.DataFrame(
        [
            dict(
                idx_ref=31,
                types=["Cu"] * 3,
                initial_positions=INITIAL.copy(),
                saddle_positions=INITIAL + DISPLACEMENT,
                final_positions=INITIAL + 2 * DISPLACEMENT,
                sym_matrix=[np.eye(3), np.diag([-1.0, 1.0, 1.0])],
                sym_perm=[np.arange(3), np.array([1, 0, 2])],
            )
        ],
        index=[73],
    )
    saved_reference = {
        key: np.array(reference.iloc[0][key], copy=True)
        for key in (
            "initial_positions",
            "saddle_positions",
            "final_positions",
            "sym_matrix",
            "sym_perm",
        )
    }
    psr = SimpleNamespace(
        rotation_matrix=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        translation_matrix=np.array([15.0, 10.0, 3.0]),
        permutation_matrix=np.array([2, 0, 1]),
        matching_score=0.0,
    )

    def matching_boundary(_cfg, working, _reference, _neighbors, _center):
        np.testing.assert_array_equal(working.positions, SOURCE)
        return SimpleNamespace(match=lambda: Ok(psr))

    monkeypatch.setattr(basin_module, "PointSetRegistration", matching_boundary)
    cfg = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        frozen_atoms=None,
        basin=SimpleNamespace(style=style),
        psr=SimpleNamespace(matching_score_thr=1e-10),
        reconstruction=SimpleNamespace(push_fraction=0.1),
    )
    basin = basin_module.BasinsGenericEvents.__new__(basin_module.BasinsGenericEvents)
    basin.config, basin.states = cfg, {0: state}
    basin.reference_table = SimpleNamespace(table=reference)
    return basin, source, reference, saved_reference


def unchanged(source, reference, saved_reference):
    np.testing.assert_array_equal(source.positions, SOURCE)
    np.testing.assert_array_equal(source.index, IDS)
    np.testing.assert_array_equal(source.pbc, [True, True, True])
    assert tuple(source.types) == ("Cu",) * 3
    for key, value in saved_reference.items():
        np.testing.assert_array_equal(np.asarray(reference.iloc[0][key]), value)


@pytest.mark.parametrize("style", ["global", "global/reconstruction"])
@pytest.mark.parametrize("sym_idx", [0, 1])
def test_basin_stationary_triplet_keeps_initial_aligned_and_source_immutable(
    monkeypatch, style, sym_idx
):
    basin, source, reference, saved = fixture(monkeypatch, style)
    saddle = SOURCE + EXPECTED_DELTA[sym_idx]
    final = SOURCE + 2 * EXPECTED_DELTA[sym_idx]
    validations, dispatches = [], []
    original_validate = ResolvedConstraints.validate_positions

    def validate(constraints, positions, *args, **kwargs):
        validations.append(np.array(positions, copy=True))
        return original_validate(constraints, positions, *args, **kwargs)

    monkeypatch.setattr(ResolvedConstraints, "validate_positions", validate)

    def minimize(**kwargs):
        np.testing.assert_allclose(kwargs["positions"], final, atol=ATOL, rtol=0)
        assert tuple(kwargs["types"]) == ("Cu",) * 3
        assert kwargs["constraints"].source_ids == IDS
        dispatches.append("global endpoint")
        return final.copy(), -1.0

    class ReconstructionBoundary:
        def __init__(self, cfg, manager, types=None, constraints=None, pbc=True):
            assert tuple(types) == ("Cu",) * 3
            assert constraints.source_ids == IDS
            assert tuple(pbc) == (True, True, True)

        def reconstruct(self, first, last, working_saddle, cell, threshold, neighbors):
            np.testing.assert_allclose(first, SOURCE, atol=ATOL, rtol=0)
            np.testing.assert_allclose(last, final, atol=ATOL, rtol=0)
            np.testing.assert_allclose(working_saddle, saddle, atol=ATOL, rtol=0)
            np.testing.assert_array_equal(neighbors, np.arange(3))
            dispatches.append("two-endpoint reconstruction")
            return Ok(SimpleNamespace(min2_positions=final.copy()))

    basin.manager = SimpleNamespace(group_minimize_with_results=minimize)
    monkeypatch.setattr(basin_module, "Reconstruction", ReconstructionBoundary)
    result = basin.system_from_state(0, 31, 0, sym_idx)
    assert result.is_ok()
    assert len(dispatches) == 1
    np.testing.assert_allclose(result.ok_value().positions, final, atol=ATOL, rtol=0)
    # R04 requires validation of the complete claimed triplet before either
    # caller overlays it. That boundary exposes the initial even in global mode.
    assert len(validations) >= 3
    for actual, expected in zip(validations[:3], (SOURCE, saddle, final)):
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    unchanged(source, reference, saved)


def completed(value):
    future = Future()
    future.set_result(value)
    return future


@pytest.mark.parametrize("sym_idx", [0, 1])
def test_absorbing_refinement_receives_symmetric_displacement_about_original_initial(
    monkeypatch, sym_idx
):
    basin, source, reference, saved = fixture(monkeypatch, "global/reconstruction")
    expected_saddle = SOURCE + EXPECTED_DELTA[sym_idx]
    energy_calls, refine_calls = [], []

    def energy(**kwargs):
        np.testing.assert_array_equal(kwargs["positions"], SOURCE)
        energy_calls.append(1)
        return completed(-1.0)

    def refine(**kwargs):
        np.testing.assert_allclose(
            kwargs["positions"], expected_saddle, atol=ATOL, rtol=0
        )
        np.testing.assert_array_equal(kwargs["saddle_idx"], np.arange(3))
        refine_calls.append(1)
        return completed(
            Ok(SimpleNamespace(E_saddle=-0.8, saddle_positions=expected_saddle.copy()))
        )

    basin.manager = SimpleNamespace(get_total_energy=energy, partn_refine=refine)
    basin.connectivity_table = SimpleNamespace(
        df=pd.DataFrame(
            [
                dict(
                    transient=False,
                    state=0,
                    state_connexion=5,
                    event_connexion=31,
                    central_atom=0,
                    sym=sym_idx,
                )
            ]
        )
    )
    basin.absorbing_saddle_positions = {}
    # Rate arithmetic is intentionally outside this geometry-only oracle.
    basin._absorbing_rate = lambda *_: 0.25
    assert basin.refine_absorbing(source).is_ok()
    assert len(energy_calls) == len(refine_calls) == 1
    np.testing.assert_allclose(
        basin.absorbing_saddle_positions[(0, 5)], expected_saddle, atol=ATOL, rtol=0
    )
    unchanged(source, reference, saved)
