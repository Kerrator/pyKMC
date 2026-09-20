"""The active-volume shell is a transport restriction, not a coordinate contract.

Overlay/endpoint validation applies to the user constraint set only
(``config.frozen_atoms``), at a tolerance tied to ``psr.matching_score_thr``,
and re-clamps user rows with ``protect_positions``; a violated user constraint
on the reconstruction/basin paths is ``Err(RECONSTRUCTION_INVALID_EVENT_DATA)``,
never a bare ``ValueError`` out of the KMC loop (contracts 7f policy 5). The
AV-resolved union still travels to the manager for transport/freezing.
"""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc
import pykmc.basins.basin as basin_module
import pykmc.kmc as kmc_module
from pykmc.config import Config, RateConstantConfig, RegionConfig
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.neighbors_list import NeighborsList
from pykmc.physics import ResolvedConstraints
from pykmc.reconstruction import Reconstruction
from pykmc.refinement import Refinement
import pykmc.refinement as refinement_module
from pykmc.result import EventRefinementOutput, ErrorType, Ok, PSROutput
from pykmc.system import System


# ---------------------------------------------------------------------------
# Basin / reconstruction geometry (source IDs differ from local rows)
# ---------------------------------------------------------------------------

IDS = (40, 8, 19, 2, 77)
TYPES = ("Ni", "Fe", "Ni", "Ni", "Ni")
CELL = np.diag([8.0, 8.0, 8.0])
PBC = (True, True, True)
NEIGHBORS = np.array([3, 0, 4, 1, 2])
USER_ROW = 1  # the Fe atom, source ID 8, the only user-frozen atom
SHELL_ROW = 2  # source ID 19: farther than rmov from the centre, not user-frozen
INVALID = ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA


def event():
    b = np.sqrt(2.0 ** (1.0 / 3.0) - 1.0)
    saddle = np.array(
        [
            [4.0, 7.2, 4.0],
            [4.0, 0.2, 4.0],
            [4.0, 6.2, 4.0],
            [4.0, 7.2, 5.0],
            [4.0, 7.2, 3.0],
        ]
    )
    first, second = saddle.copy(), saddle.copy()
    first[0, 0] -= b
    second[0, 0] += b
    return first, saddle, second


def config(style="global/reconstruction", active=True, thr=0.1):
    return SimpleNamespace(
        frozen_atoms=RegionConfig(types=["Fe"]),
        control=SimpleNamespace(active_volume=active),
        activevolume=SimpleNamespace(rmov=1.3, ract=2.25),
        basin=SimpleNamespace(style=style),
        reconstruction=SimpleNamespace(push_fraction=0.1),
        psr=SimpleNamespace(matching_score_thr=thr),
    )


def payload(cfg):
    first, _, _ = event()
    active = cfg.control.active_volume
    return ResolvedConstraints.resolve(
        first,
        TYPES,
        cfg.frozen_atoms,
        atom_ids=IDS,
        cell=CELL,
        pbc=PBC,
        center_id=IDS[USER_ROW] if active else None,
        rmov=cfg.activevolume.rmov if active else None,
    )


class EndpointManager:
    """Return the expected endpoints; record what the endpoint received."""

    def __init__(self, outputs):
        self.outputs = [np.array(p, copy=True) for p in outputs]
        self.calls = []

    def group_minimize_with_results(self, **kwargs):
        self.calls.append(
            (np.array(kwargs["positions"], copy=True), kwargs["constraints"])
        )
        return self.outputs[len(self.calls) - 1].copy(), -0.29365234375


class ProtocolSystem:
    def __init__(self, positions, types, cell, pbc, index):
        self.positions = np.array(positions, copy=True)
        self.types, self.cell, self.pbc, self.index = types, cell, pbc, index

    def update_positions(self, new_positions, atom_idx=None):
        if atom_idx is None:
            self.positions = np.array(new_positions, copy=True)
        else:
            self.positions[atom_idx] = new_positions


def basin_from(monkeypatch, cfg, triplet, translation):
    first, saddle, second = triplet
    manager = EndpointManager(
        [second] if cfg.basin.style == "global" else [first, second]
    )
    source = ProtocolSystem(first, TYPES, CELL, np.array(PBC), np.array(IDS))
    neighbors = SimpleNamespace(get_neighbors=lambda *_: NEIGHBORS)
    state = SimpleNamespace(
        system=source, neighbors_list=neighbors, ensure_full_state=lambda *_: None
    )
    ref = pd.DataFrame(
        [
            dict(
                idx_ref=31,
                initial_positions=first[NEIGHBORS],
                saddle_positions=saddle[NEIGHBORS],
                final_positions=second[NEIGHBORS],
            )
        ]
    )
    psr = SimpleNamespace(
        rotation_matrix=np.eye(3),
        translation_matrix=np.asarray(translation, dtype=float),
        permutation_matrix=np.arange(5),
    )
    monkeypatch.setattr(basin_module, "System", ProtocolSystem)
    monkeypatch.setattr(
        basin_module,
        "PointSetRegistration",
        lambda *_args, **_kw: SimpleNamespace(match=lambda: Ok(psr)),
    )
    monkeypatch.setattr(basin_module, "check_match", lambda result, *_: result)
    basin = basin_module.BasinsGenericEvents.__new__(basin_module.BasinsGenericEvents)
    basin.config, basin.manager = cfg, manager
    basin.reference_table = SimpleNamespace(table=ref)
    basin.states = {0: state}
    return basin, source, manager


@pytest.mark.parametrize("style", ["global", "global/reconstruction"])
def test_basin_accepts_psr_residual_on_user_frozen_atom_within_tolerance(
    monkeypatch, style
):
    """A 0.01 A PSR residual on a frozen atom (thr 0.1 A) is re-clamped, not fatal."""
    cfg = config(style, active=False, thr=0.1)
    first, saddle, second = event()
    basin, source, manager = basin_from(
        monkeypatch, cfg, (first, saddle, second), [0.01, 0.0, 0.0]
    )
    result = basin.system_from_state(0, 31, USER_ROW, 0)
    assert result.is_ok(), getattr(result, "err_value", lambda: None)()
    assert len(manager.calls) == (1 if style == "global" else 2)
    for positions, constraints in manager.calls:
        # The user-frozen row is re-clamped to its reference coordinate.
        np.testing.assert_array_equal(positions[USER_ROW], first[USER_ROW])
        # The residual on a movable row is the accepted PSR mapping, kept as is.
        assert abs(positions[SHELL_ROW, 0] - first[SHELL_ROW, 0]) == pytest.approx(
            0.01, abs=1e-12
        )
        assert constraints.fixed_ids == (IDS[USER_ROW],)
    np.testing.assert_array_equal(source.positions, first)


@pytest.mark.parametrize("style", ["global", "global/reconstruction"])
def test_basin_violated_user_constraint_returns_err_not_valueerror(monkeypatch, style):
    """A 0.2 A residual on a frozen atom (thr 0.1 A) is an Err that purges the row."""
    cfg = config(style, active=False, thr=0.1)
    first, saddle, second = event()
    basin, source, manager = basin_from(
        monkeypatch, cfg, (first, saddle, second), [0.2, 0.0, 0.0]
    )
    result = basin.system_from_state(0, 31, USER_ROW, 0)
    assert not result.is_ok()
    assert result.err_value().type is INVALID
    assert manager.calls == []
    np.testing.assert_array_equal(source.positions, first)


@pytest.mark.parametrize("style", ["global", "global/reconstruction"])
def test_basin_av_shell_displacement_is_protected_not_rejected(monkeypatch, style):
    """A catalogue final geometry moving an rmov-shell atom is re-clamped, not fatal."""
    cfg = config(style, active=True, thr=0.1)
    first, saddle, second = event()
    second = second.copy()
    second[SHELL_ROW, 0] += 0.05  # outside rmov, inside rcut, not user-frozen
    basin, source, manager = basin_from(
        monkeypatch, cfg, (first, saddle, second), [0.0, 0.0, 0.0]
    )
    result = basin.system_from_state(0, 31, USER_ROW, 0)
    assert result.is_ok(), getattr(result, "err_value", lambda: None)()
    assert len(manager.calls) == (1 if style == "global" else 2)
    for positions, constraints in manager.calls:
        # Endpoint minimisations still receive the full user+AV transport union
        # with every fixed row (shell included) at its source coordinate.
        assert set(constraints.fixed_ids) == {8, 19, 2, 77}
        np.testing.assert_array_equal(positions[1:], first[1:])
    np.testing.assert_array_equal(source.positions, first)


def test_reconstruction_violated_user_constraint_is_err_not_valueerror():
    cfg = config(active=True, thr=1e-7)
    first, saddle, second = event()
    bad_second = second.copy()
    bad_second[USER_ROW, 0] += 0.2
    manager = EndpointManager([first, second])
    recon = Reconstruction(cfg, manager, types=TYPES, constraints=payload(cfg), pbc=PBC)
    result = recon.reconstruct(
        first[NEIGHBORS], bad_second[NEIGHBORS], saddle, CELL, 1e-7, NEIGHBORS
    )
    assert not result.is_ok()
    assert result.err_value().type is INVALID
    assert manager.calls == []


def test_reconstruction_av_shell_residual_is_protected_and_dispatched():
    cfg = config(active=True, thr=0.1)
    first, saddle, second = event()
    shifted_second = second.copy()
    shifted_second[SHELL_ROW, 0] += 0.05
    manager = EndpointManager([first, second])
    recon = Reconstruction(cfg, manager, types=TYPES, constraints=payload(cfg), pbc=PBC)
    result = recon.reconstruct(
        first[NEIGHBORS], shifted_second[NEIGHBORS], saddle, CELL, 0.1, NEIGHBORS
    )
    assert result.is_ok(), result.err_value()
    assert len(manager.calls) == 2
    for positions, _ in manager.calls:
        np.testing.assert_array_equal(positions[1:], first[1:])


# ---------------------------------------------------------------------------
# KMC reconstruction call site: a ValueError from Reconstruction is an Err
# ---------------------------------------------------------------------------


def _kmc_case(monkeypatch, reconstruction_double):
    root = Path(pykmc.__file__).resolve().parent.parent
    cfg = Config.from_ini_file(str(root / "tests/data/input.in"))
    cfg.control.active_volume = False
    cfg.control.reference_table = None
    cfg.control.seed = None
    cfg.frozen_atoms = RegionConfig(indices=[1])
    cfg.rateconstant = RateConstantConfig(style="constant", k0=7.0)
    cfg.atomicenvironment.rnei = 2.0
    cfg.atomicenvironment.rcut = 2.0
    source = np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [20.0, 10.0, 10.0]])
    system = System(
        types=np.array(["Si"] * 3),
        positions=source.copy(),
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
        index=np.array([42, 8, 91]),
    )
    neighbors = NeighborsList(system, rnei=2.0, rcut=2.0)
    table = ActiveEventTable(cfg)
    table.add_events(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=np.array([[10.3, 10.0, 10.0], [11.0, 10.0, 10.0]]),
            E_saddle=0.1,
            min2_positions=np.array([[10.6, 10.0, 10.0], [11.0, 10.0, 10.0]]),
            dE_forward=0.1,
            num_reference_event=47,
            refined="F",
            crop_atom_ids=(42, 8),
        )
    )
    monkeypatch.setattr(kmc_module, "Reconstruction", reconstruction_double)
    sim = KMC(cfg, manager=object())
    sim.system, sim.neighbors_list = system, neighbors
    return sim, table, source


def test_kmc_reconstruction_valueerror_becomes_err_for_the_purge_loop(monkeypatch):
    class RaisingReconstruction:
        def __init__(self, config, manager, **kwargs):
            pass

        def reconstruct(self, *args, **kwargs):
            raise ValueError("event changes fixed reference coordinates")

    sim, table, source = _kmc_case(monkeypatch, RaisingReconstruction)
    result = sim._reconstruction_active_event(0, table)
    assert not result.is_ok()
    assert result.err_value().type is INVALID
    assert "fixed reference" in result.err_value().message
    np.testing.assert_array_equal(sim.system.positions, source)


# ---------------------------------------------------------------------------
# Refinement overlay: shell placed as given, user rows re-clamped or Err
# ---------------------------------------------------------------------------

R_IDS = (42, 17, 8, 91)
R_TYPES = ("Ni",) * 4
R_CELL = np.diag([10.0, 12.0, 14.0])
R_PBC = (True, False, True)
R_CENTER = 1  # source ID 17
R_USER_ROW = 2  # source ID 8: y = 3.5 lies in the frozen plane region y >= 3.3
R_SHELL_ROW = 3  # source ID 91: 4.5 A from the centre, beyond rmov, inside ract
R_ROWS = np.array([3, 2, 0, 1])  # rcut patch includes the shell atom


def refine_config(active=True, thr=0.1):
    return SimpleNamespace(
        frozen_atoms=RegionConfig(
            region_type="plane", normal="y", side="above", threshold=3.3
        ),
        control=SimpleNamespace(active_volume=active),
        activevolume=SimpleNamespace(rmov=1.6, ract=5.0, AV_debug=False),
        atomicenvironment=SimpleNamespace(rcut=2.0),
        eventsearch=SimpleNamespace(refined_energy_thr=1e-7),
        psr=SimpleNamespace(matching_score_thr=thr),
    )


def refine_geometries(user_shift=0.0, shell_shift=0.0):
    source = np.array(
        [[2.25, 3.0, 3.0], [3.5, 3.0, 3.0], [3.5, 3.5, 3.0], [8.0, 3.0, 3.0]]
    )
    saddle, final = source.copy(), source.copy()
    saddle[0, 0] += 0.4
    final[0, 0] += 0.8
    saddle[R_USER_ROW, 0] += user_shift
    saddle[R_SHELL_ROW, 0] += shell_shift
    return source, saddle, final


class RefineManager:
    def __init__(self):
        self.calls = []

    def partn_refine(self, **kwargs):
        self.calls.append(kwargs)
        output = EventRefinementOutput(
            central_atom_index=R_CENTER,
            saddle_positions=np.array(kwargs["positions"], copy=True),
            E_saddle=0.2,
            refined="T",
        )
        result = Future()
        result.set_result(Ok(output))
        return result


def refinery(monkeypatch, cfg, triplet):
    source, saddle, final = triplet

    class Matching:
        def __init__(self, *args, **kwargs):
            pass

        def match(self):
            return Ok(
                PSROutput(
                    rotation_matrix=np.eye(3),
                    translation_matrix=np.zeros(3),
                    permutation_matrix=np.arange(len(R_ROWS)),
                    matching_score=0.0,
                )
            )

    monkeypatch.setattr(refinement_module, "PointSetRegistration", Matching)
    user = ResolvedConstraints.resolve(
        source, R_TYPES, cfg.frozen_atoms, R_IDS, cell=R_CELL, pbc=R_PBC
    )
    assert user.fixed_ids == (R_IDS[R_USER_ROW],)
    system = System(
        types=np.array(R_TYPES),
        positions=source.copy(),
        cell=R_CELL.copy(),
        pbc=R_PBC,
        index=np.array(R_IDS),
    )
    neighbors = SimpleNamespace(get_neighbors=lambda *_: R_ROWS.copy())
    manager = RefineManager()
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        progress_bar=lambda *a, **k: None,
    )
    caller = Refinement(
        cfg,
        logger,
        system,
        neighbors,
        SimpleNamespace(get_atoms_with_id=lambda _: [R_CENTER]),
        manager,
        global_constraints=user,
    )
    row = pd.Series(
        {
            "idx_ref": 5,
            "event_id": "fixture",
            "energy_barrier": 0.2,
            "initial_positions": source[R_ROWS].copy(),
            "saddle_positions": saddle[R_ROWS].copy(),
            "final_positions": final[R_ROWS].copy(),
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(len(R_ROWS))],
        }
    )
    caller._carry_prefactors = False
    return caller, manager, row, system


def _refine(caller, row):
    context = {}
    futures = caller.refine_single(R_CENTER, row, 0.0, context, e_thr=1.0)
    assert isinstance(futures, list) and len(futures) == 1
    return futures[0].result(), context[futures[0]]


def test_refinement_places_av_shell_overlay_as_given(monkeypatch):
    cfg = refine_config(active=True, thr=1e-7)
    triplet = refine_geometries(shell_shift=0.05)
    caller, manager, row, system = refinery(monkeypatch, cfg, triplet)
    result, _ = _refine(caller, row)
    assert result.is_ok()
    assert len(manager.calls) == 1
    call = manager.calls[0]
    # Transport still carries the full user+AV union.
    assert set(call["constraints"].fixed_ids) == {8, 91}
    shell_local = int(np.flatnonzero(R_ROWS == R_SHELL_ROW)[0])
    np.testing.assert_array_equal(
        call["saddle_positions"][shell_local], triplet[1][R_SHELL_ROW]
    )
    np.testing.assert_array_equal(system.positions, triplet[0])


@pytest.mark.parametrize("active", [True, False], ids=["av", "non-av"])
def test_refinement_reclamps_user_row_within_tolerance(monkeypatch, active):
    cfg = refine_config(active=active, thr=0.1)
    triplet = refine_geometries(user_shift=0.01)
    caller, manager, row, _ = refinery(monkeypatch, cfg, triplet)
    result, _ = _refine(caller, row)
    assert result.is_ok()
    assert len(manager.calls) == 1
    call = manager.calls[0]
    if active:
        user_local = int(np.flatnonzero(R_ROWS == R_USER_ROW)[0])
        placed = call["saddle_positions"][user_local]
    else:
        placed = call["positions"][R_USER_ROW]
    np.testing.assert_array_equal(placed, triplet[0][R_USER_ROW])
    # A movable row keeps its overlay coordinate.
    movable_local = int(np.flatnonzero(R_ROWS == 0)[0])
    moved = call["saddle_positions"][movable_local] if active else call["positions"][0]
    np.testing.assert_allclose(moved, triplet[1][0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("active", [True, False], ids=["av", "non-av"])
def test_refinement_violated_user_constraint_is_err_future(monkeypatch, active):
    cfg = refine_config(active=active, thr=0.1)
    triplet = refine_geometries(user_shift=0.2)
    caller, manager, row, system = refinery(monkeypatch, cfg, triplet)
    result, context = _refine(caller, row)
    assert not result.is_ok()
    assert result.err_value().type is INVALID
    assert manager.calls == []
    assert context["num_reference_event"] == 5
    np.testing.assert_array_equal(system.positions, triplet[0])
