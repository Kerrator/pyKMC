"""Cluster D: serial == engine basin reconstruction acceptance parity.

The branch invariant is that the serial (``basin.strategy=serial``, rank-0 global
pool) and the engine (MPI pool) basin reconstruction paths accept/reject
identically. These tests cover the three parity levers made canonical to the
engine behaviour:

* ``per_atom_displacement`` honours a per-axis ``pbc`` so the acceptance metric
  is not full-MIC (pbc-blind) on a slab -- and defaults to full PBC so existing
  callers are unchanged (findings #13 metric).
* ``Reconstruction.reconstruct`` threads the runtime ``pbc`` into the ase wrap
  and ``push_towards`` calls exactly as the engine does (finding #13).
* The serial path requests the active-volume outer-sphere frozen minimize
  exactly when the engine gate (``control.active_volume`` + ``activevolume``)
  would, over the same frozen atom set (findings #12/#14).
"""
from unittest.mock import Mock

import numpy as np
import pytest

import ase.geometry
import pykmc.reconstruction as reconstruction_module
import pykmc.enginemanager.lmpi.lammps_operations as ops
from pykmc.reconstruction import Reconstruction
from pykmc.utils.geometry import per_atom_displacement


CELL = np.diag([10.0, 10.0, 10.0])


# ---------------------------------------------------------------------------
# (1) per_atom_displacement: slab pbc vs full PBC on a cross-boundary move
# ---------------------------------------------------------------------------
def test_per_atom_displacement_slab_pbc_vs_full_pbc() -> None:
    """A displacement across the free-surface (z) boundary is folded back under
    full PBC but reported at its true magnitude with pbc=[T, T, F]."""
    pre = np.array([[5.0, 5.0, 0.5]])
    post = np.array([[5.0, 5.0, 9.5]])  # moved 9.0 A in z, or 1.0 A wrapped

    full = per_atom_displacement(pre.copy(), post.copy(), CELL)  # default -> full PBC
    slab = per_atom_displacement(pre.copy(), post.copy(), CELL, pbc=[True, True, False])

    assert full[0] == pytest.approx(1.0)  # wrapped across z
    assert slab[0] == pytest.approx(9.0)  # not wrapped: real across-surface distance
    assert slab[0] > full[0]


# ---------------------------------------------------------------------------
# (4) default-parameter equivalence: full pbc reproduces the old numbers
# ---------------------------------------------------------------------------
def test_per_atom_displacement_default_matches_full_pbc_explicit() -> None:
    """Omitting pbc (the historical signature) is identical to pbc=[T, T, T]."""
    rng = np.random.default_rng(0)
    pre = rng.uniform(0.0, 10.0, size=(8, 3))
    post = rng.uniform(0.0, 10.0, size=(8, 3))

    old = per_atom_displacement(pre.copy(), post.copy(), CELL)
    new = per_atom_displacement(pre.copy(), post.copy(), CELL, pbc=[True, True, True])
    also = per_atom_displacement(pre.copy(), post.copy(), CELL, pbc=True)

    np.testing.assert_allclose(new, old)
    np.testing.assert_allclose(also, old)


def _config(active_volume: bool = False, rmov: float = 3.0) -> Mock:
    """A Mock config sufficient for Reconstruction.reconstruct."""
    config = Mock()
    config.reconstruction.push_fraction = 0.15
    config.reconstruction.n_movers = 3
    config.reconstruction.containment_margin = 1.0
    config.reconstruction.shell_tolerance = 1.0
    config.atomicenvironment.rcut = 6.5
    config.psr.matching_score_thr = 0.1
    config.control.active_volume = active_volume
    if active_volume:
        config.activevolume.rmov = rmov
    else:
        config.activevolume = None
    return config


# Simple 1D two-atom event used for the pbc / freeze routing tests.
_SADDLE = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
_MIN1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
_MIN2 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# (2) serial reconstruct threads the runtime pbc into wrap/push
# ---------------------------------------------------------------------------
def test_reconstruct_threads_pbc_into_wrap_and_push(monkeypatch) -> None:
    """The serial path must pass the runtime pbc to ase wrap_positions and to
    push_towards, not the hardcoded full-PBC default (finding #13)."""
    slab_pbc = [True, True, False]

    wrap_calls: list = []
    real_wrap = ase.geometry.wrap_positions

    def spy_wrap(positions, cell, pbc=True, **kwargs):
        wrap_calls.append(pbc)
        return real_wrap(positions, cell, pbc=pbc, **kwargs)

    monkeypatch.setattr(ase.geometry, "wrap_positions", spy_wrap)

    # Stub push_towards so its INTERNAL wrap_positions call does not pollute
    # wrap_calls; we only want the two reconstruct-level wraps here.
    push_calls: list = []

    def spy_push(current, target, fraction=0.1, cell=None, pbc=None):
        push_calls.append(pbc)
        return np.asarray(current, dtype=float)

    monkeypatch.setattr(reconstruction_module, "push_towards", spy_push)

    manager = Mock()
    manager.global_minimize_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(_config(active_volume=False), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), CELL, pbc=slab_pbc,
    )

    assert result.is_ok()
    # Both wrap sites (min1 + min2) received the runtime slab pbc, not True.
    assert wrap_calls == [slab_pbc, slab_pbc]
    # Both push sites (toward min1 + toward min2) received the runtime slab pbc.
    assert push_calls == [slab_pbc, slab_pbc]


def test_reconstruct_default_pbc_is_full_periodic(monkeypatch) -> None:
    """With pbc omitted, the wrap call keeps the historical full-PBC value so
    non-slab callers are byte-for-byte unchanged (finding #13, backward compat)."""
    wrap_calls: list = []
    real_wrap = ase.geometry.wrap_positions

    def spy_wrap(positions, cell, pbc=True, **kwargs):
        wrap_calls.append(pbc)
        return real_wrap(positions, cell, pbc=pbc, **kwargs)

    monkeypatch.setattr(ase.geometry, "wrap_positions", spy_wrap)

    def spy_push(current, target, fraction=0.1, cell=None, pbc=None):
        return np.asarray(current, dtype=float)

    monkeypatch.setattr(reconstruction_module, "push_towards", spy_push)

    manager = Mock()
    manager.global_minimize_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(_config(active_volume=False), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(_MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), CELL)

    assert result.is_ok()
    # Defaulted pbc -> the wrap sites fall back to full PBC (True), unchanged.
    assert wrap_calls == [True, True]


# ---------------------------------------------------------------------------
# (3) serial path requests the frozen minimize exactly when the engine gate fires
# ---------------------------------------------------------------------------
def test_reconstruct_routes_freeze_op_under_active_volume() -> None:
    """Under active_volume the serial path must call the outer-sphere frozen
    minimize op (not the unconstrained one), with the from-state freeze geometry,
    the central atom, the rmov radius, and the runtime pbc -- mirroring the engine
    (findings #12)."""
    rmov = 3.0
    from_positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    slab_pbc = [True, True, False]

    manager = Mock()
    manager.global_minimize_freeze_outer_sphere_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(
        _config(active_volume=True, rmov=rmov), manager, types=["Ni", "Ni"],
    )

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), CELL,
        central_atom=0, pbc=slab_pbc, from_positions=from_positions,
    )

    assert result.is_ok()
    # The unconstrained op must never be used under active volume.
    manager.global_minimize_with_results.assert_not_called()
    assert manager.global_minimize_freeze_outer_sphere_with_results.call_count == 2
    # Inspect the min1 call: same frozen-set inputs the engine gate uses.
    _args, kwargs = manager.global_minimize_freeze_outer_sphere_with_results.call_args_list[0]
    assert kwargs["central_atom"] == 0
    assert kwargs["rmov"] == rmov
    assert kwargs["pbc"] == slab_pbc
    np.testing.assert_allclose(kwargs["freeze_positions"], from_positions)


def test_reconstruct_uses_unconstrained_op_without_active_volume() -> None:
    """With active_volume off the serial path keeps the historical unconstrained
    global minimize; the frozen op is never touched (backward compat)."""
    manager = Mock()
    manager.global_minimize_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(_config(active_volume=False), manager, types=["Ni", "Ni"])

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), CELL, central_atom=0,
    )

    assert result.is_ok()
    assert manager.global_minimize_with_results.call_count == 2
    manager.global_minimize_freeze_outer_sphere_with_results.assert_not_called()


def test_reconstruct_main_loop_path_stays_unconstrained_under_active_volume() -> None:
    """Main-loop reconstruction (no from_positions) stays unconstrained under AV.

    AV on but ``from_positions`` omitted (the main KMC loop caller) must keep the
    plain unconstrained minimize with the frozen-group types, not route through
    the outer-sphere freeze op.

    The engine applies the outer-sphere freeze only inside its basin
    reconstruction op; the main loop reconstruction relaxes unconstrained even
    under active volume. The serial path marks the basin case by passing
    ``from_positions``, so with it omitted (as ``kmc.py`` does) the freeze must
    stay off regardless of ``config.control.active_volume``.
    """
    manager = Mock()
    manager.global_minimize_with_results.side_effect = [
        (_MIN1.copy(), 0.0),
        (_MIN2.copy(), -5.0),
    ]
    recon = Reconstruction(
        _config(active_volume=True, rmov=3.0), manager, types=["Ni", "Ni"],
    )

    result = recon.reconstruct(
        _MIN1.copy(), _MIN2.copy(), _SADDLE.copy(), CELL, central_atom=0,
    )

    assert result.is_ok()
    # Historical path preserved: unconstrained op carrying the frozen-group types.
    assert manager.global_minimize_with_results.call_count == 2
    for _args, kwargs in manager.global_minimize_with_results.call_args_list:
        assert kwargs["types"] == ["Ni", "Ni"]
    manager.global_minimize_freeze_outer_sphere_with_results.assert_not_called()


# ---------------------------------------------------------------------------
# Engine op: minimize_freeze_outer_sphere_with_results routing (no real LAMMPS)
# ---------------------------------------------------------------------------
def test_engine_freeze_op_calls_outer_sphere_when_rmov_set(monkeypatch) -> None:
    """With rmov set the engine op must apply the outer-sphere freeze (the same
    _minimize_freeze_outer_sphere the basin engine path uses), not a plain
    minimize -- so the serial global op relaxes the engine's constrained geometry.
    """
    engine = Mock()
    engine.rank = 0
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    freeze_positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])

    monkeypatch.setattr(ops, "_require_finite_positions", lambda *a, **k: None)
    monkeypatch.setattr(ops, "set_positions", lambda **k: None)
    monkeypatch.setattr(ops, "get_positions", lambda e: positions)
    monkeypatch.setattr(ops, "get_total_energy", lambda e: -3.0)
    plain = Mock()
    monkeypatch.setattr(ops, "minimize", plain)
    freeze = Mock()
    monkeypatch.setattr(ops, "_minimize_freeze_outer_sphere", freeze)

    out = ops.minimize_freeze_outer_sphere_with_results(
        engine, Mock(), positions=positions, freeze_positions=freeze_positions,
        central_atom=0, rmov=3.0, cell=CELL, pbc=[True, True, False],
    )

    freeze.assert_called_once()
    plain.assert_not_called()
    # frozen-set reference is the from-state geometry, not the pushed positions.
    args = freeze.call_args.args
    np.testing.assert_allclose(args[2], freeze_positions)
    assert args[3] == 0  # central_atom
    assert args[4] == 3.0  # rmov
    assert out == (positions, -3.0)


def test_engine_freeze_op_plain_minimize_when_rmov_none(monkeypatch) -> None:
    """rmov=None (active volume off) reduces to a plain minimize, matching the
    engine's ``av_rmov is None`` branch."""
    engine = Mock()
    engine.rank = 0
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    monkeypatch.setattr(ops, "_require_finite_positions", lambda *a, **k: None)
    monkeypatch.setattr(ops, "set_positions", lambda **k: None)
    monkeypatch.setattr(ops, "get_positions", lambda e: positions)
    monkeypatch.setattr(ops, "get_total_energy", lambda e: -3.0)
    plain = Mock()
    monkeypatch.setattr(ops, "minimize", plain)
    freeze = Mock()
    monkeypatch.setattr(ops, "_minimize_freeze_outer_sphere", freeze)

    out = ops.minimize_freeze_outer_sphere_with_results(
        engine, Mock(), positions=positions, rmov=None, cell=CELL,
    )

    plain.assert_called_once()
    freeze.assert_not_called()
    assert out == (positions, -3.0)
