"""Mocked tests for the active-volume full-system restore in the pARTn operations."""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.activevolume import active_volume as av
from pykmc.enginemanager.lmpi import lammps_operations as ops
from pykmc.result import Err, ErrorType, Ok


class _FakeComm:
    def bcast(self, value, root=0):
        return value


class _FakeLammps:
    def __init__(self, natoms):
        self._natoms = natoms
        self.commands = []

    def get_natoms(self):
        return self._natoms

    def command(self, cmd):
        self.commands.append(cmd)


class _FakeEngine:
    def __init__(self, natoms):
        self.rank = 0
        self.engine_id = 0
        self.local_engine_comm = _FakeComm()
        self.lmp = _FakeLammps(natoms)
        self.commands = []

    def command(self, cmd):
        self.commands.append(cmd)


class _SearchArtn:
    lib = SimpleNamespace(_name="libartn-lmp.so")

    def reset_input(self):
        pass

    def set(self, *_args):
        pass

    def get_error(self):
        return [0]

    def extract(self, key):
        values = {
            "delr_min1": 0.1,
            "delr_min2": 1.0,
            "etot_sad": 1.5,
            "etot_min1": 1.0,
            "etot_min2": 0.8,
        }
        return values[key]


class _RefineArtnSuccess:
    lib = SimpleNamespace(_name="libartn-lmp.so")

    def reset_input(self):
        pass

    def set(self, *_args):
        pass

    def get_error(self):
        return [0]

    def extract(self, key):
        values = {
            "delr_sad": 0.1,
            "etot_sad": 1.5,
            "tau_sad": np.array([[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]]),
        }
        return values[key]


class _RefineArtnFailure:
    lib = SimpleNamespace(_name="libartn-lmp.so")

    def reset_input(self):
        pass

    def set(self, *_args):
        pass

    def get_error(self):
        return [77, "failed"]

    def extract(self, key):
        if key == "delr_sad":
            return 1.0
        raise KeyError(key)


def _config():
    return SimpleNamespace(
        control=SimpleNamespace(active_volume=True),
        frozen_atoms=None,
        eventsearch=SimpleNamespace(delr_thr=0.5),
        partn=SimpleNamespace(
            path_artnso="libartn.so",
            verbosity=0,
            dmax=0.2,
            delr_thr=0.5,
            zseed=7,
            nevalf_max=1000,
            r_nevalf_max=1000,
            evalf_max=1000,
            r_evalf_max=1000,
            push_mode="rad",
            push_dist_thr=0.1,
            push_step_size=0.1,
            ninit=1,
            lanczos_min_size=1,
            lanczos_max_size=2,
            lanczos_disp=0.1,
            lanczos_eval_conv_thr=1e-3,
            eigval_thr=-1.0,
            eigen_step_size=0.1,
            nsmooth=1,
            neigen=1,
            alpha_mix_cr=0.1,
            nnewchance=1,
            nperp=None,
            nperp_limitation=None,
            forc_thr=0.1,
            push_over=0.1,
            r_push_mode="rad",
            r_push_dist_thr=0.1,
            r_push_step_size=0.1,
            r_ninit=1,
            r_lanczos_min_size=1,
            r_lanczos_max_size=2,
            r_lanczos_disp=0.1,
            r_lanczos_eval_conv_thr=1e-3,
            r_eigval_thr=-1.0,
            r_eigen_step_size=0.1,
            r_nsmooth=1,
            r_neigen=1,
            r_alpha_mix_cr=0.1,
            r_nnewchance=1,
            r_nperp=None,
            r_nperp_limitation=None,
            r_forc_thr=0.1,
            r_max_attempts=1,
            r_dmax=0.2,
            r_delr_sad_thr=0.5,
        ),
        lammps=SimpleNamespace(
            boundary="p p p",
            pair_style="eam/alloy",
            pair_coeff="* * dummy.eam Ni",
            min_style="cg",
            minimize="1e-6 1e-8 10 10",
            frz_min="1e-6 1e-8 10 10",
        ),
        atomicenvironment=SimpleNamespace(rcut=6.5),
        activevolume=SimpleNamespace(AV_debug=False),
    )


def _patch_stdout_redirect(monkeypatch):
    monkeypatch.setattr(ops.os, "dup", lambda _fd: 10)
    monkeypatch.setattr(ops.os, "open", lambda _path, _flags: 11)
    monkeypatch.setattr(ops.os, "dup2", lambda _src, _dst: None)
    monkeypatch.setattr(ops.os, "close", lambda _fd: None)


def test_ensure_full_system_reinitializes_after_atom_count_mismatch(monkeypatch):
    captured = {}

    engine = _FakeEngine(natoms=2)
    config = SimpleNamespace(lammps=SimpleNamespace(boundary="p p p"))
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])

    #_ensure_full_system inlines the rebuild: clear + the same initialize sequence
    #the engine boot path uses. Capture each stage instead of one reinit function.
    monkeypatch.setattr(ops, "initialize_parameters", lambda eng: captured.update(params_engine=eng))
    monkeypatch.setattr(ops, "initialize_system", lambda eng, system: captured.update(system=system))
    monkeypatch.setattr(ops, "initialize_potential", lambda eng, cfg: captured.update(config=cfg))

    ops._ensure_full_system(engine, config, positions, cell, types)

    assert engine.lmp.commands == ["clear"]
    assert captured["params_engine"] is engine
    assert captured["config"] is config
    system = captured["system"]
    assert np.array_equal(system.positions, positions)
    assert np.array_equal(system.cell, cell)
    assert np.array_equal(system.types, types)
    assert np.array_equal(system.pbc, np.array([True, True, True]))
    assert np.array_equal(system.index, np.arange(len(positions)))


def test_partn_search_restores_full_system_after_active_volume(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])
    engine = _FakeEngine(natoms=2)
    config = _config()
    restore_calls = []

    _patch_stdout_redirect(monkeypatch)
    monkeypatch.setattr(ops, "partn_search_AV", lambda *_args: (np.array([0, 2]), np.array([1])))
    monkeypatch.setattr(
        ops,
        "position_results_AV",
        lambda *_args: (positions.copy(), positions.copy(), positions.copy(), 0),
    )
    monkeypatch.setattr(ops.pypARTn, "artn", lambda engine="lmp": _SearchArtn())
    monkeypatch.setattr(
        ops,
        "_ensure_full_system",
        lambda engine, config, positions, cell, types: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True))
        ),
    )

    result = ops.partn_search(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        types=types,
    )

    assert isinstance(result, Ok)
    assert len(restore_calls) == 1
    assert np.array_equal(restore_calls[0][0], positions)


def test_partn_refine_restores_full_system_after_successful_active_volume(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])
    engine = _FakeEngine(natoms=2)
    config = _config()
    restore_calls = []

    monkeypatch.setattr(
        ops,
        "partn_refine_AV",
        lambda *_args: (0.3, np.array([0, 2]), np.array([1])),
    )
    monkeypatch.setattr(ops.pypARTn, "artn", lambda engine="lmp": _RefineArtnSuccess())
    monkeypatch.setattr(
        ops,
        "_ensure_full_system",
        lambda engine, config, positions, cell, types: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True))
        ),
    )

    result = ops.partn_refine(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        types=types,
        saddle_idx=np.array([0, 2]),
        saddle_positions=positions[[0, 2]],
    )

    assert isinstance(result, Ok)
    assert result.ok_value().E_saddle == 1.2
    assert len(restore_calls) == 1
    assert np.array_equal(result.ok_value().saddle_positions[1], positions[1])


def test_partn_refine_restores_full_system_after_failed_active_volume(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])
    engine = _FakeEngine(natoms=2)
    config = _config()
    restore_calls = []

    monkeypatch.setattr(
        ops,
        "partn_refine_AV",
        lambda *_args: (0.3, np.array([0, 2]), np.array([1])),
    )
    monkeypatch.setattr(ops.pypARTn, "artn", lambda engine="lmp": _RefineArtnFailure())
    monkeypatch.setattr(
        ops,
        "_ensure_full_system",
        lambda engine, config, positions, cell, types: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True))
        ),
    )

    result = ops.partn_refine(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        types=types,
        saddle_idx=np.array([0, 2]),
        saddle_positions=positions[[0, 2]],
    )

    assert isinstance(result, Err)
    assert result.err_value().type == ErrorType.EVENT_NOT_FOUND
    assert len(restore_calls) == 1


def _patch_av_helpers(monkeypatch, av_idx):
    """Patch the LAMMPS-touching helpers so partn_refine_AV runs pure Python."""
    av_idx = np.asarray(av_idx, dtype=int)

    def fake_define_AV(_config, _central_atom_idx, positions, _cell):
        return positions[av_idx].copy(), av_idx, np.array([], dtype=int)

    monkeypatch.setattr(av, "reset", lambda *_a, **_kw: None)
    monkeypatch.setattr(av, "define_AV", fake_define_AV)
    monkeypatch.setattr(av, "redefine_atoms", lambda *_a, **_kw: None)
    monkeypatch.setattr(av, "make_AV", lambda *_a, **_kw: None)
    monkeypatch.setattr(av, "get_potential_energy", lambda _engine: -1.23)
    monkeypatch.setattr(av, "set_positions", lambda *_a, **_kw: None)


def test_partn_refine_AV_multi_saddle_atoms(monkeypatch):
    """partn_refine_AV must handle multiple saddle atoms without tripping
    numpy's ``int()``-on-1D-array restriction (numpy >= 1.25)."""
    natoms = 15
    positions = np.arange(natoms * 3, dtype=float).reshape(natoms, 3)
    cell = np.eye(3) * 100.0
    types = np.array(["Ni"] * natoms)
    av_idx = [10, 5, 7, 12]
    _patch_av_helpers(monkeypatch, av_idx)

    engine = _FakeEngine(natoms=natoms)
    config = _config()
    config.activevolume = SimpleNamespace(ract=20.0, rmov=14.0, AV_debug=False)

    saddle_idx = np.array([7, 10])
    saddle_positions = positions[saddle_idx] + 0.1

    E_init, atom_map, central_lammps_id = av.partn_refine_AV(
        engine,
        config,
        central_atom_idx=10,
        positions=positions,
        cell=cell,
        type=types,
        saddle_idx=saddle_idx,
        saddle_positions=saddle_positions,
    )

    assert E_init == -1.23
    assert np.array_equal(atom_map, np.array(av_idx))
    # central_atom_idx=10 is at atom_map index 0 -> LAMMPS id 1
    assert np.array_equal(np.asarray(central_lammps_id), np.array([1]))


def test_partn_refine_AV_single_saddle_atom(monkeypatch):
    """Single-atom saddle_idx is the corner case that silently worked on
    older numpy and must still work after the fix."""
    natoms = 8
    positions = np.arange(natoms * 3, dtype=float).reshape(natoms, 3)
    cell = np.eye(3) * 100.0
    types = np.array(["Ni"] * natoms)
    av_idx = [3, 1, 6]
    _patch_av_helpers(monkeypatch, av_idx)

    engine = _FakeEngine(natoms=natoms)
    config = _config()
    config.activevolume = SimpleNamespace(ract=20.0, rmov=14.0, AV_debug=False)

    av.partn_refine_AV(
        engine,
        config,
        central_atom_idx=3,
        positions=positions,
        cell=cell,
        type=types,
        saddle_idx=np.array([6]),
        saddle_positions=np.array([[9.0, 9.0, 9.0]]),
    )


def test_partn_refine_AV_wraps_saddle_positions_before_scatter(monkeypatch):
    """Transformed basin saddles may be one periodic image outside the box;
    AV refinement should wrap them before handing positions to LAMMPS."""
    natoms = 4
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [50.0, 50.0, 50.0],
            [75.0, 75.0, 75.0],
        ]
    )
    cell = np.eye(3) * 100.0
    types = np.array(["Ni"] * natoms)
    av_idx = [0, 1]
    _patch_av_helpers(monkeypatch, av_idx)

    captured = {}
    monkeypatch.setattr(
        av,
        "set_positions",
        lambda _engine, av_positions: captured.setdefault("positions", av_positions.copy()),
    )

    engine = _FakeEngine(natoms=natoms)
    config = _config()
    config.activevolume = SimpleNamespace(ract=20.0, rmov=14.0, AV_debug=False)

    av.partn_refine_AV(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        type=types,
        saddle_idx=np.array([1]),
        saddle_positions=np.array([[102.0, 1.0, 1.0]]),
    )

    assert np.allclose(captured["positions"][1], np.array([2.0, 1.0, 1.0]))


def test_partn_refine_AV_rejects_saddle_outside_active_radius(monkeypatch):
    natoms = 4
    positions = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [50.0, 50.0, 50.0],
            [75.0, 75.0, 75.0],
        ]
    )
    cell = np.eye(3) * 100.0
    types = np.array(["Ni"] * natoms)
    av_idx = [0, 1]
    _patch_av_helpers(monkeypatch, av_idx)

    engine = _FakeEngine(natoms=natoms)
    config = _config()
    config.activevolume = SimpleNamespace(ract=20.0, rmov=14.0, AV_debug=False)

    with pytest.raises(ValueError, match="outside active radius"):
        av.partn_refine_AV(
            engine,
            config,
            central_atom_idx=0,
            positions=positions,
            cell=cell,
            type=types,
            saddle_idx=np.array([1]),
            saddle_positions=np.array([[40.0, 1.0, 1.0]]),
        )


def test_partn_refine_AV_raises_on_missing_saddle_atom(monkeypatch):
    """Saddle atom not present in the AV map is a programming error —
    the function must fail fast with a clear message."""
    natoms = 8
    positions = np.arange(natoms * 3, dtype=float).reshape(natoms, 3)
    cell = np.eye(3) * 100.0
    types = np.array(["Ni"] * natoms)
    av_idx = [3, 1, 6]
    _patch_av_helpers(monkeypatch, av_idx)

    engine = _FakeEngine(natoms=natoms)
    config = _config()
    config.activevolume = SimpleNamespace(ract=20.0, rmov=14.0, AV_debug=False)

    with pytest.raises(ValueError, match="expected exactly 1"):
        av.partn_refine_AV(
            engine,
            config,
            central_atom_idx=3,
            positions=positions,
            cell=cell,
            type=types,
            saddle_idx=np.array([99]),
            saddle_positions=np.array([[0.0, 0.0, 0.0]]),
        )


def test_partn_refine_returns_err_for_invalid_active_volume_saddle(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.8, 0.8, 0.8]])
    cell = np.eye(3)
    types = np.array(["Ni", "Ni", "Ni"])
    engine = _FakeEngine(natoms=2)
    config = _config()
    restore_calls = []

    def bad_av(*_args):
        raise ValueError("saddle escaped active volume")

    monkeypatch.setattr(ops, "partn_refine_AV", bad_av)
    monkeypatch.setattr(
        ops,
        "_ensure_full_system",
        lambda engine, config, positions, cell, types: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True))
        ),
    )

    result = ops.partn_refine(
        engine,
        config,
        central_atom_idx=0,
        positions=positions,
        cell=cell,
        types=types,
        saddle_idx=np.array([0, 2]),
        saddle_positions=positions[[0, 2]],
    )

    assert isinstance(result, Err)
    assert result.err_value().type == ErrorType.REFINEMENT_INVALID_MINIMA
    assert result.err_value().message == "saddle escaped active volume"
    assert len(restore_calls) == 1


def _freeze_test_geometry():
    """Line of 6 atoms along x in a 20 A periodic box, central atom at x=1.

    Atom 5 sits at x=17: 16 A away naively but 4 A away through the periodic
    boundary — the case a non-PBC-aware region sphere gets wrong.
    """
    cell = np.diag([20.0, 20.0, 20.0])
    pbc = np.array([True, True, True])
    positions = np.array([
        [1.0, 1.0, 1.0],   # 0: central atom
        [3.0, 1.0, 1.0],   # 1: 2 A away -> mobile
        [6.0, 1.0, 1.0],   # 2: 5 A away -> mobile at rmov=6
        [9.0, 1.0, 1.0],   # 3: 8 A away -> frozen at rmov=6
        [12.0, 1.0, 1.0],  # 4: 9 A away (MIC) -> frozen
        [17.0, 1.0, 1.0],  # 5: 4 A away THROUGH the boundary -> mobile
    ])
    return positions, cell, pbc


def test_minimize_freeze_outer_sphere_uses_minimum_image_selection():
    """The frozen set must follow the minimum-image convention (matching
    define_AV): an atom near the opposite box edge is NOT frozen when it is
    within rmov through the periodic boundary."""
    engine = _FakeEngine(natoms=6)
    config = SimpleNamespace(
        lammps=SimpleNamespace(min_style="cg", minimize="1e-10 1e-12 100 100")
    )
    positions, cell, pbc = _freeze_test_geometry()

    ops._minimize_freeze_outer_sphere(engine, config, positions, 0, 6.0, cell, pbc)

    cmds = engine.commands
    group_cmds = [c for c in cmds if c.startswith("group _av_outer id ")]
    assert len(group_cmds) == 1
    frozen = {int(t) for t in group_cmds[0].split()[3:]}
    # LAMMPS ids are 1-based: atoms 3 and 4 (ids 4, 5) are the only ones > 6 A
    # away under MIC; atom 5 (id 6) is 4 A away through the boundary.
    assert frozen == {4, 5}
    assert "fix _av_freeze _av_outer setforce 0.0 0.0 0.0" in cmds
    # Minimize phase (issued by ops.minimize)
    assert "min_style cg" in cmds
    assert "minimize 1e-10 1e-12 100 100" in cmds
    # Cleanup phase, in correct order
    cleanup_idx = cmds.index("unfix _av_freeze")
    assert cmds[cleanup_idx + 1] == "group _av_outer delete"


def test_minimize_freeze_outer_sphere_no_frozen_atoms_plain_minimize():
    """With every atom inside rmov there is nothing to freeze: no group/fix
    commands, just the minimize."""
    engine = _FakeEngine(natoms=6)
    config = SimpleNamespace(
        lammps=SimpleNamespace(min_style="cg", minimize="1e-10 1e-12 100 100")
    )
    positions, cell, pbc = _freeze_test_geometry()

    ops._minimize_freeze_outer_sphere(engine, config, positions, 0, 15.0, cell, pbc)

    cmds = engine.commands
    assert not any(c.startswith("group _av_outer") for c in cmds)
    assert not any(c.startswith("fix _av_freeze") for c in cmds)
    assert "min_style cg" in cmds


def test_minimize_freeze_outer_sphere_cleans_up_on_error(monkeypatch):
    """If LAMMPS minimize raises, the helper must still tear down the freeze
    state so subsequent calls don't trip over orphaned groups/fixes — and the
    cleanup itself must be best-effort (a cleanup failure is swallowed)."""
    engine = _FakeEngine(natoms=6)
    config = SimpleNamespace(
        lammps=SimpleNamespace(min_style="cg", minimize="1e-10 1e-12 100 100")
    )
    positions, cell, pbc = _freeze_test_geometry()

    def boom(eng, cfg, positions=None):
        raise RuntimeError("simulated lammps failure")

    monkeypatch.setattr(ops, "minimize", boom)

    with pytest.raises(RuntimeError, match="simulated lammps failure"):
        ops._minimize_freeze_outer_sphere(engine, config, positions, 0, 6.0, cell, pbc)

    # Cleanup must have run despite the exception
    cmds = engine.commands
    assert "unfix _av_freeze" in cmds
    assert "group _av_outer delete" in cmds


# --- basin_reconstruct: must leave the pooled engine whole (issue B) ---------
#
# A basin reconstruction sets full-system positions then minimizes; on a bad
# geometry LAMMPS can lose atoms, leaving the engine with a reduced atom count.
# Because the engine is reused from the session pool, that corruption then
# poisons every later set_positions ("scatter_atoms: Atom-IDs must exist").
# Like the partn active-volume ops, basin_reconstruct must restore the full
# system on exit via ``_ensure_full_system`` -- on BOTH the success and the
# failure path.


def _basin_reconstruct_args():
    from_positions = np.array([[0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    from_types = np.array(["Ni", "Ni"])
    cell = np.eye(3)
    pbc = np.array([True, True, True])
    return dict(
        from_positions=from_positions,
        from_types=from_types,
        cell=cell,
        pbc=pbc,
        ref_initial_positions=from_positions.copy(),
        ref_saddle_positions=from_positions.copy(),
        ref_final_positions=from_positions.copy(),
        ref_initial_types=list(from_types),
        sym_matrices=[np.eye(3)],
        sym_perms=[np.array([0, 1])],
        central_atom=0,
        sym_idx=0,
        neighbor_indices=np.array([0, 1]),
        matching_score_thr=0.4,
        kmax_factor=2.0,
        atom_coloring_mode="grey",
    )


def test_basin_reconstruct_restores_full_system_on_success(monkeypatch):
    """A successful reconstruction must still restore the full system on exit."""
    engine = _FakeEngine(natoms=2)
    config = SimpleNamespace(basin=SimpleNamespace(style="global/reconstruction"))
    args = _basin_reconstruct_args()
    restore_calls = []

    monkeypatch.setattr(
        ops, "_basin_reconstruct_impl",
        lambda *a, **k: {"ok": True, "min2_positions": args["from_positions"], "min2_etot": -1.0},
    )
    monkeypatch.setattr(
        ops, "_ensure_full_system",
        lambda engine, config, positions, cell, types, force=False: restore_calls.append(
            (positions.copy(), cell.copy(), np.array(types, copy=True), force)
        ),
    )

    result = ops.basin_reconstruct(engine, config, **args)

    assert result["ok"] is True
    assert len(restore_calls) == 1
    assert np.array_equal(restore_calls[0][0], args["from_positions"])
    assert np.array_equal(restore_calls[0][2], args["from_types"])
    # Success path heals via the cheap count check, not a forced rebuild.
    assert restore_calls[0][3] is False


def test_basin_reconstruct_restores_full_system_on_failure(monkeypatch):
    """A reconstruction whose engine work raises must still restore the engine.

    This is the corruption-cascade guard: a lost-atoms minimize raises, and the
    engine must be healed before it returns to the pool, or every later
    set_positions on the reused engine fails.
    """
    engine = _FakeEngine(natoms=2)
    config = SimpleNamespace(basin=SimpleNamespace(style="global/reconstruction"))
    args = _basin_reconstruct_args()
    restore_calls = []

    def boom(*a, **k):
        raise RuntimeError("Lost atoms: original 2 current 1")

    monkeypatch.setattr(ops, "_basin_reconstruct_impl", boom)
    monkeypatch.setattr(
        ops, "_ensure_full_system",
        lambda engine, config, positions, cell, types, force=False: restore_calls.append(
            (positions.copy(), force)
        ),
    )

    result = ops.basin_reconstruct(engine, config, **args)

    # Failure is reported as an error payload (rank 0) ...
    assert result["ok"] is False
    assert "Lost atoms" in result["message"]
    # ... and the engine was healed on the way out, with a FORCED rebuild
    # (get_natoms is unreliable after a lost-atoms error).
    assert len(restore_calls) == 1
    assert np.array_equal(restore_calls[0][0], args["from_positions"])
    assert restore_calls[0][1] is True


def test_basin_reconstruct_force_rebuilds_when_natoms_stale_after_error(monkeypatch):
    """Reproduce-first guard for the real corruption cascade.

    After a LAMMPS "Lost atoms" error, ``atom->natoms`` is NOT decremented (the
    error message itself reports "original N current M" — N stays the stored
    global), so ``get_natoms()`` keeps returning the full count while the atom map
    is corrupt. The count-based short-circuit in ``_ensure_full_system`` is then
    fooled: it sees ``natoms == expected`` and no-ops, leaving the poisoned engine
    in the pool where every later ``set_positions`` raises
    ``scatter_atoms: Atom-IDs must exist`` (observed 21k times in the binary run).

    Unlike the two tests above, this uses the REAL ``_ensure_full_system`` (only
    the LAMMPS-touching rebuild stages are stubbed) with ``get_natoms()`` pinned to
    the STALE FULL COUNT — the exact production condition. So it FAILS on the
    count-only guard and only passes once the failure path forces an unconditional
    rebuild.
    """
    args = _basin_reconstruct_args()
    n_full = len(args["from_positions"])
    # Engine reports the stale full count after the simulated lost-atoms error.
    engine = _FakeEngine(natoms=n_full)
    config = SimpleNamespace(basin=SimpleNamespace(style="global"))

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("ERROR: Lost atoms: original 2 current 1")

    monkeypatch.setattr(ops, "_basin_reconstruct_impl", boom)

    # Stub ONLY the LAMMPS-touching rebuild stages; exercise the real
    # _ensure_full_system so the count short-circuit is actually under test.
    rebuilt = {"params": False, "system": None, "potential": False}
    monkeypatch.setattr(ops, "initialize_parameters", lambda eng: rebuilt.update(params=True))
    monkeypatch.setattr(ops, "initialize_system", lambda eng, system: rebuilt.update(system=system))
    monkeypatch.setattr(ops, "initialize_potential", lambda eng, cfg: rebuilt.update(potential=True))

    result = ops.basin_reconstruct(engine, config, **args)

    assert result["ok"] is False
    assert "Lost atoms" in result["message"]
    # Despite get_natoms() == full count, the engine MUST have been rebuilt.
    assert engine.lmp.commands == ["clear"], (
        "engine not rebuilt: the count-based guard no-op'd on the stale full "
        "count after a lost-atoms error — this is the corruption cascade"
    )
    assert rebuilt["params"] and rebuilt["potential"]
    assert rebuilt["system"] is not None
    assert np.array_equal(rebuilt["system"].positions, args["from_positions"])
    assert np.array_equal(rebuilt["system"].types, args["from_types"])
