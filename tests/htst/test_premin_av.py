"""Engine-op tests for HTST prefactor pre-minimization and active-volume support.

Team requirements: (1) before each ``dynamical_matrix`` call the surroundings of
the event core (atoms outside ``atomicenvironment.rcut`` of the central atom)
must be relaxed with the core frozen — the ``minimize_freeze_core`` pattern
``partn_refine`` already uses; (2) the prefactor procedure must work with
active volumes (cropped AV subsystem, buffer frozen, indices remapped).

Serial-LAMMPS tests on the committed Ni(100) surface-hop fixture with the
tracked Ni EAM potential (no canon comparison — the canonical-nu0 tests pin
``premin = False`` to preserve their reference values).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lammps")

from pykmc.enginemanager.lmpi import lammps_operations as ops  # noqa: E402
from pykmc.htst import profiling  # noqa: E402
from pykmc.rate_constant.prefactor import EventPrefactors  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POTENTIAL = _REPO_ROOT / "tests" / "data" / "Ni_v6_2.0_LKBeland2016.eam"
_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "htst_ni100_surface_hop.npz"

_RCUT = 3.0  # small frozen core so the surroundings actually relax in tests
_FREE_RADIUS = 4.0


class _RC:
    """Rate-constant config shim (htst style)."""

    style = "htst"
    k0 = 10.0
    free_radius = _FREE_RADIUS
    # AV-off zone: 12 A covers the whole 130-atom fixture (max ~9.44 A from the
    # moving atom), so the zone crop == the full system and these tests keep the
    # original full-Hessian reference physics. The dedicated crop test below uses
    # a smaller radius to exercise the actual cropping.
    nu0_zone_radius = 12.0
    fd_step = 0.01
    nu0_min_THz = 1e-6
    nu0_max_THz = 1e6
    require_one_negative_mode = True
    premin = True


class _AE:
    """Atomic-environment shim: rcut defines the frozen event core."""

    rcut = _RCUT


class _Control:
    """Control shim (active_volume toggled per test)."""

    active_volume = False


class _AVParams:
    """Active-volume radii (ract must cover free_radius)."""

    ract = 6.0
    rmov = 4.0
    AV_debug = False


class _Lammps:
    """LAMMPS potential section (needed by the AV reset path)."""

    pair_style = "eam/alloy"
    pair_coeff = f"* * {_POTENTIAL} Ni"


class _Cfg:
    """Full-config shim for the engine op."""

    def __init__(
        self, premin: bool = True, active_volume: bool = False, ract: float = 6.0
    ) -> None:
        self.rateconstant = _RC()
        self.rateconstant.premin = premin
        self.atomicenvironment = _AE()
        self.control = _Control()
        self.control.active_volume = active_volume
        self.activevolume = _AVParams()
        self.activevolume.ract = ract
        self.lammps = _Lammps()


def _load_fixture() -> "tuple[np.ndarray, np.ndarray, np.ndarray, int, int]":
    data = np.load(_FIXTURE)
    return (
        data["initial_positions"],
        data["saddle_positions"],
        data["final_positions"],
        int(data["move_atom_idx"]),
        int(data["n_atoms"]),
    )


def _engine_for(positions: np.ndarray) -> "tuple[object, np.ndarray]":
    return profiling.build_serial_engine(
        positions, potential=str(_POTENTIAL), pair_style="eam/alloy", element="Ni"
    )


@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_premin_surroundings_relaxes() -> None:
    """Pre-minimization lowers the energy of a rattled environment, core frozen."""
    init, sad, fin, move, n = _load_fixture()
    rng = np.random.default_rng(42)
    rattled = sad.copy()
    dist = np.linalg.norm(rattled - rattled[move], axis=1)
    far = dist > _RCUT
    rattled[far] += rng.normal(0.0, 0.05, size=(int(far.sum()), 3))

    engine, _cell = _engine_for(sad)
    cfg = _Cfg(premin=True)

    e_before = ops.get_total_energy(engine, positions=rattled.copy())
    relaxed = ops._premin_surroundings(engine, cfg, rattled.copy(), move)
    e_after = ops.get_total_energy(engine, positions=relaxed.copy())

    assert e_after < e_before  # surroundings relaxed downhill
    # the frozen core did not move
    core = ~far
    assert np.allclose(relaxed[core], rattled[core], atol=1e-8)
    # something outside the core actually moved
    assert not np.allclose(relaxed[far], rattled[far], atol=1e-10)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_premin_recovers_rattled_environment() -> None:
    """premin=True pulls a rattled far-field back toward the pristine nu0.

    Observable: |nu0(premin, rattled) - nu0(ref)| < |nu0(no premin, rattled) - nu0(ref)|
    where ref = nu0(no premin, pristine). This fails if premin is a no-op.
    """
    init, sad, fin, move, n = _load_fixture()
    rng = np.random.default_rng(7)

    def _rattle(geom: np.ndarray) -> np.ndarray:
        out = geom.copy()
        dist = np.linalg.norm(out - out[move], axis=1)
        far = dist > _RCUT
        out[far] += rng.normal(0.0, 0.03, size=(int(far.sum()), 3))
        return out

    r_init, r_sad, r_fin = _rattle(init), _rattle(sad), _rattle(fin)
    engine, cell = _engine_for(sad)
    kwargs = {"central_atom_idx": move, "types": ["Ni"] * n, "cell": cell}

    ref = ops.compute_event_prefactors(
        engine, _Cfg(premin=False),
        min1_positions=init, saddle_positions=sad, min2_positions=fin, **kwargs,
    ).nu0_forward
    raw = ops.compute_event_prefactors(
        engine, _Cfg(premin=False),
        min1_positions=r_init, saddle_positions=r_sad, min2_positions=r_fin, **kwargs,
    ).nu0_forward
    healed = ops.compute_event_prefactors(
        engine, _Cfg(premin=True),
        min1_positions=r_init, saddle_positions=r_sad, min2_positions=r_fin, **kwargs,
    ).nu0_forward

    assert ref is not None and raw is not None
    assert healed is not None, "premin path produced no nu0"
    assert abs(healed - ref) < abs(raw - ref)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_premin_false_is_deterministic_old_path() -> None:
    """premin=False reproduces the original path (deterministic nu0)."""
    init, sad, fin, move, n = _load_fixture()
    engine, cell = _engine_for(sad)
    cfg = _Cfg(premin=False)

    res1 = ops.compute_event_prefactors(
        engine, cfg, central_atom_idx=move,
        min1_positions=init, saddle_positions=sad, min2_positions=fin,
        types=["Ni"] * n, cell=cell,
    )
    res2 = ops.compute_event_prefactors(
        engine, cfg, central_atom_idx=move,
        min1_positions=init, saddle_positions=sad, min2_positions=fin,
        types=["Ni"] * n, cell=cell,
    )

    assert res1.nu0_forward is not None
    assert res2.nu0_forward == pytest.approx(res1.nu0_forward, rel=1e-12)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_active_volume_branch_returns_prefactors() -> None:
    """With active_volume on, the op crops to the AV subsystem and computes nu0."""
    init, sad, fin, move, n = _load_fixture()
    engine, cell = _engine_for(sad)

    res = ops.compute_event_prefactors(
        engine, _Cfg(premin=False, active_volume=True, ract=6.0),
        central_atom_idx=move,
        min1_positions=init, saddle_positions=sad, min2_positions=fin,
        types=["Ni"] * n, cell=cell,
    )

    assert isinstance(res, EventPrefactors)
    assert res.n_free >= 1
    assert res.nu0_forward is not None, f"AV nu0 failed: {res.reason}"
    # the Hessian runs on the separate serial COMM_SELF engine (the multi-rank
    # session engine deadlocks on dynamical_matrix); that engine must hold the
    # CROPPED AV subsystem, not the full system.
    serial = engine._serial_hessian_engine
    assert int(serial.lmp.get_natoms()) < n


@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_nu0_zone_crop_smaller_than_full() -> None:
    """AV-off: a small nu0_zone_radius crops the serial subsystem below the full N.

    The deadlock fix runs the Hessian on a serial COMM_SELF engine holding only
    the ``nu0_zone_radius`` sphere around the moving atom (free_radius..zone shell
    frozen). A 7 A zone selects ~75 of the 130 fixture atoms, so the serial engine
    must hold fewer than N and still return a real nu0.
    """
    init, sad, fin, move, n = _load_fixture()
    engine, cell = _engine_for(sad)

    cfg = _Cfg(premin=False, active_volume=False)
    cfg.rateconstant.nu0_zone_radius = 7.0  # ~75 atoms < 130 (free_radius=4 inside)

    res = ops.compute_event_prefactors(
        engine, cfg, central_atom_idx=move,
        min1_positions=init, saddle_positions=sad, min2_positions=fin,
        types=["Ni"] * n, cell=cell,
    )

    assert isinstance(res, EventPrefactors)
    assert res.n_free >= 1
    assert res.nu0_forward is not None, f"zone nu0 failed: {res.reason}"
    serial = engine._serial_hessian_engine
    n_zone = int(serial.lmp.get_natoms())
    assert n_zone < n, f"zone crop did not shrink the system ({n_zone} == {n})"


@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_active_volume_ract_guard() -> None:
    """Ract < free_radius falls back gracefully (no raise, reason set)."""
    init, sad, fin, move, n = _load_fixture()
    engine, cell = _engine_for(sad)

    res = ops.compute_event_prefactors(
        engine, _Cfg(premin=False, active_volume=True, ract=3.0),
        central_atom_idx=move,
        min1_positions=init, saddle_positions=sad, min2_positions=fin,
        types=["Ni"] * n, cell=cell,
    )

    assert isinstance(res, EventPrefactors)
    assert res.nu0_forward is None
    assert "ract" in res.reason
