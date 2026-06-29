"""Serial-LAMMPS smoke tests for the HTST engine ops (no MPI).

``compute_event_prefactors`` runs the partial Hessian on a private serial
``COMM_SELF`` LAMMPS it builds from ``config.lammps`` (the multi-rank session
engine deadlocks on ``dynamical_matrix``), cropping to the AV or ``nu0_zone``
subsystem. These tests exercise that path on a real EAM engine.

Tests
-----
- test_get_forces_shape: gather_atoms("f") returns a finite (N,3) array.
- test_compute_event_prefactors_runs_on_engine: end-to-end call returns an
  EventPrefactors dataclass without raising (identical geometries -> graceful
  fallback, n_free still reported).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lammps")

from lammps import lammps  # noqa: E402

from pykmc.enginemanager.lmpi import lammps_operations as ops  # noqa: E402
from pykmc.htst import profiling  # noqa: E402
from pykmc.rate_constant.prefactor import EventPrefactors  # noqa: E402
from pykmc.htst.constants import hz_to_thz  # noqa: E402

# Tracked Ni EAM potential + surface-hop fixture (shared with test_premin_av).
_TRACKED_POT = Path(__file__).resolve().parents[1] / "data" / "Ni_v6_2.0_LKBeland2016.eam"
_HOP_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "htst_ni100_surface_hop.npz"


class _SerialEngine:
    """Minimal engine shim exposing the .command/.lmp/.rank interface the ops use."""

    def __init__(self, lmp: object) -> None:
        self.lmp = lmp
        self.rank = 0

    def command(self, cmd: str) -> None:
        """Delegate to the underlying LAMMPS instance."""
        self.lmp.command(cmd)


def _build_lj_ni() -> _SerialEngine:
    """Create a 5-atom LJ system in a 12 Å box (no real potential file)."""
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    for cmd in [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "region box block 0 12 0 12 0 12",
        "create_box 1 box",
        # a small cluster around the center
        "create_atoms 1 single 6.0 6.0 6.0",
        "create_atoms 1 single 7.5 6.0 6.0",
        "create_atoms 1 single 6.0 7.5 6.0",
        "create_atoms 1 single 6.0 6.0 7.5",
        "create_atoms 1 single 4.5 6.0 6.0",
        "mass 1 58.69",
        "pair_style lj/cut 5.0",
        "pair_coeff 1 1 0.4 2.3",
        "run 0",
    ]:
        lmp.command(cmd)
    return _SerialEngine(lmp)


class _RC:
    """Rate-constant shim (htst style); AV-off zone covers the whole fixture."""

    style = "htst"
    free_radius = 4.0
    nu0_zone_radius = 12.0  # >= fixture extent (~9.44 A) -> zone == full system
    fd_step = 0.01
    nu0_min_THz = 1e-6
    nu0_max_THz = 1e6
    require_one_negative_mode = True
    premin = False  # pin the original path; premin behavior is owned by test_premin_av


class _Control:
    active_volume = False


class _AE:
    rcut = 3.0


class _Lammps:
    pair_style = "eam/alloy"
    pair_coeff = f"* * {_TRACKED_POT} Ni"


class _AVParams:
    ract = 6.0
    rmov = 4.0
    AV_debug = False


class _Cfg:
    """Full-config shim: the op builds its own serial engine from config.lammps."""

    def __init__(self) -> None:
        self.rateconstant = _RC()
        self.control = _Control()
        self.atomicenvironment = _AE()
        self.lammps = _Lammps()
        self.activevolume = _AVParams()


def test_get_forces_shape() -> None:
    """get_forces returns a finite (N, 3) array for all atoms."""
    eng = _build_lj_ni()
    f = ops.get_forces(eng)
    assert f.shape[1] == 3
    assert f.shape[0] >= 5
    assert np.isfinite(f).all()


@pytest.mark.skipif(not _TRACKED_POT.exists(), reason="tracked Ni EAM unavailable")
@pytest.mark.skipif(not _HOP_FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_compute_event_prefactors_runs_on_engine() -> None:
    """compute_event_prefactors runs end-to-end and returns an EventPrefactors.

    min1==saddle==min2 (all identical) so there is no real saddle — the
    orchestrator falls back gracefully, but must not raise and must return the
    dataclass with n_free >= 1. Runs on a real serial EAM engine the op builds
    itself from ``config.lammps`` (no pre-loaded system needed).
    """
    data = np.load(_HOP_FIXTURE)
    pos = data["saddle_positions"]
    move = int(data["move_atom_idx"])
    n = int(data["n_atoms"])
    eng, cell = profiling.build_serial_engine(
        pos, potential=str(_TRACKED_POT), pair_style="eam/alloy", element="Ni"
    )
    res = ops.compute_event_prefactors(
        eng,
        _Cfg(),
        central_atom_idx=move,
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        types=["Ni"] * n,
        cell=cell,
    )
    # min1==saddle==min2 (all minima) -> no real saddle -> graceful fallback,
    # but the op must run end-to-end on a real engine and return the dataclass.
    assert isinstance(res, EventPrefactors)
    assert res.n_free >= 1


_REPO_ROOT = Path(__file__).resolve().parents[2]
_POTENTIAL = _REPO_ROOT / "basin_testing" / "NiAlH_jea.eam"
_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "htst_ni100_surface_hop.npz"


class _RCXval:
    """RateConstant shim for the cross-validation (smaller free radius)."""

    style = "htst"
    free_radius = 4.0  # keep free atoms inside the 130-atom subset (EAM cutoff ~5.65 A)
    nu0_zone_radius = 12.0  # >= fixture extent -> zone == full system (canon ~12.6 THz)
    fd_step = 0.01
    nu0_min_THz = 1e-6
    nu0_max_THz = 1e6
    require_one_negative_mode = True
    premin = False  # pin the canonical ~12.6 THz path; premin owned by test_premin_av


class _ControlXval:
    active_volume = False


class _AEXval:
    rcut = 3.0


class _LammpsXval:
    pair_style = "eam/alloy"
    pair_coeff = f"* * {_POTENTIAL} Ni"


class _AVParamsXval:
    ract = 6.0
    rmov = 4.0
    AV_debug = False


class _CfgXval:
    """Full-config shim; the op builds its serial engine from config.lammps."""

    rateconstant = _RCXval()
    control = _ControlXval()
    atomicenvironment = _AEXval()
    lammps = _LammpsXval()
    activevolume = _AVParamsXval()


def _build_eam_engine(positions: np.ndarray) -> tuple[_SerialEngine, np.ndarray]:
    """Build a serial EAM-Ni engine holding positions (N,3) as type-1 Ni."""
    n = positions.shape[0]
    lo = positions.min(axis=0) - 15.0
    hi = positions.max(axis=0) + 15.0
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map array")
    lmp.command("boundary f f f")
    bounds = " ".join(f"{v:.3f}" for v in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    lmp.command(f"region box block {bounds} units box")
    lmp.command("create_box 1 box")
    lmp.create_atoms(n, None, [1] * n, positions.astype(float).reshape(-1).tolist())
    lmp.command("mass 1 58.6934")
    lmp.command("pair_style eam/alloy")
    lmp.command(f"pair_coeff * * {_POTENTIAL} Ni")
    lmp.command("run 0")
    return _SerialEngine(lmp), np.diag(hi - lo)


@pytest.mark.skipif(not _POTENTIAL.exists(), reason="NiAlH_jea.eam unavailable")
@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_ni100_surface_1nn_prefactor_matches_analysis_canon() -> None:
    """Cross-validate pyKMC nu0 for the Ni(100) surface_1NN hop vs analysis canon.

    The analysis-side toolchain (apps/PyKMC_Analysis/Analysis/HTST.md) reports
    nu0 = 13.1 THz for surface_1NN_inplane on NiAlH_jea.eam. Computing nu0 on a
    real EAM surface-hop saddle (Ea ~= 0.60 eV) through the committed engine op
    must land in the same physical window, validating the whole Vineyard path.
    """
    data = np.load(_FIXTURE)
    init = data["initial_positions"]
    sad = data["saddle_positions"]
    fin = data["final_positions"]
    move = int(data["move_atom_idx"])
    n = int(data["n_atoms"])

    eng, cell = _build_eam_engine(sad)
    res = ops.compute_event_prefactors(
        eng,
        _CfgXval(),
        central_atom_idx=move,
        min1_positions=init,
        saddle_positions=sad,
        min2_positions=fin,
        types=["Ni"] * n,
        cell=cell,
    )

    assert res.ok_forward, f"forward nu0 failed: {res.reason}"
    assert res.nu0_forward is not None
    nu0_thz = hz_to_thz(res.nu0_forward)
    # canon 13.1 THz; ~12.6 THz computed at free_radius=4 A. Window rules out
    # non-physical values while tolerating the radius/geometry difference.
    assert 8.0 <= nu0_thz <= 20.0, f"nu0={nu0_thz:.2f} THz outside physical window"
    assert res.n_free >= 5
