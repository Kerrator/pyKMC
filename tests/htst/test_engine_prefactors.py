"""Serial-LAMMPS smoke tests for the HTST engine ops (no MPI, no potential file).

Tests
-----
- test_get_forces_shape: gather_atoms("f") returns a finite (N,3) array.
- test_compute_event_prefactors_runs_on_engine: end-to-end call on a real
  LAMMPS engine returns an EventPrefactors dataclass without raising.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lammps")

from dataclasses import dataclass  # noqa: E402

from ase.cell import Cell  # noqa: E402

from pykmc.engine.lammps import LammpsEngine  # noqa: E402
from pykmc.htst import profiling  # noqa: E402
from pykmc.htst.constants import hz_to_thz  # noqa: E402
from pykmc.htst.lammps_extension import HtstLammpsExtension  # noqa: E402
from pykmc.rate_constant.prefactor import EventPrefactors  # noqa: E402


@dataclass
class _LjConfig:
    """LammpsEngine config for a 5-atom LJ cluster (no real potential file)."""

    pair_style: str = "lj/cut 5.0"
    pair_coeff: str = "1 1 0.4 2.3"
    min_style: str = "cg"
    minimize: str = "1.0e-6 1.0e-8 1000 1000"
    frz_min: str = "1.0e-6 1.0e-8 10 10"
    verbosity: int = 0


def _build_lj_ni() -> LammpsEngine:
    """Create a 5-atom LJ system in a 12 Å box on a serial LammpsEngine + extension."""
    positions = np.array(
        [
            [6.0, 6.0, 6.0],  # a small cluster around the center
            [7.5, 6.0, 6.0],
            [6.0, 7.5, 6.0],
            [6.0, 6.0, 7.5],
            [4.5, 6.0, 6.0],
        ]
    )
    engine = LammpsEngine(config=_LjConfig())
    engine.start()
    engine.initialize_parameters()
    engine.initialize_system(
        types=["Ni"] * 5,
        positions=positions,
        cell=Cell(np.diag([12.0, 12.0, 12.0])),
        pbc=[True, True, True],
    )
    engine.initialize_potential()
    engine.command("run 0")
    HtstLammpsExtension(engine)
    return engine


class _RC:
    style = "htst"
    free_radius = 5.0
    fd_step = 0.01
    nu0_min_THz = 1e-6
    nu0_max_THz = 1e6
    require_one_negative_mode = True
    premin = False  # pin the original path; premin behavior is owned by test_premin_av


class _Cfg:
    rateconstant = _RC()


def test_get_forces_shape() -> None:
    """get_forces returns a finite (N, 3) array for all atoms."""
    eng = _build_lj_ni()
    f = eng.get_forces()
    assert f.shape[1] == 3
    assert f.shape[0] >= 5
    assert np.isfinite(f).all()


def test_compute_event_prefactors_runs_on_engine() -> None:
    """compute_event_prefactors runs end-to-end and returns an EventPrefactors.

    min1==saddle==min2 (all identical) so there is no real saddle — the
    orchestrator falls back gracefully, but must not raise and must return
    the correct dataclass with n_free >= 1.
    """
    eng = _build_lj_ni()
    pos = eng.get_positions()
    cell = np.diag([12.0, 12.0, 12.0])
    res = eng.compute_event_prefactors(
        _Cfg(),
        central_atom_idx=0,
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        types=["Ni"] * pos.shape[0],
        cell=cell,
    )
    # min1==saddle==min2 (all minima) -> no real saddle -> graceful fallback,
    # but the op must run end-to-end on a real engine and return the dataclass.
    assert isinstance(res, EventPrefactors)
    assert res.n_free >= 1


_REPO_ROOT = Path(__file__).resolve().parents[2]
_POTENTIAL = _REPO_ROOT / "basin_testing" / "NiAlH_jea.eam"
_FIXTURE = Path(__file__).resolve().parent / "data" / "htst_ni100_surface_hop.npz"


class _RCXval:
    """RateConstant shim for the cross-validation (smaller free radius)."""

    style = "htst"
    free_radius = 4.0  # keep free atoms inside the 130-atom subset (EAM cutoff ~5.65 A)
    fd_step = 0.01
    nu0_min_THz = 1e-6
    nu0_max_THz = 1e6
    require_one_negative_mode = True
    premin = False  # pin the canonical ~12.6 THz path; premin owned by test_premin_av


class _CfgXval:
    """Config shim exposing only rateconstant."""

    rateconstant = _RCXval()


def _build_eam_engine(positions: np.ndarray) -> tuple[LammpsEngine, np.ndarray]:
    """Build a serial EAM-Ni engine holding positions (N,3) as type-1 Ni."""
    return profiling.build_serial_engine(
        positions, potential=str(_POTENTIAL), pair_style="eam/alloy", element="Ni"
    )


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
    res = eng.compute_event_prefactors(
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
