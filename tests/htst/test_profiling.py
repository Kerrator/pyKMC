"""Guardrail for the HTST prefactor profiling harness (eskm vs Python-FD).

Runs both Hessian modes through :func:`pykmc.htst.profiling.time_event` on the
committed Ni(100) surface-hop fixture and asserts (a) the two modes agree on the
forward Vineyard nu0 to <1% (the validated agreement is <0.5%), (b) the FD mode
makes exactly 2*3*n_free force calls per Hessian, and (c) the per-row round-trip
accounting matches the mode (1 for eskm, 2*3*n_free for FD). The absolute
8-20 THz window is asserted only when the NiAlH potential (the one the fixture
canon was computed with) is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lammps")

from pykmc.htst import profiling  # noqa: E402
from pykmc.htst.constants import hz_to_thz  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NIALH = _REPO_ROOT / "basin_testing" / "NiAlH_jea.eam"
_NI_V6 = _REPO_ROOT / "tests" / "data" / "Ni_v6_2.0_LKBeland2016.eam"
_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "htst_ni100_surface_hop.npz"

_FREE_RADIUS = 4.0  # keep free atoms inside the 130-atom subset (EAM cutoff ~5.65 A)
_FD_STEP = 0.01


def _pick_potential() -> tuple[Path, bool]:
    """Return (potential path, is_nialh); prefer the fixture's canonical NiAlH."""
    if _NIALH.exists():
        return _NIALH, True
    return _NI_V6, False


@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_modes_agree_and_count_round_trips() -> None:
    """Cross-check eskm vs FD: same forward nu0 (<1%) with correct call accounting."""
    potential, is_nialh = _pick_potential()
    data = np.load(_FIXTURE)
    init = data["initial_positions"]
    sad = data["saddle_positions"]
    fin = data["final_positions"]
    move = int(data["move_atom_idx"])
    n = int(data["n_atoms"])
    types = ["Ni"] * n
    pbc = np.array([True, True, True])

    engine, cell = profiling.build_serial_engine(
        sad, potential=str(potential), pair_style="eam/alloy", element="Ni"
    )

    rows_eskm = profiling.time_event(
        engine,
        mode="eskm",
        min1=init,
        saddle=sad,
        min2=fin,
        types=types,
        central_index=move,
        free_radius=_FREE_RADIUS,
        fd_step=_FD_STEP,
        cell=cell,
        pbc=pbc,
    )
    rows_fd = profiling.time_event(
        engine,
        mode="fd",
        min1=init,
        saddle=sad,
        min2=fin,
        types=types,
        central_index=move,
        free_radius=_FREE_RADIUS,
        fd_step=_FD_STEP,
        cell=cell,
        pbc=pbc,
    )

    # one row per Hessian (min1, saddle, min2)
    assert len(rows_eskm) == 3
    assert len(rows_fd) == 3

    # FD call accounting: 2 force evaluations per DOF, 3 DOF per free atom
    for row in rows_fd:
        assert row.n_calls == 2 * 3 * row.n_free
        assert row.round_trips == row.n_calls
        assert row.t_hessian > 0.0
        assert row.t_forces_total is not None and row.t_forces_total > 0.0

    # eskm: one engine command round-trip per Hessian, with segment timings
    for row in rows_eskm:
        assert row.round_trips == 1
        assert row.t_hessian > 0.0
        assert row.t_dynmat_cmd is not None and row.t_dynmat_cmd > 0.0
        assert row.t_file_read is not None and row.t_file_read > 0.0

    # cross-mode accuracy gate (forward nu0, both finite)
    nu0_eskm = rows_eskm[0].nu0_hz
    nu0_fd = rows_fd[0].nu0_hz
    assert nu0_eskm is not None, "eskm produced no forward nu0"
    assert nu0_fd is not None, "FD produced no forward nu0"
    rel_err = abs(nu0_eskm - nu0_fd) / nu0_fd
    assert rel_err < 0.01, f"modes disagree: rel_err={rel_err:.4f}"

    # absolute canon (NiAlH-specific: ~12.6 THz at free_radius=4.0)
    if is_nialh:
        nu0_thz = hz_to_thz(nu0_eskm)
        assert 8.0 <= nu0_thz <= 20.0, f"nu0={nu0_thz:.2f} THz outside physical window"
