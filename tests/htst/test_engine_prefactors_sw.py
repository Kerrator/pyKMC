"""SW-Si prefactors on the serial HTST engine (no MPI, no pARTn search).

``compute_event_prefactors`` rebuilds its nu0 zone through
``activevolume.active_volume.reset``, which used to emit no ``mass`` command.
``pair_style eam/alloy`` reads masses from the setfl file, so this was
invisible on every EAM system; ``sw`` (like ``tersoff`` and ``mlip``) does
not, LAMMPS aborted the zone's first rebalance with "Not all per-type masses
are set", and because the op never raises every SW-Si event silently fell
back to ``k0`` (``nu0 = None`` on every reference row).

Both tests run the production op on the real serial ``COMM_SELF`` LAMMPS it
builds itself from ``config.lammps``:

- test_sw_si_zone_builds_with_masses: identical geometries (no real saddle)
  on a small diamond cell; the op must get past the zone rebuild
  (``n_free >= 1``) and its fallback reason must not be the mass abort.
- test_sw_si_vacancy_hop_has_nu0: a pARTn-found symmetric monovacancy hop in
  a 6x6x6 SW-Si cell (an ``event_geometry_output`` dump) must yield a finite
  nu0 in both directions inside the configured window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

pytest.importorskip("lammps")

from pykmc.enginemanager.lmpi import lammps_operations as ops  # noqa: E402
from pykmc.htst.constants import hz_to_thz  # noqa: E402
from pykmc.rate_constant.prefactor import EventPrefactors  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_SI_SW = _DATA / "Si.sw"
_HOP_FIXTURE = _DATA / "htst_si_vacancy_hop.npz"


class _StubEngine:
    """The op reads only ``rank`` / ``engine_id``; the Hessian LAMMPS is its own."""

    rank = 0
    engine_id = 0


class _RC:
    """Rate-constant shim; ``free_radius``/``nu0_zone_radius`` set per test."""

    style = "htst"
    fd_step = 0.01
    nu0_min_THz = 0.01
    nu0_max_THz = 1000.0
    require_one_negative_mode = True
    premin = False

    def __init__(self, free_radius: float, nu0_zone_radius: float) -> None:
        self.free_radius = free_radius
        self.nu0_zone_radius = nu0_zone_radius


class _Control:
    active_volume = False


class _AE:
    rcut = 6.5


class _Lammps:
    pair_style = "sw"
    pair_coeff = f"* * {_SI_SW} Si"


class _AVParams:
    ract = 6.0
    rmov = 4.0
    AV_debug = False


class _Cfg:
    """Full-config shim: the op builds its own serial engine from config.lammps.

    The zone buffer (``nu0_zone_radius - free_radius``) is kept above the SW
    cutoff (1.80 * 2.0951 = 3.77 Angstrom) so every free atom sees a complete
    frozen surrounding.
    """

    def __init__(self, free_radius: float, nu0_zone_radius: float) -> None:
        self.rateconstant = _RC(free_radius, nu0_zone_radius)
        self.control = _Control()
        self.atomicenvironment = _AE()
        self.lammps = _Lammps()
        self.activevolume = _AVParams()


def _run(cfg: _Cfg, **geometry: object) -> EventPrefactors:
    """Call the production op on a fresh stub, closing its serial LAMMPS after."""
    eng = _StubEngine()
    try:
        return ops.compute_event_prefactors(eng, cfg, **geometry)
    finally:
        serial = getattr(eng, "_serial_hessian_engine", None)
        if serial is not None:
            serial.close()


@pytest.mark.skipif(not _SI_SW.exists(), reason="tracked Si.sw unavailable")
def test_sw_si_zone_builds_with_masses() -> None:
    """The SW-Si nu0 zone rebuild survives its first rebalance.

    min1 == saddle == min2, so there is no real saddle and the op falls back
    gracefully; but it must get past ``reset``/``redefine_atoms`` (``n_free``
    reported) and the reason must not be LAMMPS's missing-mass abort.
    """
    atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True)
    atoms = atoms.repeat([3, 3, 3])
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    res = _run(
        _Cfg(free_radius=4.0, nu0_zone_radius=8.0),
        central_atom_idx=0,
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        types=["Si"] * len(pos),
        cell=cell,
    )
    assert isinstance(res, EventPrefactors)
    assert "mass" not in res.reason.lower(), res.reason
    assert res.n_free >= 1


@pytest.mark.skipif(not _SI_SW.exists(), reason="tracked Si.sw unavailable")
@pytest.mark.skipif(not _HOP_FIXTURE.exists(), reason="SW-Si hop fixture unavailable")
def test_sw_si_vacancy_hop_has_nu0() -> None:
    """A real SW-Si vacancy hop gets a finite nu0 in both directions.

    The fixture is the symmetric monovacancy migration event (0.509 eV each
    way, NEB-validated) pARTn found in a 6x6x6 SW-Si cell. With the
    validated ``free_radius = 6`` / ``nu0_zone_radius = 10`` zone the forward
    prefactor is 23.6 THz (43 free atoms); the pin is loose because only the
    fallback (``None``) is the bug being guarded.
    """
    d = np.load(_HOP_FIXTURE)
    res = _run(
        _Cfg(free_radius=6.0, nu0_zone_radius=10.0),
        central_atom_idx=int(d["central_atom_idx"]),
        min1_positions=d["min1_positions"],
        saddle_positions=d["saddle_positions"],
        min2_positions=d["min2_positions"],
        types=[str(t) for t in d["types"]],
        cell=d["cell"],
    )
    assert res.nu0_forward is not None, res.reason
    assert res.nu0_backward is not None, res.reason
    assert res.ok_forward and res.ok_backward
    assert res.n_free >= 1
    for nu0 in (res.nu0_forward, res.nu0_backward):
        assert 0.01 <= hz_to_thz(nu0) <= 1000.0
    assert hz_to_thz(res.nu0_forward) == pytest.approx(23.6, rel=0.05)
