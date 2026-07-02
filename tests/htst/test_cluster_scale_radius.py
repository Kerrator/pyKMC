"""Cluster-scale HTST free_radius test (EXPECTED to xfail on a workstation).

A 50 Angstrom ``free_radius`` on the large Ni structure vibrates ~48,000 atoms,
so the HTST partial Hessian is a ~144,000-column ``dynamical_matrix`` on a single
``COMM_SELF`` rank (the deadlock fix) -- intractable on a local machine. This
test documents that boundary: it is marked ``xfail`` because the local
workstation is *supposed* to be unable to finish it within the budget; a cluster
node (more RAM / faster single core) would pass it (``xpass``).

It is the assertion counterpart to the radius auto-calibration probe
(``toolkit/profiling/htst_radius_probe.py``), which sweeps ``free_radius`` DOWN
from 50 A to find the largest locally-feasible vibrating region and recommends
Normand's default cell (~686 atoms; Khosravi, Song & Mousseau, PRM 7, 123605).

Slow by construction: it waits up to ``PYKMC_CLUSTER_BUDGET_S`` (default 60 s)
for the intractable radius to time out. Skips cleanly if the large structure or
the EAM potential is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("lammps")

_REPO = Path(__file__).resolve().parents[3]          # /home/kerr/pykmc
_PROBE_DIR = _REPO / "toolkit" / "profiling"
_STRUCTURE = _PROBE_DIR / "large_ni" / "initial_config.xyz"
_POTENTIAL = (
    _REPO / "pyKMC" / "benchmarks" / "Ni_fcc_32000at_4vac+4sia"
    / "Ni_v6_2.0_LKBeland2016.eam"
)

_CLUSTER_FREE_RADIUS = 50.0  # the spec's "start at 50 A, it should fail" radius
_BUFFER = 6.5                # nu0_zone_radius = free_radius + buffer
_LOCAL_BUDGET_S = float(os.environ.get("PYKMC_CLUSTER_BUDGET_S", "60"))


@pytest.mark.skipif(not _STRUCTURE.exists(),
                    reason="large Ni structure absent (run toolkit/profiling/gen_large_ni.py)")
@pytest.mark.skipif(not _POTENTIAL.exists(), reason="Ni EAM potential absent")
@pytest.mark.xfail(
    reason="free_radius=50 A (~48k vibrating atoms, single-rank ~144k-column Hessian) "
           "is cluster-scale; a local workstation cannot finish it within the budget",
    strict=False,
)
def test_cluster_scale_free_radius_50_is_local_intractable() -> None:
    """free_radius=50 A Hessian must complete within the local budget -- expected to xfail."""
    sys.path.insert(0, str(_PROBE_DIR))
    import htst_radius_probe as probe

    elapsed, feasible = probe.measure_hessian(
        _STRUCTURE, radius=_CLUSTER_FREE_RADIUS + _BUFFER,
        free_radius=_CLUSTER_FREE_RADIUS,
        potential=_POTENTIAL, budget_s=_LOCAL_BUDGET_S,
    )
    assert feasible, (
        f"free_radius={_CLUSTER_FREE_RADIUS:.0f} A Hessian did not complete within "
        f"{_LOCAL_BUDGET_S:.0f}s on this machine (elapsed={elapsed}); "
        "cluster-scale as expected"
    )
