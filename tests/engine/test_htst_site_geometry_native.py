"""F1: exercise full-saddle transport through refinement and native site HTST.

Promoted unchanged in substance from the independent review's red-to-green
pack (``test_site_geometry_native.py``); every assertion and tolerance is the
pack's. Only the expensive pARTn/PSR producer and scheduling estimates are
substituted. ``Refinement.execute``, result transport, active-table updates,
service, scratch LAMMPS Hessians and numerical prefactors all execute for
real.

Adapters: data paths resolve from the repository root; the serial-only guard
is the engine tests' ``pytest.skip`` under ``mpirun`` (the pack asserted a
single rank instead).
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

from pykmc.config import Config, RateConstantConfig
from pykmc.engine.htst_lammps import LammpsHTSTExtension
from pykmc.engine.lammps import LammpsEngine
from pykmc.event_table import ActiveEventTable
from pykmc.htst import select_free_indices
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.refinement import Refinement
from pykmc.result import EventRefinementOutput, Ok
from tests.engine.test_htst_lammps import _SWConfig, _initialize
from tests.lifecycle.conftest import FakeManager

_ROOT = Path(__file__).resolve().parents[2]
_HOP_FIXTURE = _ROOT / "tests" / "data" / "htst_si_vacancy_hop.npz"
_INPUT = _ROOT / "tests" / "data" / "input.in"


@pytest.fixture(autouse=True)
def require_serial() -> None:
    """Skip under ``mpirun``: the native lane uses a ``COMM_SELF`` engine."""
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial tests must run without mpirun")


@pytest.fixture
def native_case() -> Any:
    """Use one geometry/settings combination for both sides of the oracle."""
    with np.load(_HOP_FIXTURE, allow_pickle=False) as data:
        hop = {key: np.array(data[key]) for key in data.files}
    center = int(hop["central_atom_idx"])
    types = tuple(str(t) for t in hop["types"])
    system = SimpleNamespace(
        positions=hop["min1_positions"],
        types=types,
        cell=hop["cell"],
        pbc=(True, True, True),
        # Refinement records stable crop identities from the source index.
        index=np.arange(len(hop["min1_positions"])),
    )
    engine = LammpsEngine(_SWConfig(), comm=MPI.COMM_SELF)
    engine.start()
    try:
        _initialize(engine, types, system.positions, system.cell)
        extension = LammpsHTSTExtension(engine)
        base = Config.from_ini_file(str(_INPUT))
        # The service must describe the potential the engine actually runs
        # (SW-Si), not the committed input's EAM-Ni: the extension's
        # force-model pre-check compares the two before any Hessian.
        config = base.model_copy(
            update={
                "rateconstant": RateConstantConfig(
                    style="htst", free_radius=6.0, premin=False, k0=1.0
                ),
                "control": base.control.model_copy(update={"active_volume": False}),
                "lammps": base.lammps.model_copy(
                    update={
                        "pair_style": engine.config.pair_style,
                        "pair_coeff": engine.config.pair_coeff,
                        "min_style": engine.config.min_style,
                        "minimize": engine.config.minimize,
                        "frz_min": engine.config.frz_min,
                    }
                ),
            }
        )
        manager = FakeManager(extension.compute_event_prefactors)
        service = PrefactorService(
            config, manager, create_rate_constant(config.rateconstant)
        )
        request = service.build_request(
            event_key=("full-geometry-oracle",),
            min1_positions=system.positions,
            saddle_positions=hop["saddle_positions"],
            min2_positions=hop["min2_positions"],
            types=types,
            cell=system.cell,
            pbc=system.pbc,
            center_index=center,
        )
        expected = extension.compute_event_prefactors(request).forward
        assert expected.ok, expected.reason
        assert 1e12 < expected.nu0_hz < 100e12
        yield SimpleNamespace(
            hop=hop,
            center=center,
            system=system,
            config=config,
            manager=manager,
            service=service,
            request=request,
            expected=expected,
        )
    finally:
        engine.close()


def test_native_full_geometry_service_positive_control(native_case: Any) -> None:
    """The service preserves a valid estimate when given the full geometry."""
    case = native_case
    actual = case.service.compute([case.request])[case.request.event_key].forward
    assert actual.ok, actual.reason
    assert actual.nu0_hz == pytest.approx(case.expected.nu0_hz, rel=1e-8)


def test_full_refined_saddle_reaches_native_site_estimate(
    native_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correct site estimate must survive the real refinement crop boundary."""
    case = native_case
    crop = select_free_indices(
        case.system.positions, case.center, 6.3, case.system.cell, case.system.pbc
    )
    outside = np.setdiff1d(np.arange(len(case.system.positions)), crop)
    assert (
        np.linalg.norm(
            case.hop["saddle_positions"][outside] - case.system.positions[outside]
        )
        > 1e-5
    ), "Fixture must include refined motion outside the classification crop"
    neighbors = SimpleNamespace(get_neighbors=lambda key, atom: crop)
    logs = SimpleNamespace(info=lambda *args: None, progress_bar=lambda *args: None)
    environment = SimpleNamespace(get_atoms_with_id=lambda event_id: [case.center])
    refinement = Refinement(
        case.config, logs, case.system, neighbors, environment, case.manager
    )
    monkeypatch.setattr(refinement, "get_total_refinements_todo", lambda rows: (1, 1.0))
    monkeypatch.setattr(refinement, "get_energy_thr_refine", lambda rows, ktot: 1.0)

    # Deliberately distinct: blindly retaining the reference must not pass a
    # full-data transport test. Cropped-only fallback has its own analytic test.
    inherited_hz = 2e12
    assert abs(case.expected.nu0_hz / inherited_hz - 1) > 0.1

    def full_saddle_producer(
        atom: int, reference: Any, energy: float, contexts: dict, threshold: float
    ) -> Future:
        future: Future = Future()
        future.set_result(
            Ok(
                EventRefinementOutput(
                    central_atom_index=atom,
                    saddle_positions=case.hop["saddle_positions"].copy(),
                    E_saddle=0.5089,
                    refined="T",
                )
            )
        )
        contexts[future] = {
            "min2_positions": case.hop["min2_positions"][crop].copy(),
            "num_reference_event": 0,
            "neighbors": crop,
            "reference_energy_barrier": 0.5089,
            "estimate": {
                "nu0_hz": inherited_hz,
                "nu0_status": "ok",
                "nu0_reason": "",
                "nu0_source": "reference",
            },
        }
        return future

    monkeypatch.setattr(refinement, "refine_single", full_saddle_producer)
    references = pd.DataFrame(
        [
            {
                "idx_ref": 0,
                "event_id": "fixture",
                "nu0_status": "ok",
            }
        ]
    )
    refinement.execute(references, total_energy=0.0)
    assert len(refinement.results) == 1 and refinement.results[0].is_ok()
    active = ActiveEventTable(case.config, prefactor_service=case.service)
    active.add_events(refinement.results[0].ok_value())
    active.request_site_prefactors(case.system, neighbors)
    row = active.table.iloc[0]
    assert row["nu0_status"] == "ok"
    assert row["nu0_source"] == "site", (
        "Full refined geometry must enable a site estimate"
    )
    assert row["nu0"] == pytest.approx(case.expected.nu0_hz, rel=1e-8), (
        "Full refined geometry was altered before the site Hessian: "
        f"expected={case.expected.nu0_hz}, actual={row['nu0']}"
    )
