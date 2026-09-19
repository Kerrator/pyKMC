"""Behavioural regressions for review findings F1, F2 and F3 (promoted pack).

Promoted unchanged in substance from the independent review's red-to-green
pack (``test_lifecycle_contracts.py``); every assertion, tolerance and
accepted-policy branch is the pack's. These tests use real table admission,
graph/IRA identity, request construction and HTST mathematics; only the
worker/Hessian provider is analytic. No LAMMPS instance or MPI pool is
created. Config fixtures resolve from the imported checkout.

The F1 test deliberately supplies only a crop: safe reference retention is
an acceptable result. The separate native module
(``tests/engine/test_htst_site_geometry_native.py``) proves that full
geometries survive ``Refinement.execute`` and yield a site estimate.

Cache-policy choices accepted here: reject an incompatible restart
explicitly, or invalidate its estimates and use ``k0``. Retaining obsolete
finite frequencies as historical data also requires retaining their original
settings metadata. If the implementation adopts explicit per-row provenance
instead, extend that assertion to verify the new documented schema rather
than removing it.

Adapter: the pack's ``AnalyticManager`` is the repository ``FakeManager``
(same ``submit`` and ``prefactor_requests`` surface).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.htst.prefactor import compute_event_prefactors
from pykmc.htst.result import DirectionalPrefactor
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventRefinementOutput, EventSearchOutput
from tests.lifecycle.conftest import FakeManager

import pykmc

from .protocol_producers import (
    archived_frequency,
    protocol_event_prefactors,
    protocol_table,
)


def config(path: str | None = None, **settings: Any) -> Config:
    """Return the committed test input in htst style with ``k0 = 1``.

    Parameters
    ----------
    path : str or None, optional
        Reference table to load through ``control.reference_table``.
    **settings : Any
        Extra ``RateConstantConfig`` fields.

    Returns
    -------
    Config
        The adjusted configuration.

    """
    checkout = Path(pykmc.__file__).resolve().parent.parent
    cfg = Config.from_ini_file(str(checkout / "tests/data/input.in"))
    return cfg.model_copy(
        update={
            "rateconstant": RateConstantConfig(style="htst", k0=1.0, **settings),
            "control": cfg.control.model_copy(update={"reference_table": path}),
        }
    )


def row(barrier: float = 0.5) -> pd.Series:
    """Return one pending same-topology reference row with a three-atom crop."""
    saddle = np.array([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.5, 0.0]])
    return pd.Series(
        {
            "idx_ref": -1,
            "event_id": "same-topology",
            "initial_positions": saddle - [0.1, 0.0, 0.0],
            "saddle_positions": saddle.copy(),
            "final_positions": saddle + [0.1, 0.0, 0.0],
            "types": ["Ni", "Ni", "Ni"],
            "energy_barrier": barrier,
            "k": 0.0,
            "id_saddle": "saddle",
            "id_final": "same-topology",
            "move_atom_idx": 0,
            "sym_matrix": [np.eye(3)],
            "sym_perm": [np.arange(3)],
            "idx_backward": -1,
            "dra": 0.1,
            "k_prefactor": 1.0,
            "nu0": np.nan,
            "nu0_status": "pending",
            "nu0_reason": "",
        }
    )


def saved_table(
    tmp_path: Path, nu0: float = 20e12, *, incomplete: bool = False
) -> Path:
    """Save a declared synthetic full-request protocol result for policy tests.

    The three-row geometry is explicitly the whole unit fixture. Frequencies
    are supplied unit values; the separate analytic catalogue oracle computes
    its own numerical estimates. Incomplete variants declare one unavailable
    original source row and must never support speculative recomputation.
    """
    table = protocol_table(config(free_radius=6.0, fd_step=0.01))
    item = row()
    table.add(item.to_frame().T)
    request = table.prefactor_service.build_request(
        event_key=("synthetic-policy-row", 0),
        min1_positions=item["initial_positions"],
        saddle_positions=item["saddle_positions"],
        min2_positions=item["final_positions"],
        types=item["types"],
        cell=np.eye(3) * 10,
        pbc=(False,) * 3,
        center_index=0,
    )
    if incomplete:
        request = replace(
            request,
            constraints=replace(request.constraints, source_ids=(0, 1, 2, 3)),
            user_constraints=None,
        )
    estimate = DirectionalPrefactor.accepted(
        nu0,
        n_free=3,
        n_positive_min=9,
        n_negative_saddle=1,
    )
    result = protocol_event_prefactors(request, estimate, estimate)
    table._patch_row(
        0, result.forward, calculation=result.calculation("forward"), fresh=True
    )
    assert table.table.iloc[0]["nu0_status"] == "ok"
    assert result.provenance.source.is_complete is (not incomplete)
    path = tmp_path / "original.pickle"
    table.save(str(path))
    return path


def load_or_explicit_rejection(
    path: Path, diagnostic_terms: tuple[str, ...], **settings: Any
) -> ReferenceEventTable | None:
    """Permit a deliberate compatibility error, never an arbitrary failure."""
    try:
        return protocol_table(config(str(path), **settings))
    except ValueError as exc:
        message = str(exc).lower()
        assert any(term in message for term in diagnostic_terms), (
            f"Restart rejection must identify the incompatible setting; received: {exc}"
        )
        assert any(
            term in message
            for term in (
                "incompatib",
                "mismatch",
                "differ",
                "changed",
                "invalid",
                "stale",
                "outside",
                "reject",
                "exceed",
                "window",
            )
        ), f"Not an explicit cache-compatibility rejection: {exc}"
        return None


def assert_invalidated_fallback(table: ReferenceEventTable) -> None:
    """Assert the cached row is no longer accepted and sits on the ``k0`` fallback."""
    cached = table.table.iloc[0]
    assert cached["nu0_status"] != "ok", (
        "An incompatible cached estimate is still accepted without recalculation"
    )
    reason = cached["nu0_reason"]
    assert isinstance(reason, str) and reason.strip(), (
        "Invalidation must retain a useful reason"
    )
    assert float(cached["k_prefactor"]) == pytest.approx(1.0)
    fallback = create_rate_constant(table.config.rateconstant).compute_rate(
        float(cached["energy_barrier"])
    )
    assert float(cached["k"]) == pytest.approx(fallback.rate)


@pytest.mark.parametrize(
    "changed",
    [
        {"free_radius": 12.0},
        {"fd_step": 0.02},
        {"zone_radius": 8.0},
        {"premin": True},
    ],
    ids=["free-radius", "fd-step", "zone-radius", "premin"],
)
def test_changed_settings_do_not_reuse_or_relabel_cached_spectra(
    tmp_path: Path, changed: dict[str, Any]
) -> None:
    """F2: a changed kernel setting invalidates the cached estimate truthfully."""
    path = saved_table(tmp_path, incomplete=True)
    original = pd.read_pickle(path)
    settings = {"free_radius": 6.0, "fd_step": 0.01, **changed}
    loaded = load_or_explicit_rejection(path, tuple(changed), **settings)
    if loaded is None:
        return
    # No service was attached, so no scientific recalculation can have occurred.
    assert_invalidated_fallback(loaded)
    assert archived_frequency(loaded, 0, 20e12)
    assert not loaded.prefactor_service.manager.prefactor_requests
    resaved = tmp_path / "resaved.pickle"
    loaded.save(str(resaved))
    after = pd.read_pickle(resaved)
    retained = pd.to_numeric(after["nu0"], errors="coerce").notna().any()
    if retained:
        assert after.attrs["settings"] == original.attrs["settings"], (
            "Unrecomputed retained frequencies now claim different settings; "
            "clear obsolete nu0 or preserve its actual provenance"
        )
    assert after.iloc[0]["nu0_status"] != "ok"
    assert float(after.iloc[0]["k_prefactor"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "window",
    [
        {"nu0_max_THz": 10.0},
        {"nu0_min_THz": 30.0},
    ],
    ids=["above-maximum", "below-minimum"],
)
def test_reload_rechecks_current_acceptance_window(
    tmp_path: Path, window: dict[str, float]
) -> None:
    """F2: both bounds of the current window apply to a reloaded estimate."""
    path = saved_table(tmp_path, nu0=20e12)
    loaded = load_or_explicit_rejection(
        path,
        ("nu0", "prefactor", "frequency", "window"),
        free_radius=6.0,
        fd_step=0.01,
        **window,
    )
    if loaded is None:
        return
    assert_invalidated_fallback(loaded)
    assert archived_frequency(loaded, 0, 20e12)
    assert not loaded.prefactor_service.manager.prefactor_requests


def test_same_settings_reload_preserves_accepted_frequency_and_provenance(
    tmp_path: Path,
) -> None:
    """Positive control: rejecting or discarding every cache is not a fix."""
    path = saved_table(tmp_path, nu0=20e12)
    original = pd.read_pickle(path)
    loaded = protocol_table(config(str(path), free_radius=6.0, fd_step=0.01))
    cached = loaded.table.iloc[0]
    assert cached["nu0_status"] == "ok"
    assert float(cached["nu0"]) == pytest.approx(20e12)
    assert float(cached["k_prefactor"]) == pytest.approx(20.0)
    assert float(cached["k"]) == pytest.approx(float(original.iloc[0]["k"]))
    resaved = tmp_path / "same-settings.pickle"
    loaded.save(str(resaved))
    assert pd.read_pickle(resaved).attrs["settings"] == original.attrs["settings"]


def test_crop_only_refinement_does_not_overwrite_valid_reference_with_fake_geometry() -> (
    None
):
    """F1: a crop-only refined row keeps its valid inherited estimate."""
    cfg = config(free_radius=0.5)
    positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    saddle = np.array([[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]])
    final = np.array([[0.2, 0.0, 0.0], [1.3, 0.0, 0.0]])
    system = SimpleNamespace(
        positions=positions,
        types=["Ni", "Ni"],
        cell=np.eye(3) * 10,
        pbc=[False] * 3,
        index=np.arange(2),
    )
    neighbors = SimpleNamespace(get_neighbors=lambda key, atom: np.array([0]))

    def hessian(points: np.ndarray, free: np.ndarray) -> np.ndarray:
        assert list(free) == [0]
        if np.isclose(points[0, 0], 0.1):
            return np.diag([-1.0, 2.0 * points[1, 0], 2.0])
        return np.eye(3) * 2.0

    manager = FakeManager(lambda req: compute_event_prefactors(req, hessian))
    service = PrefactorService(
        cfg, manager, create_rate_constant(cfg.rateconstant), method="fd"
    )
    request = service.build_request(
        event_key=("full-geometry-reference",),
        min1_positions=positions,
        saddle_positions=saddle,
        min2_positions=final,
        types=system.types,
        cell=system.cell,
        pbc=system.pbc,
        center_index=0,
    )
    expected = compute_event_prefactors(request, hessian).forward
    assert expected.ok
    active = ActiveEventTable(cfg, prefactor_service=service)
    active.add_events(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=saddle[[0]],
            E_saddle=0.5,
            min2_positions=final[[0]],
            dE_forward=0.5,
            num_reference_event=0,
            refined="T",
            nu0_hz=expected.nu0_hz,
            nu0_status="ok",
            crop_atom_ids=(0,),
        )
    )
    active.request_site_prefactors(system, neighbors)
    result = active.table.iloc[0]
    assert float(result["nu0"]) == pytest.approx(expected.nu0_hz, rel=1e-12), (
        "A crop-only event silently replaced the correct inherited frequency"
    )
    # No producer supplied the outer saddle coordinate: synthesizing it from
    # the minimum must not be accepted as a measured full stationary geometry.
    assert not manager.prefactor_requests, (
        "Insufficient stationary geometry must retain the reference estimate, "
        "not dispatch a fabricated full-system Hessian request"
    )
    assert result["nu0_source"] == "reference"
    assert float(result["k_prefactor"]) == pytest.approx(expected.nu0_hz / 1e12)


def admit_with_analytic_hessian(
    initial: np.ndarray,
    saddle: np.ndarray,
    final: np.ndarray,
    forward: float = 0.5,
    backward: float = 0.7,
) -> ReferenceEventTable:
    """Admit one search through the real gate with an analytic Hessian provider."""
    cfg = config()

    def hessian(points: np.ndarray, free: np.ndarray) -> np.ndarray:
        diagonal = np.full(3 * len(free), 2.0)
        if np.allclose(points, saddle):
            diagonal[0] = -1.0
        return np.diag(diagonal)

    manager = FakeManager(lambda req: compute_event_prefactors(req, hessian))
    service = PrefactorService(
        cfg, manager, create_rate_constant(cfg.rateconstant), method="fd"
    )
    table = ReferenceEventTable(cfg, prefactor_service=service)
    search = EventSearchOutput(
        central_atom_index=0,
        min1_positions=initial,
        saddle_positions=saddle,
        min2_positions=final,
        dE_forward=forward,
        dE_backward=backward,
        move_atom_index=0,
        cell=np.eye(3) * 10,
        types=["Ni"] * len(saddle),
    )
    results = table.add_events([search], pbc=[True] * 3)
    assert results[0].is_ok()
    assert len(manager.prefactor_requests) == 1
    assert (table.table["nu0_status"] == "ok").all()
    return table


def test_unequal_barriers_keep_both_directions_even_with_equal_prefactors() -> None:
    """F3: equal curvatures never delete a physically different reverse barrier."""
    saddle = np.array(
        [[1.1, 1.0, 1.0], [3.0, 1.0, 1.0], [1.0, 3.5, 1.0], [1.0, 1.0, 4.0]]
    )
    initial, final = saddle.copy(), saddle.copy()
    initial[0, 0] -= 0.1
    final[0, 0] += 0.1
    table = admit_with_analytic_hessian(initial, saddle, final)
    assert len(table.table) == 2, (
        "Equal curvatures do not justify deleting a different reverse barrier; "
        f"remaining barriers={list(table.table['energy_barrier'])}"
    )
    frame = table.table.sort_values("energy_barrier")
    assert frame["energy_barrier"].to_numpy(dtype=float) == pytest.approx([0.5, 0.7])
    assert float(frame.iloc[0]["nu0"]) == pytest.approx(float(frame.iloc[1]["nu0"]))
    identities = [int(v) for v in frame["idx_ref"]]
    assert [int(v) for v in frame["idx_backward"]] == identities[::-1]
    assert float(frame.iloc[0]["k"]) > float(frame.iloc[1]["k"])


def test_symmetric_equal_barrier_event_remains_usable() -> None:
    """A proper 180-degree rotation maps the two minima and species exactly.

    Permit either conservative reciprocal rows or proven self-collapse; both
    represent the same physical rates. Rejecting all symmetric events fails.
    """
    saddle = np.array(
        [
            [5.0, 5.0, 5.0],
            [5.0, 7.0, 5.0],
            [5.0, 3.0, 5.0],
            [5.0, 5.0, 7.0],
            [5.0, 5.0, 3.0],
        ]
    )
    initial, final = saddle.copy(), saddle.copy()
    initial[0, 0] -= 0.1
    final[0, 0] += 0.1
    rotation = np.diag([-1.0, 1.0, -1.0])
    permutation = [0, 1, 2, 4, 3]
    assert np.allclose((initial - saddle[0]) @ rotation + saddle[0], final[permutation])
    table = admit_with_analytic_hessian(initial, saddle, final, backward=0.5)
    assert len(table.table) in (1, 2)
    assert table.table["energy_barrier"].to_numpy(dtype=float) == pytest.approx(0.5)
    assert (table.table["k"].astype(float) > 0.0).all()
    assert table.table["nu0"].to_numpy(dtype=float) == pytest.approx(
        float(table.table.iloc[0]["nu0"])
    )
    assert table.table["k"].to_numpy(dtype=float) == pytest.approx(
        float(table.table.iloc[0]["k"])
    )
    identities = [int(v) for v in table.table["idx_ref"]]
    assert [int(v) for v in table.table["idx_backward"]] == identities[::-1]
