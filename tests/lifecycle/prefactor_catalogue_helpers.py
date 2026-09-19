"""Analytic full-source fixtures for prefactor catalogue lifecycle tests."""

import hashlib
import importlib.util
import inspect
import json
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pykmc
import pykmc.event_table as tables
import pytest
from pykmc.algorithms import rejection_free
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.htst import compute_event_prefactors, fd_hessian_fn
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.refinement import Refinement
from pykmc.result import EventRefinementOutput

MASS_A = 28.0855

MASS_B = 40.0

CELL = np.eye(3) * 20.0

FIRST = np.array([[-1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

SADDLE = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

FINAL = np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

GEOMETRIES = ("initial_positions", "saddle_positions", "final_positions")

REGISTRY_KEYS = (
    "descriptors",
    "calculations",
    "estimate_references",
    "estimate_history",
    "legacy_metadata",
)


def require_schema2():
    assert getattr(tables, "TABLE_SCHEMA_VERSION", None) == 2, (
        "Schema2 requires per-calculation producing provenance"
    )
    assert importlib.util.find_spec("pykmc.htst.catalogue") is not None
    assert "calculation" in inspect.signature(ReferenceEventTable._patch_row).parameters
    assert "method" in inspect.signature(PrefactorService).parameters
    assert "compute_energies" in inspect.signature(PrefactorService.compute).parameters
    assert callable(getattr(PrefactorService, "request_from_snapshot", None))


def write_potential(path, *, a=0.0, scale=1.0, bias=0.0):
    payload = {"a": a, "scale": scale, "bias": bias}
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(potential, saved=None, *, opaque=False):
    root = Path(pykmc.__file__).resolve().parent.parent
    original = Config.from_ini_file(str(root / "tests/data/input.in"))
    return original.model_copy(
        update={
            "frozen_atoms": None,
            "rateconstant": RateConstantConfig(
                style="htst",
                k0=1.0,
                T=300.0,
                free_radius=1.1,
                fd_step=0.01,
                premin=False,
                nu0_min_THz=1e-4,
                nu0_max_THz=1e8,
            ),
            "control": original.control.model_copy(
                update={
                    "reference_table": None if saved is None else str(saved),
                    "active_volume": False,
                }
            ),
            "lammps": original.lammps.model_copy(
                update={
                    "pair_style": "polynomial/opaque" if opaque else "eam/alloy",
                    "pair_coeff": f"* * {potential} Si",
                }
            ),
        }
    )


def energy(positions, parameters):
    """Full U: one quartic well and one stationary distant harmonic spectator."""
    a, scale, bias = (parameters[key] for key in ("a", "scale", "bias"))
    x, y, z = np.asarray(positions)[0]
    well = (x * x - 1) ** 2 * (1 + a * x * x)
    asymmetry = bias * (x**5 / 5 - x**3 / 3)
    spectator = np.asarray(positions)[1] - np.array([10.0, 0.0, 0.0])
    return float(
        scale * (well + asymmetry + (y * y + z * z) / 2) + spectator @ spectator / 2
    )


class PolynomialWorker:
    def __init__(self, potential, *, forbid=False):
        self.potential = Path(potential)
        self.forbid = forbid
        self.calls = []
        self.before_submit = None

    def submit(
        self, operation, *, request, compute_backward=True, compute_energies=False
    ):
        assert not self.forbid, (
            "crop-only unknown provenance must not submit a speculative calculation"
        )
        assert operation == "compute_event_prefactors"
        if self.before_submit is not None:
            self.before_submit(request)
        parameters = json.loads(self.potential.read_text())
        a, scale, bias = (parameters[key] for key in ("a", "scale", "bias"))
        # Two source rows are mandatory. Rebuilding from the local crop cannot
        # accidentally pass by evaluating a smaller but shape-valid potential.
        assert len(request.types) == 2, "recomputation lost full source spectator"
        assert request.types == ("Si", "Si")
        assert not request.settings.premin

        def forces(positions):
            positions = np.asarray(positions)
            force = -np.array(positions, copy=True)
            x = positions[0, 0]
            derivative = 6 * a * x**5 + 4 * (1 - 2 * a) * x**3 + 2 * (a - 2) * x
            derivative += bias * (x**4 - x**2)
            force[0] *= scale
            force[0, 0] = -scale * derivative
            force[1] = -(positions[1] - np.array([10.0, 0.0, 0.0]))
            return force

        # This independent analytic derivative verifies the exact stationary
        # triplet; no force projection or relaxation masks a bad input.
        for positions in (
            request.min1_positions,
            request.saddle_positions,
            request.min2_positions,
        ):
            np.testing.assert_allclose(forces(positions), 0.0, atol=1e-14, rtol=0)
        mass_map = dict(zip(request.species, request.masses, strict=True))
        masses = np.array([mass_map[symbol] for symbol in request.types])
        result = compute_event_prefactors(
            request,
            fd_hessian_fn(forces, masses, request.settings.fd_step),
            method="fd",
            compute_backward=compute_backward,
        )
        if compute_energies:
            values = tuple(
                energy(positions, parameters)
                for positions in (
                    request.min1_positions,
                    request.saddle_positions,
                    request.min2_positions,
                )
            )
            result = replace(
                result, provenance=replace(result.provenance, energies=values)
            )
        self.calls.append((request, compute_backward, compute_energies, result))
        future = Future()
        future.set_result(result)
        return future


def service(potential, *, saved=None, mass=MASS_A, opaque=False, forbid=False):
    cfg = config(potential, saved, opaque=opaque)
    worker = PolynomialWorker(potential, forbid=forbid)
    svc = PrefactorService(
        cfg,
        worker,
        create_rate_constant(cfg.rateconstant),
        species_masses=(("Si",), (mass,)),
        method="fd",
    )
    return svc, worker


def calculate(svc, key):
    constraints = ResolvedConstraints.resolve(
        FIRST, ("Si", "Si"), cell=CELL, pbc=(False, False, False)
    )
    request = svc.build_request(
        event_key=(key,),
        min1_positions=FIRST,
        saddle_positions=SADDLE,
        min2_positions=FINAL,
        types=("Si", "Si"),
        cell=CELL,
        pbc=(False, False, False),
        center_index=0,
        constraints=constraints,
    )
    result = svc.compute([request], compute_energies=True)[request.event_key]
    assert result.forward.ok and result.backward.ok
    assert result.provenance.source.is_complete
    assert result.provenance.free_indices == (0,)
    assert result.provenance.zone_indices == (0, 1)
    return result


def crop_row(idx, result, direction, *, linked=None):
    record = result.calculation(direction)
    source = record.provenance.source
    energies = record.provenance.energies
    reverse = direction == "backward"
    first = source.min2_positions if reverse else source.min1_positions
    final = source.min1_positions if reverse else source.min2_positions
    return {
        "idx_ref": idx,
        "idx_backward": idx if linked is None else linked,
        "event_id": f"event-{idx}",
        "id_final": f"final-{idx}",
        "id_saddle": f"saddle-{idx}",
        "initial_positions": np.array(first[:1]),
        "saddle_positions": np.array(source.saddle_positions[:1]),
        "final_positions": np.array(final[:1]),
        "types": ["Si"],
        "energy_barrier": energies[1] - energies[2 if reverse else 0],
        "k": 0.0,
        "move_atom_idx": 0,
        "sym_matrix": [np.eye(3)],
        "sym_perm": [np.array([0])],
        "dra": 2.0,
        "k_prefactor": 1.0,
        "nu0": np.nan,
        "nu0_status": "pending",
        "nu0_reason": "",
    }


def table_for(svc, records, unknown=()):
    """Explicit association only, using actual result records from the worker."""
    table = ReferenceEventTable(svc.config, prefactor_service=svc)
    table.table = pd.DataFrame(
        [
            crop_row(idx, result, direction, linked=linked)
            for idx, result, direction, linked in records
        ]
    )
    for idx, result, direction, _ in records:
        estimate = getattr(result, direction)
        if idx in unknown:
            table._patch_row(idx, estimate)
        else:
            table._patch_row(
                idx, estimate, calculation=result.calculation(direction), fresh=True
            )
    return table


def row_at(table, idx):
    selected = table.table.loc[table.table.idx_ref == idx]
    assert len(selected) == 1
    return selected.iloc[0]


def geometric_rows(frame):
    return {
        int(row.idx_ref): tuple(np.array(row[key], copy=True) for key in GEOMETRIES)
        for _, row in frame.iterrows()
    }


def assert_geometry_unchanged(frame, original):
    assert set(int(i) for i in frame.idx_ref) == set(original)
    for _, row in frame.iterrows():
        for key, expected in zip(GEOMETRIES, original[int(row.idx_ref)], strict=True):
            np.testing.assert_array_equal(row[key], expected)


def active_and_draw(table, idx, monkeypatch, *, reference_row=None):
    if reference_row is None:
        estimate = table.reference_estimate(idx)
        reference = row_at(table, idx)
    else:
        reference = reference_row
        refinement = Refinement.__new__(Refinement)
        refinement._carry_prefactors = True
        estimate = refinement._inherited_estimate(reference)
    active = ActiveEventTable(table.config, prefactor_service=table.prefactor_service)
    active.add_events(
        EventRefinementOutput(
            central_atom_index=0,
            saddle_positions=reference.saddle_positions,
            E_saddle=float(reference.energy_barrier),
            dE_forward=float(reference.energy_barrier),
            min2_positions=reference.final_positions,
            num_reference_event=idx,
            refined="F",
            **estimate,
        )
    )
    draws = iter((0.25, 0.5))
    monkeypatch.setattr("pykmc.algorithms.random.random", lambda: next(draws))
    selected, elapsed_ps, total = rejection_free(active.table.k.to_numpy(dtype=float))
    assert selected == 0
    assert elapsed_ps * total == pytest.approx(np.log(2), rel=1e-14)
    row = active.table.iloc[0]
    # Fixed existing pyKMC physical constant; this checks the active arithmetic
    # independently of the production rate facade used by both table classes.
    expected_rate = float(row.k_prefactor) * np.exp(
        -float(reference.energy_barrier) / (8.6173303e-5 * 300.0)
    )
    assert row.k == pytest.approx(expected_rate, rel=1e-13)
    return row


def assert_fallback(table, idx, monkeypatch):
    row = active_and_draw(table, idx, monkeypatch)
    assert row.nu0_status != "ok" and row.nu0_source == "k0"
    assert row.nu0_reason and np.isnan(row.nu0)
    assert row.k_prefactor == pytest.approx(table.config.rateconstant.k0)


def linked_calculation(table, idx):
    link = table.prefactor_archive.references[idx]
    assert link is not None
    return table.prefactor_archive.calculations[link.calculation_id]


def longitudinal_curvature(a, bias, sign, h):
    # Central-difference derivative of dU/dx at x=+/-1, before scale/mass.
    return 8 + 8 * a + (4 + 52 * a) * h**2 + 6 * a * h**4 + sign * bias * (2 + 4 * h**2)
