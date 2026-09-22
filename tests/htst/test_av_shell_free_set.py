"""The active-volume shell is fixed for the Hessian, never a free-set member.

Contracts 7f policy 5 as amended by R14/N09: an HTST/RPA request carries the
user+AV union as its free-set constraint. Shell atoms (beyond
``activevolume.rmov`` from the search centre) that fall inside ``free_radius``
are excluded from the Vineyard free set like user-frozen atoms, because the
search relaxed the core only while ``fix setforce`` held the shell: a shell
atom carries a residual force and is not at a stationary point. The user view
stays the coordinate contract. The adapter logs the free-set size and the
number of shell atoms it excluded.

Unit-level copy of the R04 LJ6 callers' site channel: the buffer atom inside
``free_radius`` carries 0.009 eV/Å (> ``force_tol`` 0.005). With the union the
free set is the mover alone (n_free 1) and the estimate is accepted; with the
user projection (the pre-R14 request) the buffer atom is free, n_free 2, and
the stationarity guard rejects every geometry.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import RateConstantConfig
from pykmc.engine.htst_lammps import LammpsHTSTExtension
from pykmc.htst.free_region import common_free_indices, select_free_indices
from pykmc.physics import EnginePhysics, resolve_event_constraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService

SPECIES = ("Ni", "Cu", "Fe")
MASSES = (58.6934, 63.546, 55.845)
TYPES = ("Ni", "Fe", "Cu")
IDS = (42, 8, 17)
MOVER, BUFFER, FAR = 0, 1, 2
CELL = np.eye(3) * 20.0
PBC = (False, False, False)
FREE_RADIUS = 1.1
RMOV = 1.2
BUFFER_FORCE = (0.0, 0.0, 0.00908203125)  # the R04 oracle's BUFFER_FORCE_Z
FAR_FORCE = (100.0, -200.0, 300.0)


def geometries():
    """Mover on x; the buffer is 1.70 Å from the AV centre, 0.94 Å from the saddle."""
    source = np.array([[2.0, 0.0, 0.0], [3.5, -0.8, 0.0], [8.0, 0.0, 0.0]])
    saddle, final = source.copy(), source.copy()
    saddle[MOVER, 0], final[MOVER, 0] = 3.0, 4.0
    return source, saddle, final


def label_of(positions):
    x = float(positions[MOVER, 0])
    assert x in (2.0, 3.0, 4.0)
    return {2.0: "min1", 3.0: "saddle", 4.0: "min2"}[x]


class FakeScratch:
    """Protocol endpoint only: forces per row, no native construction."""

    def __init__(self, native, calls):
        self.config, self.calls = native, calls
        self.types = ()
        self.positions = None
        self.closed = False
        self.observed = []
        self.lmp = SimpleNamespace(command=self.command)

    def start(self):
        self.calls.append(("start",))

    def command(self, text):
        assert text == "clear"

    def initialize_parameters(self):
        pass

    def initialize_system(self, *, types, positions, cell, pbc, species, masses):
        self.types = tuple(types)
        self.positions = np.array(positions, copy=True)
        self.full_system = SimpleNamespace(species=species, masses=masses, physics=None)

    def initialize_potential(self):
        self.full_system.physics = EnginePhysics.capture(
            self.config, self.full_system.species, self.full_system.masses
        )

    def set_positions(self, positions):
        self.positions = np.array(positions, copy=True)

    def get_positions(self):
        return self.positions.copy()

    def get_forces(self, *, positions):
        geometry = np.array(positions, copy=True)
        forces = np.zeros_like(geometry)
        forces[BUFFER] = BUFFER_FORCE
        forces[FAR] = FAR_FORCE
        self.observed.append((label_of(geometry), forces.copy()))
        self.calls.append(("force", label_of(geometry)))
        return forces

    def close(self):
        self.closed = True
        self.calls.append(("close",))


def av_config():
    native = SimpleNamespace(
        pair_style="zero 10.0",
        pair_coeff="* *",
        min_style="cg",
        minimize="0.0 1e-12 100 1000",
        frz_min="0.0 1e-12 100 1000",
        verbosity=0,
    )
    return SimpleNamespace(
        lammps=native,
        frozen_atoms=None,
        control=SimpleNamespace(active_volume=True),
        activevolume=SimpleNamespace(rmov=RMOV, ract=5.0, AV_debug=False),
        rateconstant=RateConstantConfig(
            style="htst",
            k0=1.0,
            free_radius=FREE_RADIUS,
            nu0_min_THz=1e-4,
            nu0_max_THz=1e8,
        ),
    )


def union_for(config, source):
    union = resolve_event_constraints(
        config, source, TYPES, CELL, PBC, MOVER, IDS, user_constraints=None
    )
    assert union.center_id == IDS[MOVER] and union.rmov == RMOV
    assert set(union.fixed_ids) == {IDS[BUFFER], IDS[FAR]}, "both beyond rmov"
    assert union.user_fixed_ids == ()
    return union


def build(config, constraints, calls):
    service = PrefactorService(
        config,
        object(),
        create_rate_constant(config.rateconstant),
        engine_physics=EnginePhysics.capture(config.lammps, SPECIES, MASSES),
        global_constraints=None,
    )
    source, saddle, final = geometries()
    request = service.build_request(
        event_key=("site", 1),
        min1_positions=source,
        saddle_positions=saddle,
        min2_positions=final,
        types=TYPES,
        cell=CELL.copy(),
        pbc=PBC,
        center_index=MOVER,
        constraints=constraints,
    )
    sphere = select_free_indices(saddle, MOVER, FREE_RADIUS, CELL, PBC)
    assert sphere.tolist() == [MOVER, BUFFER], "the buffer sits inside free_radius"
    scratch = FakeScratch(config.lammps, calls)
    extension = object.__new__(LammpsHTSTExtension)
    extension.engine = SimpleNamespace(config=config.lammps, comm=None, engine_id=0)

    def matrix(resource, positions, free, step):
        assert resource is scratch
        assert tuple(free) == (MOVER,), "only the mover may reach the Hessian"
        calls.append(("hessian", label_of(positions)))
        # Mass-weighted; one unstable saddle mode, three stable minimum modes.
        return np.diag(
            [-1.0, 2.0, 3.0] if label_of(positions) == "saddle" else [4.0, 5.0, 6.0]
        )

    extension._new_scratch = lambda: scratch
    extension._eskm_hessian = matrix
    return request, extension, scratch


def test_union_request_excludes_the_shell_from_the_free_set_and_is_accepted(
    caplog,
) -> None:
    config = av_config()
    source = geometries()[0]
    calls: list = []
    request, extension, scratch = build(config, union_for(config, source), calls)
    assert request.constraints.local_fixed_indices == (BUFFER, FAR)
    assert common_free_indices(request).tolist() == [MOVER]
    with caplog.at_level(logging.DEBUG, logger="log"):
        result = extension.compute_event_prefactors(request, compute_backward=True)
    assert scratch.closed
    assert result.n_free == 1
    assert result.forward.ok and result.backward.ok
    assert result.provenance.free_indices == (MOVER,)
    assert [c[1] for c in calls if c[0] == "hessian"] == ["saddle", "min1", "min2"]
    # The buffer's residual force was present at every geometry and ignored.
    assert all(
        np.linalg.norm(forces[BUFFER]) == pytest.approx(BUFFER_FORCE[2])
        for _, forces in scratch.observed
    )
    assert len(scratch.observed) == 3
    lines = [r.getMessage() for r in caplog.records if "[htst]" in r.getMessage()]
    assert any(
        "free set 1 of 2" in line and "active-volume shell excluded 1" in line
        for line in lines
    ), lines


def test_user_projection_request_puts_the_buffer_in_the_free_set_and_is_rejected(
    caplog,
) -> None:
    """The pre-R14 request (policy 5 as first implemented) rejects on the buffer."""
    config = av_config()
    source = geometries()[0]
    calls: list = []
    user_view = union_for(config, source).user_view()
    assert user_view.fixed_ids == () and user_view.rmov is None
    request, extension, _ = build(config, user_view, calls)
    assert common_free_indices(request).tolist() == [MOVER, BUFFER]
    with caplog.at_level(logging.DEBUG, logger="log"):
        result = extension.compute_event_prefactors(request, compute_backward=True)
    assert result.n_free == 2
    for direction in (result.forward, result.backward):
        assert direction.status == "rejected"
        assert direction.reason_code.value == "nonstationary_geometry"
        assert "0.00908" in direction.reason
    assert [c for c in calls if c[0] == "hessian"] == []
    lines = [r.getMessage() for r in caplog.records if "[htst]" in r.getMessage()]
    assert any(
        "free set 2 of 2" in line and "active-volume shell excluded 0" in line
        for line in lines
    ), lines
