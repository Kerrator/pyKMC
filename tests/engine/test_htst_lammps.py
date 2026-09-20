"""``LammpsHTSTExtension``: eskm Hessians on a scratch engine, SW-Si oracle.

Serial tests (skip under ``mpirun``) cover the public surface, the
FD-vs-eskm Hessian oracle on a 64-atom diamond Si cell, the tracked SW-Si
vacancy-hop fixture end to end, the zone crop, premin, preflight and the
failure taxonomy with cleanup proven after injected failures. The MPI class
(``mpirun -n 4``) attaches the extension to a multi-rank search engine and
checks that only the local root computes and that no collective touches the
search engine's communicator.

Every number recorded here comes from ``examples/Si_vac/Si.sw`` and
``tests/data/htst_si_vacancy_hop.npz`` (see ``tests/data/README.md``).
"""

from __future__ import annotations

import glob
import inspect
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

pytest.importorskip("lammps")

from ase.build import bulk  # noqa: E402

from pykmc.engine.htst_lammps import (  # noqa: E402
    TMPDIR_PREFIX,
    LammpsHTSTExtension,
)
from pykmc.engine.lammps import LammpsEngine  # noqa: E402
from pykmc.htst import (  # noqa: E402
    EventPrefactors,
    HTSTEventRequest,
    HTSTRequestError,
    HTSTSettings,
    PrefactorRejection,
    compute_event_prefactors,
    fd_hessian_fn,
    select_free_indices,
)

_ROOT = Path(__file__).resolve().parents[2]
_SI_SW = _ROOT / "examples" / "Si_vac" / "Si.sw"
_HOP_FIXTURE = _ROOT / "tests" / "data" / "htst_si_vacancy_hop.npz"
_THZ = 1.0e12
_ALL_PERIODIC = (True, True, True)


@dataclass
class _SWConfig:
    """Stillinger-Weber Si; ``sw`` sets no masses, so pyKMC's map is what LAMMPS has."""

    pair_style: str = "sw"
    pair_coeff: str = f"* * {_SI_SW} Si"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


@dataclass
class _LJConfig:
    """Lennard-Jones for any number of species (two-species crop test)."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


def _count_tmpdirs() -> int:
    """Count the ``TMPDIR_PREFIX`` directories in the temp root."""
    return len(glob.glob(os.path.join(tempfile.gettempdir(), TMPDIR_PREFIX + "*")))


def _initialize(
    engine: LammpsEngine,
    types: Any,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Any = _ALL_PERIODIC,
) -> None:
    """Replay parameters / system / potential on a started engine."""
    engine.initialize_parameters()
    engine.initialize_system(types=types, positions=positions, cell=cell, pbc=pbc)
    engine.initialize_potential()


def _engine_state(engine: LammpsEngine) -> tuple[np.ndarray, float, int]:
    """Positions, total energy and atom count of a serial engine."""
    return (
        np.array(engine.get_positions(), copy=True),
        float(engine.get_total_energy()),
        int(engine.lmp.get_natoms()),
    )


def _assert_state_unchanged(
    engine: LammpsEngine, before: tuple[np.ndarray, float, int]
) -> None:
    """Assert the search engine is bit-identical to ``before``."""
    positions, energy, natoms = _engine_state(engine)
    assert natoms == before[2]
    assert np.array_equal(positions, before[0])
    assert energy == before[1]


def _si64() -> tuple[list[str], np.ndarray, np.ndarray]:
    """2x2x2 conventional diamond Si (64 atoms, 10.862 Å box)."""
    atoms = bulk("Si", crystalstructure="diamond", a=5.431, cubic=True).repeat(2)
    return (
        atoms.get_chemical_symbols(),
        atoms.get_positions(),
        np.array(atoms.get_cell()),
    )


@pytest.fixture(scope="session")
def sw_config() -> _SWConfig:
    """SW-Si config, skipping when the tracked potential is absent."""
    if not _SI_SW.is_file():
        pytest.skip(f"{_SI_SW.name} not present")
    return _SWConfig()


@pytest.fixture(scope="session")
def si_hop() -> dict[str, Any]:
    """Load the tracked SW-Si vacancy-hop fixture as plain arrays."""
    if not _HOP_FIXTURE.is_file():
        pytest.skip(f"{_HOP_FIXTURE.name} not present")
    with np.load(_HOP_FIXTURE, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


@pytest.fixture
def search_engine(sw_config: _SWConfig, si_hop: dict[str, Any]) -> LammpsEngine:
    """Build a serial search engine holding the fixture's ``min1`` geometry."""
    engine = LammpsEngine(config=sw_config, comm=None, engine_id=0)
    engine.start()
    _initialize(
        engine,
        [str(t) for t in si_hop["types"]],
        si_hop["min1_positions"],
        si_hop["cell"],
    )
    yield engine
    engine.close()


@pytest.fixture
def hop_request(
    si_hop: dict[str, Any], search_engine: LammpsEngine
) -> Callable[..., HTSTEventRequest]:
    """Build fixture requests carrying the engine's authoritative masses."""
    full_system = search_engine.full_system

    def make(**settings: Any) -> HTSTEventRequest:
        return HTSTEventRequest(
            event_key=("ref", int(si_hop["idx_ref_forward"])),
            min1_positions=si_hop["min1_positions"].copy(),
            saddle_positions=si_hop["saddle_positions"].copy(),
            min2_positions=si_hop["min2_positions"].copy(),
            types=tuple(str(t) for t in si_hop["types"]),
            species=full_system.species,
            masses=full_system.masses,
            cell=si_hop["cell"].copy(),
            pbc=_ALL_PERIODIC,
            center_index=int(si_hop["central_atom_idx"]),
            settings=HTSTSettings(**settings),
        )

    return make


@pytest.fixture
def scratch_log(monkeypatch: pytest.MonkeyPatch) -> list[LammpsEngine]:
    """Record every scratch engine the extension creates (to prove they close)."""
    created: list[LammpsEngine] = []
    original = LammpsHTSTExtension._new_scratch

    def spy(self: LammpsHTSTExtension) -> LammpsEngine:
        engine = original(self)
        created.append(engine)
        return engine

    monkeypatch.setattr(LammpsHTSTExtension, "_new_scratch", spy)
    return created


@pytest.fixture
def hessian_geometries(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Capture the geometry handed to every eskm Hessian call."""
    captured: list[np.ndarray] = []
    original = LammpsHTSTExtension._eskm_hessian

    def spy(
        self: LammpsHTSTExtension,
        scratch: LammpsEngine,
        positions: np.ndarray,
        free_indices: np.ndarray,
        fd_step: float,
    ) -> np.ndarray:
        captured.append(np.array(positions, copy=True))
        return original(self, scratch, positions, free_indices, fd_step)

    monkeypatch.setattr(LammpsHTSTExtension, "_eskm_hessian", spy)
    return captured


@pytest.fixture
def raw_force_evaluations(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record actual undisplaced native force checks; forward every real call.

    This observes stationarity before the eskm boundary without faking forces,
    changing geometry, changing minimization or increasing force_tol.
    """
    captured: list[dict[str, Any]] = []
    original = LammpsEngine.get_forces

    def spy(self, positions=None, recompute=True):
        result = original(self, positions=positions, recompute=recompute)
        if result is not None:
            geometry = self.get_positions() if positions is None else positions
            captured.append(
                {
                    "positions": np.array(geometry, dtype=float, copy=True),
                    "forces": np.array(result, dtype=float, copy=True),
                }
            )
        return result

    monkeypatch.setattr(LammpsEngine, "get_forces", spy)
    return captured


def _assert_all_closed(created: list[LammpsEngine], expected: int) -> None:
    """Exactly ``expected`` scratch engines were created and every one is closed."""
    assert len(created) == expected
    assert all(engine.lmp is None for engine in created)


class TestLammpsHTSTSerial:
    """Serial engine, no manager."""

    @pytest.fixture(autouse=True)
    def require_serial(self) -> None:
        """Skip under ``mpirun``: the serial engine uses ``comm=None``."""
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() > 1:
            pytest.skip("serial tests must run without mpirun")

    # -- surface ---------------------------------------------------------

    def test_requires_a_lammps_engine(self) -> None:
        """Attaching to anything but a ``LammpsEngine`` is a ``TypeError``."""
        fake = SimpleNamespace(register=lambda ext: None, comm=None)
        with pytest.raises(TypeError, match="LammpsEngine"):
            LammpsHTSTExtension(fake)

    def test_scratch_engines_never_write_log_files(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        sw_config: _SWConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A verbose production config must not produce one log per Hessian call.

        The search engine has already been started, so re-pointing its config
        at a verbose copy only affects the scratch engines the extension builds.
        """
        verbose = _SWConfig(**{**sw_config.__dict__, "verbosity": 2})
        monkeypatch.setattr(search_engine, "config", verbose)
        monkeypatch.chdir(tmp_path)
        ext = LammpsHTSTExtension(search_engine)
        assert ext.htst_preflight() is not None
        result = ext.compute_event_prefactors(hop_request(free_radius=4.0))
        assert result.forward.status == "ok"
        assert sorted(tmp_path.iterdir()) == []

    def test_public_surface_is_exactly_two_operations(
        self, search_engine: LammpsEngine
    ) -> None:
        """Only ``compute_event_prefactors`` and ``htst_preflight`` are exposed."""
        public = {
            name
            for name, _ in inspect.getmembers(
                LammpsHTSTExtension, predicate=inspect.isfunction
            )
            if not name.startswith("_")
        }
        assert public == {"compute_event_prefactors", "htst_preflight"}
        LammpsHTSTExtension(search_engine)
        assert callable(search_engine.compute_event_prefactors)
        assert callable(search_engine.htst_preflight)
        assert "compute_event_prefactors" in dir(search_engine)

    def test_non_root_rank_returns_none_without_a_scratch(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
    ) -> None:
        """A non-root local rank returns ``None`` immediately from both operations."""
        ext = LammpsHTSTExtension(search_engine)
        search_engine.comm = SimpleNamespace(Get_rank=lambda: 1, Get_size=lambda: 2)
        try:
            assert ext.compute_event_prefactors(hop_request()) is None
            assert ext.htst_preflight() is None
        finally:
            search_engine.comm = None
        assert scratch_log == []

    def test_invalid_request_raises_before_any_scratch(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
    ) -> None:
        """Validation errors cost no engine work and propagate as exceptions."""
        ext = LammpsHTSTExtension(search_engine)
        before = _engine_state(search_engine)
        good = hop_request()
        bad = HTSTEventRequest(
            event_key=good.event_key,
            min1_positions=good.min1_positions,
            saddle_positions=good.saddle_positions,
            min2_positions=good.min2_positions,
            types=good.types,
            species=good.species,
            masses=good.masses,
            cell=good.cell,
            pbc=good.pbc,
            center_index=len(good.types),  # out of range
            settings=good.settings,
        )
        with pytest.raises(HTSTRequestError, match="center_index"):
            ext.compute_event_prefactors(bad)
        with pytest.raises(HTSTRequestError, match="HTSTEventRequest"):
            ext.compute_event_prefactors({"not": "a request"})
        with pytest.raises(HTSTRequestError, match="zone_radius"):
            ext.compute_event_prefactors(hop_request(free_radius=6.0, zone_radius=6.0))
        with pytest.raises(HTSTRequestError, match="zone_radius"):
            ext.compute_event_prefactors(hop_request(free_radius=6.0, zone_radius=5.0))
        assert scratch_log == []
        _assert_state_unchanged(search_engine, before)

    # -- Hessian oracle ---------------------------------------------------

    def test_eskm_matches_fd_hessian_on_small_system(
        self, sw_config: _SWConfig
    ) -> None:
        """Compare eskm with the FD kernel element-wise on a perturbed 64-atom Si cell.

        Same scratch engine, geometry, masses and ``fd_step``; a non-contiguous
        free set, given sorted and unsorted (ordering check). Measured max
        relative deviation is below 1e-9 (both are central differences with the
        same step; the file carries eight decimals in eskm units).
        """
        from mpi4py import MPI

        types, positions, cell = _si64()
        rng = np.random.default_rng(0)
        geometry = positions + rng.normal(0.0, 0.05, positions.shape)
        parent = LammpsEngine(config=sw_config, comm=None, engine_id=0)
        parent.start()
        _initialize(parent, types, geometry, cell)
        ext = LammpsHTSTExtension(parent)
        scratch = LammpsEngine(config=sw_config, comm=MPI.COMM_SELF, engine_id=9)
        scratch.start()
        try:
            _initialize(scratch, types, geometry, cell)
            masses = np.repeat(scratch.full_system.masses, len(types))
            fd_step = 0.01
            fd = fd_hessian_fn(
                lambda p: scratch.get_forces(positions=p), masses, fd_step
            )
            free = np.array([0, 5, 17, 40])
            h_eskm = ext._eskm_hessian(scratch, geometry, free, fd_step)
            h_fd = fd(geometry, free)
            assert h_eskm.shape == h_fd.shape == (12, 12)
            assert np.allclose(h_eskm, h_eskm.T, atol=1e-8, rtol=0.0)
            assert np.allclose(h_fd, h_fd.T, atol=1e-8, rtol=0.0)
            scale = float(np.abs(h_fd).max())
            assert scale > 0.1  # a real Hessian, not zeros
            deviation = float(np.abs(h_eskm - h_fd).max()) / scale
            assert deviation < 1.0e-8, deviation
            # Ordering: an unsorted free set is permuted back into request order.
            perm = np.array([2, 0, 3, 1])
            h_perm = ext._eskm_hessian(scratch, geometry, free[perm], fd_step)
            rows = (3 * perm[:, None] + np.arange(3)[None, :]).reshape(-1)
            assert np.array_equal(h_perm, h_eskm[np.ix_(rows, rows)])
            fd_perm = fd(geometry, free[perm])
            assert float(np.abs(h_perm - fd_perm).max()) / scale < 1.0e-8
        finally:
            scratch.close()
            parent.close()
        assert _count_tmpdirs() == 0

    # -- SW-Si fixture -----------------------------------------------------

    @pytest.mark.parametrize(
        ("center", "n_free", "forward_thz", "backward_thz"),
        [
            ("saddle", 37, 21.1588, 21.1844),
            ("min1", 43, 23.6139, 19.6167),
        ],
    )
    def test_sw_si_fixture_end_to_end(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
        center: str,
        n_free: int,
        forward_thz: float,
        backward_thz: float,
    ) -> None:
        """Both directions accepted on the tracked vacancy hop; nothing leaks.

        Measured (free_radius 6, dx 0.01, premin False, full system) with the
        S7 measurement script and reproduced here without tuning: the
        saddle-centred free region (default) holds 37 atoms and gives forward
        21.1588 / backward 21.1844 THz (symmetric to 0.12 %, as the hop is);
        the min1-centred region of the original model holds 43 atoms and gives
        23.6139 / 19.6167 THz, a 20 % asymmetry that is entirely the
        frozen-boundary choice (the donor's 23.6 THz / 43 atoms).
        """
        ext = LammpsHTSTExtension(search_engine)
        before = _engine_state(search_engine)
        tmp_before = _count_tmpdirs()
        settings = {"free_radius": 6.0, "free_region_center": center}
        result = ext.compute_event_prefactors(hop_request(**settings))
        assert isinstance(result, EventPrefactors)
        assert result.method == "lammps_eskm"
        assert result.event_key == ("ref", 4)
        assert result.settings.free_region_center == center
        assert result.n_free == n_free
        for direction in (result.forward, result.backward):
            assert direction.status == "ok", direction.reason
            assert direction.n_free == n_free
            assert direction.n_negative_saddle == 1
            assert direction.n_positive_min == 3 * n_free
            assert 1.0 <= direction.nu0_hz / _THZ <= 100.0
        assert result.forward.nu0_hz / _THZ == pytest.approx(forward_thz, rel=1e-3)
        assert result.backward.nu0_hz / _THZ == pytest.approx(backward_thz, rel=1e-3)
        _assert_state_unchanged(search_engine, before)
        assert _count_tmpdirs() == tmp_before
        _assert_all_closed(scratch_log, 1)

    def test_default_centring_is_the_saddle(
        self, hop_request: Callable[..., HTSTEventRequest]
    ) -> None:
        """A request built without the setting is saddle-centred (the decided default)."""
        assert hop_request().settings.free_region_center == "saddle"

    def test_forward_only_request_skips_the_min2_hessian(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        hessian_geometries: list[np.ndarray],
        scratch_log: list[LammpsEngine],
    ) -> None:
        """``compute_backward=False``: two eskm calls, backward 'skipped', same forward value."""
        ext = LammpsHTSTExtension(search_engine)
        request = hop_request(free_radius=6.0)
        both = ext.compute_event_prefactors(request)
        assert len(hessian_geometries) == 3
        del hessian_geometries[:]
        forward_only = ext.compute_event_prefactors(request, compute_backward=False)
        assert len(hessian_geometries) == 2  # saddle, then min1
        assert np.array_equal(hessian_geometries[0], request.saddle_positions)
        assert np.array_equal(hessian_geometries[1], request.min1_positions)
        assert forward_only.forward == both.forward
        assert forward_only.backward.status == "skipped"
        assert forward_only.backward.reason == "not requested"
        assert forward_only.backward.nu0_hz is None
        assert forward_only.backward.n_free == 37
        assert forward_only.backward.n_negative_saddle == 1
        _assert_all_closed(scratch_log, 2)

    def test_site_geometry_from_the_full_saddle_matches_the_reference(
        self,
        search_engine: LammpsEngine,
        si_hop: dict[str, Any],
        hop_request: Callable[..., HTSTEventRequest],
        hessian_geometries: list[np.ndarray],
        raw_force_evaluations: list[dict[str, Any]],
        scratch_log: list[LammpsEngine],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A site request from the full refined saddle equals the reference; a crop does not.

        The S7 '(iv)' bias: the request ``ActiveEventTable.request_site_prefactors``
        builds (current minimum as min1, full saddle, min2 unused, forward only)
        gives the reference forward nu0 to 1e-9, whereas the previous
        construction (the ``rcut`` crop of the saddle pasted into the minimum,
        every other atom at its minimum position) does not.

        Measured with the Si_vac example's ``rcut`` 6.3 A under saddle
        centring: -14.8 % (the free atoms between 6.3 A and free_radius plus
        the SW cutoff see an unrelaxed boundary); the bias vanishes only from
        ``rcut`` 10 A on.
        """
        before = _engine_state(search_engine)
        ext = LammpsHTSTExtension(search_engine)
        full_system = search_engine.full_system
        reference = ext.compute_event_prefactors(hop_request(free_radius=6.0))
        min1 = si_hop["min1_positions"]
        saddle = si_hop["saddle_positions"]
        center = int(si_hop["central_atom_idx"])

        def site_request(saddle_geometry: np.ndarray, key: tuple) -> HTSTEventRequest:
            return HTSTEventRequest(
                event_key=key,
                min1_positions=min1.copy(),
                saddle_positions=saddle_geometry.copy(),
                min2_positions=min1.copy(),  # unused: forward only
                types=tuple(str(t) for t in si_hop["types"]),
                species=full_system.species,
                masses=full_system.masses,
                cell=si_hop["cell"].copy(),
                pbc=_ALL_PERIODIC,
                center_index=center,
                settings=HTSTSettings(free_radius=6.0),
            )

        new = ext.compute_event_prefactors(
            site_request(saddle, ("site", "full")), compute_backward=False
        )
        assert new.forward.ok and new.backward.skipped
        assert new.n_free == reference.n_free == 37
        assert new.forward.nu0_hz == pytest.approx(reference.forward.nu0_hz, rel=1e-9)

        rcut = 6.3  # examples/Si_vac atomicenvironment.rcut
        crop = select_free_indices(min1, center, rcut, si_hop["cell"], _ALL_PERIODIC)
        pasted = min1.copy()
        pasted[crop] = saddle[crop]
        pasted_request = site_request(pasted, ("site", "crop"))
        matrices_before = len(hessian_geometries)
        with caplog.at_level(logging.WARNING, logger="log"):
            rejected = ext.compute_event_prefactors(
                pasted_request, compute_backward=False
            )
        assert rejected.forward.reason_code is PrefactorRejection.NONSTATIONARY_GEOMETRY
        # The rejection is visible at run time: one WARNING per rejected event
        # naming the event, the offending force norm and the tolerance.
        stationarity_warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and "force_tol" in r.getMessage()
        ]
        assert len(stationarity_warnings) == 1, caplog.text
        assert repr(pasted_request.event_key) in stationarity_warnings[0]
        assert f"{pasted_request.settings.force_tol}" in stationarity_warnings[0]
        assert "eV/" in stationarity_warnings[0]
        assert rejected.forward.nu0_hz is None
        assert rejected.backward.skipped
        assert len(hessian_geometries) == matrices_before
        observed = raw_force_evaluations[-1]
        np.testing.assert_array_equal(observed["positions"], pasted)
        free = select_free_indices(
            pasted,
            center,
            pasted_request.settings.free_radius,
            pasted_request.cell,
            pasted_request.pbc,
        )
        norms = np.linalg.norm(observed["forces"][free], axis=1)
        assert np.isfinite(norms).all()
        assert float(norms.max()) > pasted_request.settings.force_tol

        # Historical curvature-only diagnostic of a NONSTATIONARY geometry.
        # The actual native operation above rejected it before any matrix.
        # Preserve the old measured frequency bias using the real low-level
        # eskm matrix and generic Hessian-only kernel; never relax the threshold.
        diagnostic = ext._new_scratch()
        try:
            diagnostic.start()
            ext._build_scratch(
                diagnostic,
                types=pasted_request.types,
                positions=pasted_request.min1_positions,
                cell=pasted_request.cell,
                pbc=pasted_request.pbc,
                species=pasted_request.species,
                masses=pasted_request.masses,
            )
            old = compute_event_prefactors(
                pasted_request,
                ext._hessian_fn(diagnostic, pasted_request.settings.fd_step),
                method="nonstationary_diagnostic_lammps_eskm",
                free_indices=free,
                compute_backward=False,
            )
        finally:
            diagnostic.close()
        assert old.method == "nonstationary_diagnostic_lammps_eskm"
        assert old.forward.ok
        bias = old.forward.nu0_hz / reference.forward.nu0_hz - 1.0
        assert abs(bias) > 0.01, (
            bias
        )  # the old construction is biased (measured -14.8 %)
        assert bias == pytest.approx(-0.148, abs=0.01)

        print(
            {
                "diagnostic": "nonstationary_pasted_saddle_curvature_only",
                "maximum_free_force": float(norms.max()),
                "force_tol": pasted_request.settings.force_tol,
                "native_status": rejected.forward.reason_code.value,
                "reference_nu0_hz": reference.forward.nu0_hz,
                "diagnostic_nu0_hz": old.forward.nu0_hz,
                "diagnostic_relative_bias": bias,
            }
        )
        _assert_state_unchanged(search_engine, before)
        _assert_all_closed(scratch_log, 4)

    def test_fd_and_eskm_prefactors_agree_on_fixture(
        self,
        sw_config: _SWConfig,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
    ) -> None:
        """The FD oracle and eskm give the same nu0 at ``free_radius=4``.

        Measured (saddle-centred): 19 free atoms, forward 19.809 THz, backward
        19.805 THz for both; the min1-centred selection of the original model
        gives 13 free atoms, 17.350 / 12.593 THz. The FD oracle follows the
        same centring setting as the extension, so the free sets are equal.
        """
        from mpi4py import MPI

        request = hop_request(free_radius=4.0)
        ext = LammpsHTSTExtension(search_engine)
        eskm = ext.compute_event_prefactors(request)
        scratch = LammpsEngine(config=sw_config, comm=MPI.COMM_SELF, engine_id=9)
        scratch.start()
        try:
            _initialize(scratch, request.types, request.min1_positions, request.cell)
            fd = compute_event_prefactors(
                request,
                fd_hessian_fn(
                    lambda p: scratch.get_forces(positions=p),
                    request.masses_per_atom(),
                    request.settings.fd_step,
                ),
            )
        finally:
            scratch.close()
        assert fd.method == "fd" and eskm.method == "lammps_eskm"
        assert fd.n_free == eskm.n_free == 19
        assert eskm.forward.nu0_hz / _THZ == pytest.approx(19.809, rel=1e-3)
        assert eskm.backward.nu0_hz / _THZ == pytest.approx(19.805, rel=1e-3)
        for oracle, native in (
            (fd.forward, eskm.forward),
            (fd.backward, eskm.backward),
        ):
            assert oracle.ok and native.ok
            assert native.nu0_hz / oracle.nu0_hz == pytest.approx(1.0, rel=1e-9)

    def test_zone_crop_matches_full_system(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
    ) -> None:
        """``zone_radius=10`` reproduces the full system within 2 percent.

        Measured under saddle centring (37 free atoms, 207-atom zone):
        identical to 1e-12 relative (the 4 Å shell exceeds the SW cutoff of
        3.77 Å, so every free-atom Hessian block is complete). The zone is
        selected on the same centring geometry as the free set, so the free
        set is a subset of the zone by construction.
        """
        ext = LammpsHTSTExtension(search_engine)
        full = ext.compute_event_prefactors(hop_request(free_radius=6.0))
        zone = ext.compute_event_prefactors(
            hop_request(free_radius=6.0, zone_radius=10.0)
        )
        assert zone.n_free == full.n_free == 37
        for cropped, reference in (
            (zone.forward, full.forward),
            (zone.backward, full.backward),
        ):
            assert cropped.ok and reference.ok
            assert cropped.nu0_hz == pytest.approx(reference.nu0_hz, rel=0.02)
            assert cropped.nu0_hz == pytest.approx(reference.nu0_hz, rel=1e-9)
        # The crop really held only the zone atoms (selected on the saddle
        # geometry, like the free set), with the full species map.
        request = hop_request(free_radius=6.0, zone_radius=10.0)
        zone_atoms = select_free_indices(
            request.saddle_positions,
            request.center_index,
            10.0,
            request.cell,
            request.pbc,
        )
        free_atoms = select_free_indices(
            request.saddle_positions,
            request.center_index,
            6.0,
            request.cell,
            request.pbc,
        )
        assert np.isin(free_atoms, zone_atoms).all()
        crop_engine = scratch_log[-1]
        assert 37 < crop_engine.full_system.natoms == zone_atoms.size < 1727
        assert crop_engine.full_system.species == ("Si",)
        _assert_all_closed(scratch_log, 2)

    def test_zone_crop_keeps_species_absent_from_the_crop(
        self,
        scratch_log: list[LammpsEngine],
        hessian_geometries: list[np.ndarray],
        raw_force_evaluations: list[dict[str, Any]],
    ) -> None:
        """A crop holding only Ni keeps the full ('Fe', 'Ni') map and masses."""
        atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=True).repeat(4)
        types = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        cell = np.array(atoms.get_cell())
        center = 0
        delta = positions - positions[center]
        delta -= np.diag(cell) * np.round(delta / np.diag(cell))  # minimum image
        far = int(np.argmax(np.linalg.norm(delta, axis=1)))
        types[far] = "Fe"
        engine = LammpsEngine(config=_LJConfig(), comm=None, engine_id=0)
        engine.start()
        try:
            _initialize(engine, types, positions, cell)
            full_system = engine.full_system
            assert full_system.species == ("Fe", "Ni")
            ext = LammpsHTSTExtension(engine)
            request = HTSTEventRequest(
                event_key=("lj", 0),
                min1_positions=positions.copy(),
                saddle_positions=positions.copy(),
                min2_positions=positions.copy(),
                types=tuple(types),
                species=full_system.species,
                masses=full_system.masses,
                cell=cell,
                pbc=_ALL_PERIODIC,
                center_index=center,
                settings=HTSTSettings(free_radius=2.6, zone_radius=6.0),
            )
            result = ext.compute_event_prefactors(request)
        finally:
            engine.close()
        # The unchanged truncated LJ fixture is not stationary: this earlier
        # physical rejection now precedes any saddle-spectrum classification.
        assert result.forward.reason_code is PrefactorRejection.NONSTATIONARY_GEOMETRY
        assert result.backward.reason_code is PrefactorRejection.NONSTATIONARY_GEOMETRY
        assert hessian_geometries == []
        assert len(raw_force_evaluations) == 1
        observed = raw_force_evaluations[0]
        zone = select_free_indices(positions, center, 6.0, cell, _ALL_PERIODIC)
        free_global = select_free_indices(positions, center, 2.6, cell, _ALL_PERIODIC)
        free_local = np.searchsorted(zone, free_global)
        np.testing.assert_array_equal(zone[free_local], free_global)
        np.testing.assert_array_equal(observed["positions"], positions[zone])
        norms = np.linalg.norm(observed["forces"][free_local], axis=1)
        assert np.isfinite(norms).all()
        assert float(norms.max()) > request.settings.force_tol
        print(
            {
                "fixture": "truncated_lj_species_map",
                "maximum_free_force": float(norms.max()),
                "force_tol": request.settings.force_tol,
                "native_status": result.forward.reason_code.value,
            }
        )
        crop = scratch_log[-1].full_system
        assert "Fe" not in crop.types  # the Fe atom is outside the zone
        assert crop.species == ("Fe", "Ni")
        assert crop.masses == full_system.masses
        _assert_all_closed(scratch_log, 1)

    # -- premin -------------------------------------------------------------

    def test_premin_false_leaves_every_geometry_untouched(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        hessian_geometries: list[np.ndarray],
    ) -> None:
        """Default ``premin=False``: the Hessians see the request arrays verbatim."""
        request = hop_request(free_radius=6.0)
        LammpsHTSTExtension(search_engine).compute_event_prefactors(request)
        assert len(hessian_geometries) == 3  # saddle, then min1, then min2
        assert np.array_equal(hessian_geometries[0], request.saddle_positions)
        assert np.array_equal(hessian_geometries[1], request.min1_positions)
        assert np.array_equal(hessian_geometries[2], request.min2_positions)

    def test_premin_freezes_the_core_and_relaxes_the_rest(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        hessian_geometries: list[np.ndarray],
    ) -> None:
        """``premin=True``: core rows are bit-identical, surroundings move slightly.

        Measured on the fixture (saddle-centred, 37 free atoms): forward
        21.177 THz vs 21.159 THz without premin (0.09 percent), backward
        21.190 vs 21.184 THz (0.03 percent). The core is the free set, selected
        on the saddle geometry.
        """
        ext = LammpsHTSTExtension(search_engine)
        reference = ext.compute_event_prefactors(hop_request(free_radius=6.0))
        assert len(hessian_geometries) == 3
        del hessian_geometries[:]
        request = hop_request(free_radius=6.0, premin=True)
        result = ext.compute_event_prefactors(request)
        assert len(hessian_geometries) == 3
        core = select_free_indices(
            request.saddle_positions,
            request.center_index,
            request.settings.free_radius,
            request.cell,
            request.pbc,
        )
        assert core.size == 37
        rest = np.setdiff1d(np.arange(len(request.types)), core)
        originals = (
            request.saddle_positions,
            request.min1_positions,
            request.min2_positions,
        )
        for relaxed, original in zip(hessian_geometries, originals, strict=True):
            assert np.array_equal(relaxed[core], original[core])
            moved = np.linalg.norm(relaxed[rest] - original[rest], axis=1)
            assert moved.max() > 0.0
            assert moved.max() < 0.1
        for with_premin, without in (
            (result.forward, reference.forward),
            (result.backward, reference.backward),
        ):
            assert with_premin.ok and without.ok
            assert with_premin.nu0_hz == pytest.approx(without.nu0_hz, rel=0.01)
            assert with_premin.nu0_hz != without.nu0_hz

    def test_premin_relaxes_a_displaced_surrounding_atom(
        self,
        sw_config: _SWConfig,
        hessian_geometries: list[np.ndarray],
        raw_force_evaluations: list[dict[str, Any]],
    ) -> None:
        """A surrounding atom displaced by 0.1 Å is pulled back; the core stays.

        Measured: 0.1 Å -> 0.023 Å under the test config's ``frz_min``.
        """
        types, positions, cell = _si64()
        center = 0
        core = select_free_indices(positions, center, 2.5, cell, _ALL_PERIODIC)
        outside = int(np.setdiff1d(np.arange(len(types)), core)[7])
        displaced = positions.copy()
        displaced[outside] += np.array([0.1, 0.0, 0.0])
        engine = LammpsEngine(config=sw_config, comm=None, engine_id=0)
        engine.start()
        try:
            _initialize(engine, types, positions, cell)
            ext = LammpsHTSTExtension(engine)
            request = HTSTEventRequest(
                event_key=("si64", 0),
                min1_positions=displaced.copy(),
                saddle_positions=displaced.copy(),
                min2_positions=displaced.copy(),
                types=tuple(types),
                species=engine.full_system.species,
                masses=engine.full_system.masses,
                cell=cell,
                pbc=_ALL_PERIODIC,
                center_index=center,
                settings=HTSTSettings(free_radius=2.5, premin=True),
            )
            result = ext.compute_event_prefactors(request)
        finally:
            engine.close()
        assert result.forward.reason_code is PrefactorRejection.NONSTATIONARY_GEOMETRY
        assert result.backward.reason_code is PrefactorRejection.NONSTATIONARY_GEOMETRY
        assert hessian_geometries == []
        assert len(raw_force_evaluations) == 1
        # Observe the actual post-premin geometry at the raw-force boundary.
        # The matrix is correctly never requested for this residual core force.
        observed = raw_force_evaluations[0]
        relaxed = observed["positions"]
        norms = np.linalg.norm(observed["forces"][core], axis=1)
        assert np.isfinite(norms).all()
        assert float(norms.max()) > request.settings.force_tol
        print(
            {
                "fixture": "displaced_surroundings_after_premin",
                "maximum_free_force": float(norms.max()),
                "force_tol": request.settings.force_tol,
                "native_status": result.forward.reason_code.value,
            }
        )
        assert np.array_equal(relaxed[core], displaced[core])
        # frz_min is loose ("1e-4 1e-6 100 1000"): most, not all, of the way back.
        assert np.linalg.norm(relaxed[outside] - positions[outside]) < 0.05
        assert np.linalg.norm(displaced[outside] - positions[outside]) == pytest.approx(
            0.1
        )

    # -- failure taxonomy and cleanup ------------------------------------------

    def test_bad_pair_coeff_raises_and_cleans_up(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A LAMMPS error while building the scratch propagates; nothing leaks."""
        ext = LammpsHTSTExtension(search_engine)
        before = _engine_state(search_engine)
        tmp_before = _count_tmpdirs()
        monkeypatch.setattr(search_engine.config, "pair_coeff", "* * no_such.sw Si")
        with pytest.raises(Exception, match="no_such.sw"):
            ext.compute_event_prefactors(hop_request(free_radius=4.0))
        _assert_all_closed(scratch_log, 1)
        assert _count_tmpdirs() == tmp_before
        _assert_state_unchanged(search_engine, before)
        with pytest.raises(RuntimeError, match="force-model contents changed"):
            ext.htst_preflight()
        _assert_all_closed(scratch_log, 1)

    def test_lammps_error_inside_dynamical_matrix_cleans_up(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A LAMMPS error after the group exists propagates; scratch and files go."""

        def broken(
            scratch: LammpsEngine, group: str, fd_step: float, path: str
        ) -> None:
            scratch.lmp.command(
                f"dynamical_matrix no_such_group eskm {fd_step} file {path}"
            )

        monkeypatch.setattr(
            LammpsHTSTExtension, "_run_dynamical_matrix", staticmethod(broken)
        )
        ext = LammpsHTSTExtension(search_engine)
        before = _engine_state(search_engine)
        tmp_before = _count_tmpdirs()
        with pytest.raises(Exception, match="dynamical matrix group"):
            ext.compute_event_prefactors(hop_request(free_radius=4.0))
        _assert_all_closed(scratch_log, 1)
        assert _count_tmpdirs() == tmp_before
        _assert_state_unchanged(search_engine, before)

    def test_python_error_after_scratch_exists_cleans_up(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An injected exception after the group and temp dir exist is not swallowed."""

        def injected(
            scratch: LammpsEngine, group: str, fd_step: float, path: str
        ) -> None:
            assert os.path.isdir(os.path.dirname(path))
            raise RuntimeError("injected after the scratch exists")

        monkeypatch.setattr(
            LammpsHTSTExtension, "_run_dynamical_matrix", staticmethod(injected)
        )
        ext = LammpsHTSTExtension(search_engine)
        tmp_before = _count_tmpdirs()
        with pytest.raises(RuntimeError, match="injected after"):
            ext.compute_event_prefactors(hop_request(free_radius=4.0))
        _assert_all_closed(scratch_log, 1)
        assert _count_tmpdirs() == tmp_before
        # The extension stays usable: a following valid call succeeds.
        monkeypatch.undo()
        good = ext.compute_event_prefactors(hop_request(free_radius=4.0))
        assert good.forward.ok and good.backward.ok

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("not a number\n", "cannot parse"),
            ("1.0 2.0 3.0\n", "expected"),
        ],
    )
    def test_unparsable_matrix_file_raises(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
        monkeypatch: pytest.MonkeyPatch,
        content: str,
        match: str,
    ) -> None:
        """Garbage or a wrong-sized file is a plumbing ``RuntimeError``."""

        def write_garbage(
            scratch: LammpsEngine, group: str, fd_step: float, path: str
        ) -> None:
            with open(path, "w", encoding="ascii") as handle:
                handle.write(content)

        monkeypatch.setattr(
            LammpsHTSTExtension, "_run_dynamical_matrix", staticmethod(write_garbage)
        )
        ext = LammpsHTSTExtension(search_engine)
        with pytest.raises(RuntimeError, match=match):
            ext.compute_event_prefactors(hop_request(free_radius=4.0))
        _assert_all_closed(scratch_log, 1)
        assert _count_tmpdirs() == 0

    def test_non_finite_matrix_is_a_kernel_rejection(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """NaN entries are returned to the kernel, which rejects NONFINITE_HESSIAN."""
        request = hop_request(free_radius=4.0)
        n_free = 19  # saddle-centred selection at free_radius 4

        def write_nan(
            scratch: LammpsEngine, group: str, fd_step: float, path: str
        ) -> None:
            with open(path, "w", encoding="ascii") as handle:
                for _ in range(3 * n_free * n_free):
                    handle.write("nan 0.0 0.0\n")

        monkeypatch.setattr(
            LammpsHTSTExtension, "_run_dynamical_matrix", staticmethod(write_nan)
        )
        result = LammpsHTSTExtension(search_engine).compute_event_prefactors(request)
        assert result.n_free == n_free
        for direction in (result.forward, result.backward):
            assert direction.status == "rejected"
            assert direction.reason_code is PrefactorRejection.NONFINITE_HESSIAN
        _assert_all_closed(scratch_log, 1)
        assert _count_tmpdirs() == 0

    # -- preflight ----------------------------------------------------------------

    def test_preflight_passes_on_this_build(
        self, search_engine: LammpsEngine, scratch_log: list[LammpsEngine]
    ) -> None:
        """PHONON present and the SW potential initialises in a scratch instance."""
        before = _engine_state(search_engine)
        report = LammpsHTSTExtension(search_engine).htst_preflight()
        assert report["phonon"] is True
        assert report["lammps_version"] == int(search_engine.lmp.version())
        assert report["pair_style"] == "sw"
        assert report["species"] == ("Si",)
        assert report["masses"] == search_engine.full_system.masses
        _assert_all_closed(scratch_log, 1)
        _assert_state_unchanged(search_engine, before)

    def test_preflight_rejects_a_build_without_phonon(
        self,
        search_engine: LammpsEngine,
        scratch_log: list[LammpsEngine],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A build without PHONON is refused by name; no FD fallback exists."""
        import lammps

        monkeypatch.setattr(lammps.lammps, "has_package", lambda self, name: False)
        with pytest.raises(RuntimeError, match="PHONON"):
            LammpsHTSTExtension(search_engine).htst_preflight()
        _assert_all_closed(scratch_log, 1)

    def test_preflight_needs_an_initialised_search_engine(
        self, sw_config: _SWConfig, scratch_log: list[LammpsEngine]
    ) -> None:
        """Before ``initialize_system`` there is no species map to check."""
        engine = LammpsEngine(config=sw_config, comm=None, engine_id=0)
        engine.start()
        try:
            with pytest.raises(RuntimeError, match="initialize_system"):
                LammpsHTSTExtension(engine).htst_preflight()
        finally:
            engine.close()
        assert scratch_log == []

    def test_preflight_uses_the_full_two_species_map(self) -> None:
        """A two-species LJ engine preflights with both species and masses."""
        atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=True)
        types = atoms.get_chemical_symbols()
        types[0] = "Fe"
        engine = LammpsEngine(config=_LJConfig(), comm=None, engine_id=0)
        engine.start()
        try:
            _initialize(
                engine, types, atoms.get_positions(), np.array(atoms.get_cell())
            )
            report = LammpsHTSTExtension(engine).htst_preflight()
        finally:
            engine.close()
        assert report["species"] == ("Fe", "Ni")
        assert len(report["masses"]) == 2

    # -- authoritative masses (contracts section 7d, N3) ----------------------

    def test_engine_mass_override_reaches_the_request_and_scales_nu0(
        self, sw_config: _SWConfig, si_hop: dict[str, Any]
    ) -> None:
        """The preflight species/mass map is what every live request carries.

        ``initialize_system(species=("Si",), masses=(30.0,))`` on the SW
        potential (its file sets no masses) keeps 30 amu; ``htst_preflight``
        reports it; a ``PrefactorService`` built with that map issues a
        request carrying 30 amu; the extension's forward ``nu0`` scales by
        ``sqrt(m_default / 30)`` relative to the default-mass request (the
        Vineyard ratio scales as ``m**-1/2`` for a single species), and the
        FD replay with ``request.masses_per_atom()`` agrees with eskm as the
        existing oracle does. ``m_default`` is the ASE mass the offline path
        emits (28.085 in this ASE; the contract's 28.0855 to 1e-4).
        """
        from mpi4py import MPI

        from pykmc.config import Config, RateConstantConfig
        from pykmc.rate_constant import create_rate_constant
        from pykmc.rate_constant.prefactors import PrefactorService
        from tests.lifecycle.conftest import FakeManager

        types = [str(t) for t in si_hop["types"]]
        engine = LammpsEngine(config=sw_config, comm=None, engine_id=0)
        engine.start()
        try:
            engine.initialize_parameters()
            engine.initialize_system(
                types=types,
                positions=si_hop["min1_positions"],
                cell=si_hop["cell"],
                pbc=_ALL_PERIODIC,
                species=("Si",),
                masses=(30.0,),
            )
            engine.initialize_potential()
            assert engine.full_system.masses == (30.0,)
            ext = LammpsHTSTExtension(engine)
            report = ext.htst_preflight()
            assert report["species"] == ("Si",) and report["masses"] == (30.0,)

            base = Config.from_ini_file(str(_ROOT / "tests" / "data" / "input.in"))
            config = base.model_copy(
                update={
                    "lammps": base.lammps.model_copy(
                        update={
                            "pair_style": sw_config.pair_style,
                            "pair_coeff": sw_config.pair_coeff,
                            "min_style": sw_config.min_style,
                            "frz_min": sw_config.frz_min,
                        }
                    ),
                    "rateconstant": RateConstantConfig(
                        style="htst", k0=1.0, free_radius=4.0
                    ),
                }
            )
            rate = create_rate_constant(config.rateconstant)
            geometry = dict(
                min1_positions=si_hop["min1_positions"],
                saddle_positions=si_hop["saddle_positions"],
                min2_positions=si_hop["min2_positions"],
                types=types,
                cell=si_hop["cell"],
                pbc=_ALL_PERIODIC,
                center_index=int(si_hop["central_atom_idx"]),
            )
            live = PrefactorService(
                config,
                FakeManager(ext.compute_event_prefactors),
                rate,
                species_masses=(report["species"], report["masses"]),
            )
            heavy_request = live.build_request(event_key=("mass", 30), **geometry)
            assert heavy_request.species == ("Si",)
            assert heavy_request.masses == (30.0,)
            assert np.all(heavy_request.masses_per_atom() == 30.0)
            offline = PrefactorService(
                config, FakeManager(ext.compute_event_prefactors), rate
            )
            default_request = offline.build_request(
                event_key=("mass", "ase"), **geometry
            )
            m_default = float(default_request.masses[0])
            assert m_default == pytest.approx(28.0855, rel=1e-4)

            heavy = live.compute([heavy_request])[heavy_request.event_key]
            light = offline.compute([default_request])[default_request.event_key]
            assert heavy.forward.ok and light.forward.ok
            assert heavy.n_free == light.n_free == 19
            assert heavy.forward.nu0_hz / light.forward.nu0_hz == pytest.approx(
                np.sqrt(m_default / 30.0), rel=1e-6
            )

            scratch = LammpsEngine(config=sw_config, comm=MPI.COMM_SELF, engine_id=9)
            scratch.start()
            try:
                _initialize(scratch, types, si_hop["min1_positions"], si_hop["cell"])
                fd = compute_event_prefactors(
                    heavy_request,
                    fd_hessian_fn(
                        lambda p: scratch.get_forces(positions=p),
                        heavy_request.masses_per_atom(),
                        heavy_request.settings.fd_step,
                    ),
                )
            finally:
                scratch.close()
            assert fd.method == "fd" and heavy.method == "lammps_eskm"
            assert fd.forward.ok
            assert heavy.forward.nu0_hz / fd.forward.nu0_hz == pytest.approx(
                1.0, rel=1e-9
            )
        finally:
            engine.close()
        assert _count_tmpdirs() == 0

    def test_empty_free_set_early_return_records_the_zone_provenance(
        self,
        search_engine: LammpsEngine,
        si_hop: dict[str, Any],
        scratch_log: list[LammpsEngine],
    ) -> None:
        """No vibrational DOF: the kernel's EMPTY_FREE_REGION result carries
        the same zone provenance as a computed one, without a scratch engine."""
        from pykmc.physics import ResolvedConstraints

        ext = LammpsHTSTExtension(search_engine)
        full_system = search_engine.full_system
        min1 = si_hop["min1_positions"].copy()
        cell = si_hop["cell"].copy()
        center = int(si_hop["central_atom_idx"])
        settings = HTSTSettings(free_radius=6.0, zone_radius=8.0)
        sphere = select_free_indices(min1, center, 6.0, cell, _ALL_PERIODIC)
        ids = tuple(range(len(min1)))
        constraints = ResolvedConstraints(
            ids,
            ids,
            tuple(int(i) for i in sphere),
            tuple(tuple(float(x) for x in min1[i]) for i in sphere),
        )
        # Every free-sphere atom is fixed and nothing moves, so the request is
        # valid while its common free set is empty.
        request = HTSTEventRequest(
            event_key=("ref", "empty-free"),
            min1_positions=min1.copy(),
            saddle_positions=min1.copy(),
            min2_positions=min1.copy(),
            types=tuple(str(t) for t in si_hop["types"]),
            species=full_system.species,
            masses=full_system.masses,
            cell=cell,
            pbc=_ALL_PERIODIC,
            center_index=center,
            settings=settings,
            constraints=constraints,
        )
        result = ext.compute_event_prefactors(request)
        assert result.forward.reason_code is PrefactorRejection.EMPTY_FREE_REGION
        assert result.backward.reason_code is PrefactorRejection.EMPTY_FREE_REGION
        assert result.n_free == 0 and scratch_log == []
        provenance = result.provenance
        assert provenance.method == "lammps_eskm"
        assert provenance.free_indices == ()
        zone = select_free_indices(min1, center, 8.0, cell, _ALL_PERIODIC)
        assert 0 < zone.size < len(min1)
        assert provenance.zone_indices == tuple(int(i) for i in zone), (
            "the early return must record the zone the settings select, "
            "exactly as a computed result does"
        )
        assert provenance.energies is None

    def test_force_model_mismatch_names_both_models(
        self,
        search_engine: LammpsEngine,
        hop_request: Callable[..., HTSTEventRequest],
        scratch_log: list[LammpsEngine],
    ) -> None:
        """A request declaring another potential is refused before any Hessian,
        and the refusal names the declared and the engine's force models."""
        from dataclasses import replace

        from pykmc.physics import EnginePhysics, PhysicalDescriptor

        ext = LammpsHTSTExtension(search_engine)
        request = hop_request(free_radius=6.0)
        declared = PhysicalDescriptor.from_config(
            SimpleNamespace(frozen_atoms=None),
            EnginePhysics.capture(_LJConfig(), request.species, request.masses),
            request.settings,
        )
        request = replace(request, descriptor=declared)
        request.validate()
        with pytest.raises(HTSTRequestError) as captured:
            ext.compute_event_prefactors(request)
        message = str(captured.value)
        assert "force model" in message
        assert "lj/cut" in message, message  # the request's declared model
        assert "sw" in message, message  # the engine's actual model
        assert scratch_log == []


@pytest.mark.mpi
class TestLammpsHTSTEngineMPI:
    """Run with ``mpirun -n 4``: a multi-rank search engine, no manager.

    Every rank calls both operations collectively through the engine; only the
    local root (world rank 0 here) computes, on its own ``COMM_SELF`` scratch,
    and the others return ``None`` at once. A collective on the search engine's
    communicator from rank 0 alone would hang this test.
    """

    @pytest.fixture(autouse=True)
    def require_mpi(self) -> None:
        """Skip without ``mpirun``."""
        from mpi4py import MPI

        if MPI.COMM_WORLD.Get_size() == 1:
            pytest.skip("requires mpirun -n N")
        yield
        MPI.COMM_WORLD.Barrier()

    def test_only_the_root_computes(
        self,
        sw_config: _SWConfig,
        si_hop: dict[str, Any],
    ) -> None:
        """Root result equals the serial extension's; other ranks get ``None``."""
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        types = [str(t) for t in si_hop["types"]]
        engine = LammpsEngine(config=sw_config, comm=comm, engine_id=1)
        engine.start()
        try:
            _initialize(engine, types, si_hop["min1_positions"], si_hop["cell"])
            ext = LammpsHTSTExtension(engine)
            request = HTSTEventRequest(
                event_key=("ref", 4),
                min1_positions=si_hop["min1_positions"].copy(),
                saddle_positions=si_hop["saddle_positions"].copy(),
                min2_positions=si_hop["min2_positions"].copy(),
                types=tuple(types),
                species=engine.full_system.species,
                masses=engine.full_system.masses,
                cell=si_hop["cell"].copy(),
                pbc=_ALL_PERIODIC,
                center_index=int(si_hop["central_atom_idx"]),
                settings=HTSTSettings(free_radius=6.0, zone_radius=10.0),
            )
            energy_before = engine.get_total_energy()
            report = ext.htst_preflight()
            result = ext.compute_event_prefactors(request)
            energy_after = engine.get_total_energy()
            if comm.Get_rank() == 0:
                assert report["phonon"] is True
                assert result.n_free == 37
                assert result.forward.ok and result.backward.ok
                assert energy_after == energy_before
                # comm=None would be MPI_COMM_WORLD inside the lammps wrapper (a
                # collective the other ranks never join); the reference must be
                # a genuine single-rank instance.
                serial = LammpsEngine(config=sw_config, comm=MPI.COMM_SELF, engine_id=0)
                serial.start()
                try:
                    _initialize(serial, types, si_hop["min1_positions"], si_hop["cell"])
                    reference = LammpsHTSTExtension(serial).compute_event_prefactors(
                        request
                    )
                finally:
                    serial.close()
                assert result.forward.nu0_hz == pytest.approx(
                    reference.forward.nu0_hz, rel=1e-6
                )
                assert result.backward.nu0_hz == pytest.approx(
                    reference.backward.nu0_hz, rel=1e-6
                )
            else:
                assert report is None
                assert result is None
                assert energy_after is None
        finally:
            engine.close()
        assert _count_tmpdirs() == 0
