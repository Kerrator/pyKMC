"""KMC orchestration of the prefactor lifecycle (composition, ordering, clock).

Drives real ``KMC.run`` steps with the engine-, search- and refinement stages
stubbed, a fake manager answering ``compute_event_prefactors`` and the real
``_select_event``/``rejection_free``/``total_time`` path. No LAMMPS, no MPI.
"""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import pykmc.run as run_module
from pykmc import NeighborsList
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.initializer import Initializer
from pykmc.kmc import KMC
from pykmc.rate_constant import rate_from_prefactor
from pykmc.rate_constant.prefactors import PrefactorService
from pykmc.result import EventRefinementOutput, EventSearchOutput, Ok
from tests.lifecycle.conftest import (
    DATA_INPUT,
    FakeManager,
    accepted,
    event_prefactors,
)

SEED = 2024
HOP = np.array([1.2, 0.3, 0.0])


class _Recorder:
    """Logger double recording step lines and log messages."""

    def __init__(self) -> None:
        self.step_lines: list[dict[str, Any]] = []
        self.messages: list[tuple[str, str]] = []

    def info(self, name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record an info message."""
        self.messages.append((name, str(msg)))

    def warning(self, name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Record a warning."""
        self.messages.append((name, str(msg)))

    def error(self, *args: Any, **kwargs: Any) -> None:
        """Swallow errors."""

    def new_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow blank lines."""

    def progress_bar(self, *args: Any, **kwargs: Any) -> None:
        """Swallow progress output."""

    def events_file_step_first_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file lines."""

    def events_applicable_info_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file lines."""

    def events_basin_info_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file lines."""

    def table_line_info_kmc(
        self,
        name: str,
        step: int,
        delta_t: float,
        total_time: float,
        num_reference_event: Any,
        energy_barrier: Any,
        k_event: Any,
        k_tot: Any,
        total_energy: Any,
        cpu_time: float = 0.0,
        wall_time: float = 0.0,
    ) -> None:
        """Record the step line exactly as ``KMC.run`` evaluated it."""
        self.step_lines.append(
            {
                "step": step,
                "delta_t": delta_t,
                "total_time": total_time,
                "num_reference_event": num_reference_event,
                "energy_barrier": energy_barrier,
                "k": k_event,
                "k_tot": k_tot,
            }
        )


class _FakeEventSearch:
    """Event-search double returning canned successes."""

    def __init__(self, successes: list[EventSearchOutput] | None = None) -> None:
        self._successes = successes or []
        self.results: list[Any] = []

    def get_successes_results(self) -> list[EventSearchOutput]:
        """Return the canned search results."""
        return self._successes


class _FakeRefinement:
    """Refinement double returning canned refined events."""

    def __init__(self, outputs: list[EventRefinementOutput]) -> None:
        self._outputs = outputs
        self.results: list[Any] = []

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return the canned refined events."""
        return self._outputs


def _step_config(style: str, k0: float) -> Any:
    """One-step, no-basin, no-recycle config with the given rate style."""
    from pykmc.config import Config, RateConstantConfig

    config = Config.from_ini_file(DATA_INPUT)
    control = config.control.model_copy(
        update={"n_steps": 1, "basin": False, "recycle": False}
    )
    rate = RateConstantConfig(style=style, k0=k0, T=config.rateconstant.T)
    return config.model_copy(update={"control": control, "rateconstant": rate})


def _dummy_reference_row(table: ReferenceEventTable, system: Any) -> None:
    """Insert one resolved row (id 0) so the loop does not close on an empty table."""
    pos = system.positions
    fwd, _ = table._build_event_series(
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.5,
        cell=system.cell,
        types=list(system.types),
    )
    fwd["idx_ref"] = 0
    fwd["idx_backward"] = 0
    table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)
    if table.uses_prefactors:
        table._patch_row(0, accepted(5.0e12))


def _refined(
    system: Any,
    neighbors_list: NeighborsList,
    atom: int,
    dE: float,
    offset: float = 0.1,
    **estimate: Any,
) -> EventRefinementOutput:
    """Build a refinement output with neighbour-cropped geometry shifted by ``offset``."""
    neighbors = np.asarray(neighbors_list.get_neighbors("rcut", atom), dtype=int)
    pos = np.asarray(system.positions, dtype=float)
    return EventRefinementOutput(
        central_atom_index=atom,
        saddle_positions=pos[neighbors] + offset,
        E_saddle=dE,
        min2_positions=pos[neighbors] + 2.0 * offset,
        dE_forward=dE,
        num_reference_event=0,
        **estimate,
    )


def _run_one_step(
    config: Any,
    system: Any,
    manager: FakeManager,
    refined_outputs: list[EventRefinementOutput],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    search_successes: list[EventSearchOutput] | None = None,
    on_refine: Any = None,
) -> tuple[KMC, _Recorder, dict[str, Any]]:
    """Run one real ``KMC.run`` step with the engine stages stubbed."""
    system = copy.deepcopy(system)
    sim = KMC(config, manager=manager)
    sim.system = system
    sim.loggers = _Recorder()
    if sim.uses_event_prefactors:
        sim.prefactor_service = PrefactorService(config, manager, sim.rate_constant)
    sim.reference_table = ReferenceEventTable(
        config, prefactor_service=sim.prefactor_service
    )
    _dummy_reference_row(sim.reference_table, system)
    sim.visited_environments = {"crystal"}
    positions = np.array(system.positions, copy=True)
    restart: dict[str, Any] = {}

    def fake_refinements(subset: Any, existing_pairs: Any = None) -> _FakeRefinement:
        if on_refine is not None:
            on_refine(sim, subset)
        return _FakeRefinement(refined_outputs)

    monkeypatch.setattr(sim, "minimize_system", lambda: None)
    monkeypatch.setattr(
        sim, "execute_event_searches", lambda atoms: _FakeEventSearch(search_successes)
    )
    if search_successes is None:
        monkeypatch.setattr(sim, "add_reference_events", lambda results: [])
    monkeypatch.setattr(sim, "execute_refinements", fake_refinements)
    monkeypatch.setattr(
        sim,
        "_reconstruction_active_event",
        lambda idx, table: Ok(
            types.SimpleNamespace(
                min1_positions=positions.copy(),
                saddle_positions=positions.copy(),
                min2_positions=positions.copy(),
                min2_etot=-1.0,
            )
        ),
    )
    monkeypatch.setattr(sim, "_save", lambda: None)
    monkeypatch.setattr(sim, "_append_snapshot_to_trajectory", lambda: None)
    monkeypatch.setattr(
        sim,
        "_save_restart_file",
        lambda step, total_time: restart.update(step=step, total_time=total_time),
    )
    monkeypatch.setattr(sim, "_close", lambda: None)
    monkeypatch.chdir(tmp_path)
    random.seed(SEED)
    sim.run()
    return sim, sim.loggers, restart


@pytest.fixture
def fcc_neighbors(system_single_type_fcc: Any, constant_config: Any) -> NeighborsList:
    """Build the step's rcut neighbour list on the test crystal."""
    return NeighborsList(
        system_single_type_fcc,
        constant_config.atomicenvironment.rnei,
        constant_config.atomicenvironment.rcut,
    )


class TestComposition:
    """Manager injection and the initialisation guard."""

    def test_constructor_takes_the_manager(self, constant_config: Any) -> None:
        """KMC(config, manager=...) stores it; the attribute assignment still works."""
        manager = FakeManager()
        assert KMC(constant_config, manager=manager).manager is manager
        kmc = KMC(constant_config)
        assert kmc.manager is None
        kmc.manager = manager
        assert kmc.manager is manager

    def test_initialize_refuses_a_missing_manager(self, constant_config: Any) -> None:
        """Initializer.initialize raises before touching anything else."""
        with pytest.raises(RuntimeError, match="KMC.manager is None"):
            Initializer(KMC(constant_config)).initialize()

    def test_initialize_proceeds_once_the_manager_is_assigned(
        self, constant_config: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a manager the guard passes and initialisation continues."""

        class Sentinel(Exception):
            pass

        def boom(self: Any) -> None:
            raise Sentinel()

        monkeypatch.setattr(Initializer, "initialize_loggers", boom)
        kmc = KMC(constant_config)
        kmc.manager = FakeManager()
        with pytest.raises(Sentinel):
            Initializer(kmc).initialize()

    def test_preflight_broadcast_only_for_htst(
        self, constant_config: Any, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """htst_preflight follows initialize_potential in htst; absent in constant."""
        for config, expected in (
            (
                constant_config,
                [
                    "start",
                    "initialize_parameters",
                    "initialize_system",
                    "initialize_potential",
                ],
            ),
            (
                htst_config,
                [
                    "start",
                    "initialize_parameters",
                    "initialize_system",
                    "initialize_potential",
                    "htst_preflight",
                ],
            ),
        ):
            manager = FakeManager()
            kmc = KMC(config, manager=manager)
            kmc.system = system_single_type_fcc
            Initializer(kmc).initialize_engine()
            assert [op for op, _ in manager.broadcasts] == expected
            assert [op for op, _ in manager.group_calls] == expected[:4]

    def test_prefactor_service_built_only_for_htst(
        self, constant_config: Any, htst_config: Any
    ) -> None:
        """The service exists for htst (with Hz settings) and is None for constant."""
        kmc = KMC(htst_config, manager=FakeManager())
        kmc.loggers = _Recorder()
        Initializer(kmc).initialize_prefactor_service()
        assert isinstance(kmc.prefactor_service, PrefactorService)
        assert kmc.prefactor_service.settings.nu0_min_hz == 1.0e12
        Initializer(kmc).initialize_reference_table()
        assert kmc.reference_table.prefactor_service is kmc.prefactor_service

        kmc = KMC(constant_config, manager=FakeManager())
        kmc.loggers = _Recorder()
        Initializer(kmc).initialize_prefactor_service()
        assert kmc.prefactor_service is None

    @pytest.mark.parametrize("delta,warned", [(1.0, True), (0.0, False), (-1.0, False)])
    def test_free_radius_beyond_rcut_is_warned_once(
        self, htst_config: Any, delta: float, warned: bool
    ) -> None:
        """free_radius > rcut means non-stationary site geometries: one warning."""
        from pykmc.config import RateConstantConfig

        rcut = htst_config.atomicenvironment.rcut
        rate = RateConstantConfig(
            style="htst", k0=1.0, T=htst_config.rateconstant.T, free_radius=rcut + delta
        )
        config = htst_config.model_copy(update={"rateconstant": rate})
        kmc = KMC(config, manager=FakeManager())
        kmc.loggers = _Recorder()
        Initializer(kmc).initialize_prefactor_service()
        hits = [
            m for _, m in kmc.loggers.messages if "free_radius" in m and "rcut" in m
        ]
        assert len(hits) == (1 if warned else 0)
        if warned:
            assert f"free_radius = {rcut + delta} A exceeds" in hits[0]


class TestRunWiring:
    """run.py registers the HTST extension only for htst/rpa and injects the manager."""

    @staticmethod
    def _ini(tmp_path: Path, style: str) -> Path:
        text = Path(DATA_INPUT).read_text()
        text = text.replace("style = constant", f"style = {style}").replace(
            "k0 = 1e12", "k0 = 1.0"
        )
        ini = tmp_path / "input.in"
        ini.write_text(text)
        return ini

    def _drive(
        self, monkeypatch: pytest.MonkeyPatch, ini: Path, manager: Any
    ) -> tuple[dict[str, Any], list[tuple[Any, Any]]]:
        created: dict[str, Any] = {}
        kmc_calls: list[tuple[Any, Any]] = []

        class FakeFactory:
            def __init__(self, **kwargs: Any) -> None:
                created.update(kwargs)

            def launch(self) -> Any:
                return manager

        class FakeKMC:
            def __init__(self, config: Any, manager: Any = None) -> None:
                kmc_calls.append((config, manager))

            def _initialize(self) -> None:
                pass

            def run(self) -> None:
                pass

        monkeypatch.setattr(run_module, "EngineManagerFactory", FakeFactory)
        monkeypatch.setattr(run_module, "KMC", FakeKMC)
        monkeypatch.setattr(sys, "argv", ["pykmc", "-in", str(ini)])
        run_module.main()
        return created, kmc_calls

    def test_constant_registers_no_extension_and_never_imports_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No extension, and the htst_lammps module is never imported."""
        monkeypatch.setitem(sys.modules, "pykmc.engine.htst_lammps", None)
        manager = FakeManager()
        created, kmc_calls = self._drive(
            monkeypatch, self._ini(tmp_path, "constant"), manager
        )
        assert created["engine_extensions"] is None
        assert kmc_calls[0][1] is manager

    @pytest.mark.parametrize("style", ["htst", "rpa"])
    def test_htst_registers_the_extension(
        self, style: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lazily imported LammpsHTSTExtension is handed to the factory."""

        class FakeExtension:
            pass

        module = types.ModuleType("pykmc.engine.htst_lammps")
        module.LammpsHTSTExtension = FakeExtension
        monkeypatch.setitem(sys.modules, "pykmc.engine.htst_lammps", module)
        manager = FakeManager()
        created, kmc_calls = self._drive(
            monkeypatch, self._ini(tmp_path, style), manager
        )
        assert created["engine_extensions"] == [FakeExtension]
        assert kmc_calls == [(kmc_calls[0][0], manager)]
        assert kmc_calls[0][0].rateconstant.style == style

    def test_non_root_rank_creates_no_kmc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A None manager (worker rank) never constructs KMC."""
        monkeypatch.setitem(sys.modules, "pykmc.engine.htst_lammps", None)
        _, kmc_calls = self._drive(monkeypatch, self._ini(tmp_path, "constant"), None)
        assert kmc_calls == []


class TestClockParity:
    """constant k0 = 5.0 ps^-1 and htst nu0 = 5e12 Hz drive an identical clock."""

    def _outputs(
        self, system: Any, neighbors: NeighborsList, htst: bool, refined: str = "F"
    ) -> list[EventRefinementOutput]:
        estimate = (
            {"nu0_hz": 5.0e12, "nu0_status": "ok", "nu0_source": "reference"}
            if htst
            else {}
        )
        return [
            _refined(system, neighbors, 0, 0.5, refined=refined, **estimate),
            _refined(system, neighbors, 5, 0.7, refined=refined, **estimate),
        ]

    def test_delta_t_and_total_time_are_identical(
        self,
        system_single_type_fcc: Any,
        fcc_neighbors: NeighborsList,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Same seeded draws, same rates, same delta_t and accumulated seconds."""
        const_cfg = _step_config("constant", k0=5.0)
        htst_cfg = _step_config("htst", k0=1.0)
        const_manager = FakeManager()
        htst_manager = FakeManager()

        _, const_log, const_restart = _run_one_step(
            const_cfg,
            system_single_type_fcc,
            const_manager,
            self._outputs(system_single_type_fcc, fcc_neighbors, htst=False),
            monkeypatch,
            tmp_path,
        )
        _, htst_log, htst_restart = _run_one_step(
            htst_cfg,
            system_single_type_fcc,
            htst_manager,
            self._outputs(system_single_type_fcc, fcc_neighbors, htst=True),
            monkeypatch,
            tmp_path,
        )

        assert len(const_log.step_lines) == 1 and len(htst_log.step_lines) == 1
        c, h = const_log.step_lines[0], htst_log.step_lines[0]
        assert h["delta_t"] == c["delta_t"]
        assert h["total_time"] == c["total_time"]
        assert h["k"] == c["k"] and h["k_tot"] == c["k_tot"]
        assert h["num_reference_event"] == c["num_reference_event"] == 0
        assert htst_restart["total_time"] == const_restart["total_time"]
        # the clock really advanced, in seconds (delta_t in ps * 1e-12)
        assert c["delta_t"] > 0.0
        assert c["total_time"] == pytest.approx(c["delta_t"])
        T = const_cfg.rateconstant.T
        assert c["k_tot"] == pytest.approx(
            rate_from_prefactor(5.0, 0.5, T) + rate_from_prefactor(5.0, 0.7, T)
        )
        # zero HTST jobs in constant mode, none for unrefined rows in htst mode
        assert const_manager.prefactor_requests == []
        assert htst_manager.prefactor_requests == []

    def test_1e13_hz_resolves_to_10_per_ps(
        self, system_single_type_fcc: Any, fcc_neighbors: NeighborsList
    ) -> None:
        """1e13 Hz becomes a 10.0 ps^-1 prefactor on the active row."""
        table = ActiveEventTable(_step_config("htst", k0=1.0))
        table.add_events(
            _refined(
                system_single_type_fcc,
                fcc_neighbors,
                0,
                0.5,
                nu0_hz=1.0e13,
                nu0_status="ok",
            )
        )
        row = table.table.iloc[0]
        assert row["k_prefactor"] == 10.0
        assert row["k"] == rate_from_prefactor(10.0, 0.5, table.config.rateconstant.T)


class TestStepOrdering:
    """Reference resolution precedes refinement; site requests follow dedup."""

    def test_reference_rates_resolved_before_refinement(
        self,
        system_single_type_fcc: Any,
        fcc_neighbors: NeighborsList,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When refinement runs, every reference row is already resolved."""
        config = _step_config("htst", k0=1.0)
        manager = FakeManager(
            lambda req: event_prefactors(req.event_key, accepted(5e12), accepted(3e12))
        )
        pos = np.asarray(system_single_type_fcc.positions, dtype=float)
        min2 = pos.copy()
        min2[8] += HOP
        saddle = pos.copy()
        saddle[8] += 0.5 * HOP
        search = EventSearchOutput(
            central_atom_index=8,
            min1_positions=pos.copy(),
            saddle_positions=saddle,
            min2_positions=min2,
            dE_forward=2.0,
            dE_backward=1.5,
            move_atom_index=8,
            cell=np.asarray(system_single_type_fcc.cell, dtype=float),
            types=list(system_single_type_fcc.types),
        )
        seen: dict[str, Any] = {}

        def capture(sim: KMC, subset: Any) -> None:
            table = sim.reference_table.table
            seen["statuses"] = list(table["nu0_status"])
            seen["k"] = {
                int(r): float(k)
                for r, k in zip(table["idx_ref"], table["k"], strict=True)
            }
            seen["requests"] = len(manager.prefactor_requests)

        outputs = [
            _refined(
                system_single_type_fcc,
                fcc_neighbors,
                0,
                0.5,
                refined="F",
                nu0_hz=5e12,
                nu0_status="ok",
            )
        ]
        sim, log, _ = _run_one_step(
            config,
            system_single_type_fcc,
            manager,
            outputs,
            monkeypatch,
            tmp_path,
            search_successes=[search],
            on_refine=capture,
        )
        T = config.rateconstant.T
        assert seen["requests"] == 1
        assert "pending" not in seen["statuses"]
        assert seen["k"][1] == rate_from_prefactor(5.0, 2.0, T)
        assert seen["k"][2] == rate_from_prefactor(3.0, 1.5, T)
        assert any(
            "HTST prefactors:" in msg for name, msg in log.messages if name == "log"
        )
        assert manager.prefactor_requests[0].pbc == tuple(
            bool(p) for p in system_single_type_fcc.pbc
        )

    def test_site_requests_follow_dedup(
        self,
        system_single_type_fcc: Any,
        fcc_neighbors: NeighborsList,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Duplicate refined rows are removed before the site batch is submitted."""
        config = _step_config("htst", k0=1.0)
        manager = FakeManager(
            lambda req: event_prefactors(req.event_key, accepted(6e12), accepted(6e12))
        )
        dup = _refined(
            system_single_type_fcc,
            fcc_neighbors,
            0,
            0.5,
            refined="T",
            nu0_hz=5e12,
            nu0_status="ok",
        )
        # A distinct saddle on an overlapping environment (offset 0.5 vs 0.1):
        # the symmetric-duplicate check compares the shared atoms' saddle
        # positions, so equal offsets would make it a duplicate of ``dup``.
        other = _refined(
            system_single_type_fcc,
            fcc_neighbors,
            5,
            0.7,
            offset=0.5,
            refined="T",
            nu0_hz=5e12,
            nu0_status="ok",
        )
        sim, log, _ = _run_one_step(
            config,
            system_single_type_fcc,
            manager,
            [dup, dup, other],
            monkeypatch,
            tmp_path,
        )
        assert [r.event_key[0] for r in manager.prefactor_requests] == ["site", "site"]
        assert sorted(r.center_index for r in manager.prefactor_requests) == [0, 5]
        summary_lines = [m for n, m in log.messages if "site attempts this step=2" in m]
        assert len(summary_lines) == 1
        assert "(ok=2, rejected=0)" in summary_lines[0]
        assert log.step_lines[0]["k"] in (
            rate_from_prefactor(6.0, 0.5, config.rateconstant.T),
            rate_from_prefactor(6.0, 0.7, config.rateconstant.T),
        )


class TestConstantPathIsolation:
    """The constant path constructs no service, no request and imports no HTST code."""

    def test_constant_step_never_touches_htst_modules(self, tmp_path: Path) -> None:
        """A subprocess proves the constant lifecycle imports nothing from pykmc.htst."""
        script = textwrap.dedent(
            """
            import sys
            import numpy as np
            import pykmc
            htst_like = lambda m: m.startswith("pykmc.htst") or m == "pykmc.rate_constant.prefactors"
            before = {m for m in sys.modules if htst_like(m)}
            assert "pykmc.rate_constant.prefactors" not in before
            from pykmc.config import Config
            from pykmc.kmc import KMC
            from pykmc.initializer import Initializer
            from pykmc.event_table import ActiveEventTable, ReferenceEventTable
            from pykmc.result import EventRefinementOutput, EventSearchOutput

            class M:
                def broadcast(self, *a, **k):
                    pass
                def submit(self, *a, **k):
                    raise AssertionError("constant style submitted a job")

            config = Config.from_ini_file("tests/data/input.in")
            kmc = KMC(config, manager=M())
            Initializer(kmc).initialize_prefactor_service()
            assert kmc.prefactor_service is None
            a = 3.52
            basis = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]]) * a
            pos = np.array([b + np.array([i, j, k]) * a for i in range(3) for j in range(3) for k in range(3) for b in basis])
            cell = np.eye(3) * 3 * a
            types = ["Ni"] * len(pos)
            ref = ReferenceEventTable(config, prefactor_service=None)
            ev = EventSearchOutput(central_atom_index=0, min1_positions=pos, saddle_positions=pos, min2_positions=pos, dE_forward=0.5, dE_backward=0.5, move_atom_index=0, cell=cell, types=types)
            res = ref.add_events([ev], pbc=np.array([True, True, True]))
            assert res[0].is_ok(), res
            act = ActiveEventTable(config, prefactor_service=None)
            act.add_events(EventRefinementOutput(central_atom_index=0, saddle_positions=pos[:5], E_saddle=0.5, min2_positions=pos[:5], dE_forward=0.5, num_reference_event=0, refined="T"))
            after = {m for m in sys.modules if htst_like(m)}
            assert after == before, sorted(after - before)
            assert "pykmc.rate_constant.prefactors" not in sys.modules
            print("CONSTANT_PATH_OK", pykmc.__file__)
            """
        )
        root = Path.cwd()
        env = dict(os.environ, PYTHONPATH=str(root))
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "CONSTANT_PATH_OK" in proc.stdout
        assert str(root) in proc.stdout


class TestOutputFormats:
    """The events-file block is format-identical between constant and htst."""

    def test_events_info_block_is_unchanged_by_htst_columns(
        self,
        constant_config: Any,
        htst_config: Any,
        system_single_type_fcc: Any,
        fcc_neighbors: NeighborsList,
    ) -> None:
        """Identical rates give a byte-identical ``EventsInfo`` block in both styles."""
        from dataclasses import fields

        from pykmc.info_simulation import info_active_events
        from pykmc.result import EventsInfo

        assert [f.name for f in fields(EventsInfo)] == [
            "types",
            "central_atom",
            "initial_topologies",
            "reference_events",
            "dE_forward",
            "dE_backward",
            "dE_asym",
            "k",
            "dra_i",
            "dra_f",
            "refined",
        ]
        const_cfg = _step_config("constant", k0=5.0)
        htst_cfg = _step_config("htst", k0=1.0)
        blocks = []
        for config, estimate in (
            (const_cfg, {}),
            (htst_cfg, {"nu0_hz": 5.0e12, "nu0_status": "ok"}),
        ):
            reference = ReferenceEventTable(config)
            _dummy_reference_row(reference, system_single_type_fcc)
            active = ActiveEventTable(config)
            active.add_events(
                [
                    _refined(
                        system_single_type_fcc,
                        fcc_neighbors,
                        0,
                        0.5,
                        refined="T",
                        **estimate,
                    ),
                    _refined(
                        system_single_type_fcc,
                        fcc_neighbors,
                        5,
                        0.7,
                        refined="F",
                        **estimate,
                    ),
                ]
            )
            blocks.append(
                info_active_events(
                    system_single_type_fcc.types, reference, active
                ).output_msg()
            )
        assert blocks[0] == blocks[1]
        assert blocks[0].splitlines()[0].split() == [
            "Types",
            "Central",
            "Atom",
            "Ref",
            "Event",
            "dE",
            "forward",
            "dE",
            "backward",
            "dE",
            "asym",
            "k",
            "dra_i",
            "dra_f",
            "Refined",
        ]
