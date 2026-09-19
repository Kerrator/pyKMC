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
    PREFLIGHT_REPORT,
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
    full_saddle = pos.copy()
    full_saddle[neighbors] += offset
    return EventRefinementOutput(
        central_atom_index=atom,
        saddle_positions=full_saddle[neighbors],
        E_saddle=dE,
        min2_positions=pos[neighbors] + 2.0 * offset,
        dE_forward=dE,
        num_reference_event=0,
        full_saddle_positions=full_saddle if estimate.get("refined") == "T" else None,
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
            assert kmc.htst_preflight is None
            Initializer(kmc).initialize_engine()
            assert [op for op, _ in manager.broadcasts] == expected
            assert [op for op, _ in manager.group_calls] == expected[:4]
            if expected[-1] == "htst_preflight":
                # broadcast returns nothing: the root's report comes back
                # through the Future of the same operation (N3).
                assert [op for op, _ in manager.submitted] == ["htst_preflight"]
                assert kmc.htst_preflight["species"] == ("Ni",)
                assert kmc.htst_preflight["masses"] == (58.6934,)
            else:
                assert manager.submitted == []
                assert kmc.htst_preflight is None

    def test_initialize_engine_refuses_a_missing_preflight_report(
        self, htst_config: Any, system_single_type_fcc: Any
    ) -> None:
        """A None (non-root) or mapless report is an error, never a silent ASE map."""
        for report in (None, {"phonon": True}, {"species": ("Ni",), "masses": ()}):
            kmc = KMC(htst_config, manager=FakeManager(preflight_report=report))
            kmc.system = system_single_type_fcc
            with pytest.raises(RuntimeError, match="htst_preflight"):
                Initializer(kmc).initialize_engine()

    def test_prefactor_service_built_only_for_htst(
        self, constant_config: Any, htst_config: Any
    ) -> None:
        """The service exists for htst (with Hz settings) and is None for constant."""
        kmc = KMC(htst_config, manager=FakeManager())
        kmc.loggers = _Recorder()
        with pytest.raises(RuntimeError, match="initialize_engine"):
            Initializer(kmc).initialize_prefactor_service()  # no preflight yet
        kmc.htst_preflight = dict(PREFLIGHT_REPORT)
        Initializer(kmc).initialize_prefactor_service()
        assert isinstance(kmc.prefactor_service, PrefactorService)
        assert kmc.prefactor_service.settings.nu0_min_hz == 1.0e12
        assert kmc.prefactor_service.species_masses == (("Ni",), (58.6934,))
        Initializer(kmc).initialize_reference_table()
        assert kmc.reference_table.prefactor_service is kmc.prefactor_service

        kmc = KMC(constant_config, manager=FakeManager())
        kmc.loggers = _Recorder()
        Initializer(kmc).initialize_prefactor_service()
        assert kmc.prefactor_service is None

    @pytest.mark.parametrize("delta", [1.0, 0.0, -1.0])
    def test_free_radius_is_independent_of_rcut(
        self, htst_config: Any, delta: float
    ) -> None:
        """Site requests use the full refined saddle: no rcut warning in any case."""
        from pykmc.config import RateConstantConfig

        rcut = htst_config.atomicenvironment.rcut
        rate = RateConstantConfig(
            style="htst", k0=1.0, T=htst_config.rateconstant.T, free_radius=rcut + delta
        )
        config = htst_config.model_copy(update={"rateconstant": rate})
        kmc = KMC(config, manager=FakeManager())
        kmc.loggers = _Recorder()
        kmc.htst_preflight = dict(PREFLIGHT_REPORT)
        Initializer(kmc).initialize_prefactor_service()
        assert not [m for _, m in kmc.loggers.messages if "rcut" in m]
        ready = [m for _, m in kmc.loggers.messages if "prefactor service ready" in m]
        assert len(ready) == 1
        assert "centred on the saddle geometry" in ready[0]
        assert "engine masses (amu): Ni=58.6934" in ready[0]


class TestExitStatus:
    """``_close`` chooses the status and ``run.py`` propagates it after shutdown."""

    def test_close_exits_zero_on_completion_and_one_on_failure(
        self, htst_config: Any
    ) -> None:
        """Normal completion -> SystemExit(0); an aborted simulation -> SystemExit(1)."""
        for failed, code in ((False, 0), (True, 1)):
            manager = FakeManager()
            kmc = KMC(htst_config, manager=manager)
            kmc.loggers = _Recorder()
            with pytest.raises(SystemExit) as info:
                kmc._close(failed=failed)
            assert info.value.code == code
            assert manager.shutdowns == 1  # workers are shut down before exiting
            assert ("log", ":=> End of simulation") in kmc.loggers.messages
        manager = FakeManager()
        kmc = KMC(htst_config, manager=manager)
        kmc.loggers = _Recorder()
        with pytest.raises(SystemExit) as info:
            kmc._close()  # the default is a normal completion
        assert info.value.code == 0

    def test_abort_paths_close_with_failure(
        self,
        htst_config: Any,
        system_single_type_fcc: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """'No events found' and 'all reconstructions failed' both use failed=True."""
        manager = FakeManager()
        kmc = KMC(_step_config("constant", k0=5.0), manager=manager)
        kmc.system = copy.deepcopy(system_single_type_fcc)
        kmc.loggers = _Recorder()
        kmc.reference_table = ReferenceEventTable(kmc.config)
        kmc.visited_environments = set()
        seen: list[bool] = []

        def fake_close(failed: bool = False) -> None:
            seen.append(failed)
            raise SystemExit(1 if failed else 0)

        monkeypatch.setattr(kmc, "_close", fake_close)
        monkeypatch.setattr(kmc, "minimize_system", lambda: None)
        monkeypatch.setattr(
            kmc, "execute_event_searches", lambda atoms: _FakeEventSearch([])
        )
        monkeypatch.setattr(kmc, "_append_snapshot_to_trajectory", lambda: None)
        with pytest.raises(SystemExit) as info:
            kmc.run()  # empty reference table after the first search
        assert info.value.code == 1 and seen == [True]

        # every reconstruction failing is the other abort path
        table = ActiveEventTable(kmc.config)
        table.table = pd.DataFrame(
            {
                "atom_index": [0],
                "saddle_positions": [np.zeros((1, 3))],
                "final_positions": [np.zeros((1, 3))],
                "energy_barrier": [0.5],
                "k": [1.0],
                "num_reference_event": [0],
                "refined": ["T"],
            }
        )
        kmc.reference_table.table = pd.DataFrame(
            {"idx_ref": [0], "event_id": ["X"], "idx_backward": [0]}
        )
        monkeypatch.setattr(kmc, "_select_event", lambda t: (0, 1.0, 1.0))
        monkeypatch.setattr(
            kmc,
            "_reconstruction_active_event",
            lambda idx, t: types.SimpleNamespace(
                is_ok=lambda: False,
                err_value=lambda: types.SimpleNamespace(message="boom"),
            ),
        )
        with pytest.raises(SystemExit) as info:
            kmc.reconstruction(table)
        assert info.value.code == 1 and seen == [True, True]

    @pytest.mark.parametrize("code", [0, 1])
    def test_run_propagates_the_exit_status_after_shutting_down(
        self, code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``run.main`` re-raises SystemExit with the status; workers are shut down."""
        manager = FakeManager()
        aborted: list[int] = []

        class FakeFactory:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def launch(self) -> Any:
                return manager

        class FakeKMC:
            def __init__(self, config: Any, manager: Any = None) -> None:
                self.manager = manager

            def _initialize(self) -> None:
                pass

            def run(self) -> None:
                # KMC._close: shutdown first, then exit with the status
                self.manager.shutdown()
                sys.exit(code)

        class FakeComm:
            def Get_size(self) -> int:
                return 2

            def Abort(self, status: int) -> None:
                aborted.append(status)

        ini = TestRunWiring._ini(tmp_path, "constant")
        monkeypatch.setattr(run_module, "EngineManagerFactory", FakeFactory)
        monkeypatch.setattr(run_module, "KMC", FakeKMC)
        monkeypatch.setattr(run_module.MPI, "COMM_WORLD", FakeComm())
        monkeypatch.setattr(sys, "argv", ["pykmc", "-in", str(ini)])
        with pytest.raises(SystemExit) as info:
            run_module.main()
        assert info.value.code == code
        assert manager.shutdowns >= 1  # idempotent on the real manager
        assert aborted == []  # a SystemExit is never an MPI abort

    def test_run_still_aborts_the_communicator_on_other_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The BaseException boundary is unchanged: comm.Abort(1) then re-raise."""
        aborted: list[int] = []

        class FakeFactory:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def launch(self) -> Any:
                return FakeManager()

        class FakeKMC:
            def __init__(self, config: Any, manager: Any = None) -> None:
                pass

            def _initialize(self) -> None:
                raise RuntimeError("engine died")

            def run(self) -> None:
                pass

        class FakeComm:
            def Get_size(self) -> int:
                return 2

            def Abort(self, status: int) -> None:
                aborted.append(status)

        ini = TestRunWiring._ini(tmp_path, "constant")
        monkeypatch.setattr(run_module, "EngineManagerFactory", FakeFactory)
        monkeypatch.setattr(run_module, "KMC", FakeKMC)
        monkeypatch.setattr(run_module.MPI, "COMM_WORLD", FakeComm())
        monkeypatch.setattr(sys, "argv", ["pykmc", "-in", str(ini)])
        with pytest.raises(RuntimeError, match="engine died"):
            run_module.main()
        assert aborted == [1]

    def test_run_aborts_when_the_safety_net_shutdown_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A shutdown failing inside the SystemExit handler still aborts the comm.

        A SystemExit raised without a prior ``_close`` reaches the handler with
        the workers still waiting; if that shutdown itself raises, the failure
        must go through ``comm.Abort(1)`` like any other, or the un-notified
        ranks hang under mpirun.
        """
        aborted: list[int] = []

        class BrokenManager(FakeManager):
            def shutdown(self) -> None:
                super().shutdown()
                raise RuntimeError("session send failed")

        manager = BrokenManager()

        class FakeFactory:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def launch(self) -> Any:
                return manager

        class FakeKMC:
            def __init__(self, config: Any, manager: Any = None) -> None:
                pass

            def _initialize(self) -> None:
                pass

            def run(self) -> None:
                sys.exit(0)  # raised outside _close: nothing shut down yet

        class FakeComm:
            def Get_size(self) -> int:
                return 2

            def Abort(self, status: int) -> None:
                aborted.append(status)

        ini = TestRunWiring._ini(tmp_path, "constant")
        monkeypatch.setattr(run_module, "EngineManagerFactory", FakeFactory)
        monkeypatch.setattr(run_module, "KMC", FakeKMC)
        monkeypatch.setattr(run_module.MPI, "COMM_WORLD", FakeComm())
        monkeypatch.setattr(sys, "argv", ["pykmc", "-in", str(ini)])
        with pytest.raises(RuntimeError, match="session send failed"):
            run_module.main()
        assert manager.shutdowns == 1 and aborted == [1]


class TestSeed:
    """``control.seed`` makes the searched atoms reproducible."""

    @staticmethod
    def _draw(config: Any, seed: int | None, n_atoms: int = 60) -> list[int]:
        control = config.control.model_copy(update={"seed": seed})
        kmc = KMC(config.model_copy(update={"control": control}), manager=FakeManager())
        kmc.inactive_ae = None
        kmc.atomic_environment = types.SimpleNamespace(
            atomic_environment_list=["A", "B"] * (n_atoms // 2)
        )
        return kmc.central_atoms_research(["A", "B"], nsearch=12)

    def test_same_seed_same_selection(self, htst_config: Any) -> None:
        """Two KMC objects built with one seed draw the same central atoms."""
        first = self._draw(htst_config, 11)
        second = self._draw(htst_config, 11)
        assert first == second and len(first) == 24
        assert self._draw(htst_config, 12) != first

    def test_seed_also_fixes_the_numpy_stream(self, htst_config: Any) -> None:
        """NumPy's global generator (basin exit draws) is seeded as well."""
        self._draw(htst_config, 11)
        a = np.random.random(3)
        self._draw(htst_config, 11)
        assert np.array_equal(np.random.random(3), a)

    def test_seed_is_optional_and_parsed_from_text(self, htst_config: Any) -> None:
        """The field defaults to None and accepts the INI string form."""
        from pykmc.config import ControlConfig

        assert htst_config.control.seed is None
        parsed = ControlConfig.model_validate(
            {"initial_config": "x.xyz", "n_steps": 1, "engine": "lammps", "seed": "7"}
        )
        assert parsed.seed == 7
        assert "zseed" in ControlConfig.model_fields["seed"].description

    def test_new_environments_are_sorted_through_the_real_method(
        self, htst_config: Any
    ) -> None:
        """``KMC.get_new_environments`` returns the real method's ids sorted.

        ``AtomicEnvironment.get_new_environments`` builds its list from a set,
        so its own order follows the per-process string hash; with forty
        distinct ids an accidentally sorted set order is all but impossible,
        so this fails whenever the call site stops sorting.
        """
        from pykmc.atomic_environment import AtomicEnvironment

        kmc = KMC(htst_config, manager=FakeManager())
        kmc.loggers = _Recorder()
        ids = [f"env-{i:02d}-{'x' * (i % 7)}" for i in range(40)]
        ae = AtomicEnvironment.__new__(AtomicEnvironment)
        ae.atomic_environment_list = [ids[i % 40] for i in range(200)] + ["cr"]
        kmc.atomic_environment = ae
        kmc.visited_environments = {"cr", ids[3]}
        new = kmc.get_new_environments()
        assert new == sorted(set(ids) - {ids[3]}) and len(new) == 39
        assert ("log", "\t :=> 39 new atomic environments found") in (
            kmc.loggers.messages
        )

    def test_atomic_environment_returns_new_ids_sorted(self) -> None:
        """``AtomicEnvironment.get_new_environments`` itself sorts (contract 7c)."""
        from pykmc.atomic_environment import AtomicEnvironment

        ids = [f"env-{i:02d}-{'y' * (i % 5)}" for i in range(40)]
        ae = AtomicEnvironment.__new__(AtomicEnvironment)
        ae.atomic_environment_list = [ids[(7 * i) % 40] for i in range(200)] + ["cr"]
        new = ae.get_new_environments({"cr", ids[3], ids[17]})
        assert isinstance(new, list)
        assert new == sorted(set(ids) - {ids[3], ids[17]}) and len(new) == 38
        assert ae.get_new_environments(set(ids) | {"cr"}) == []

    def test_seed_reproduces_the_selection_across_hash_seeds(self) -> None:
        """Interpreters with different ``PYTHONHASHSEED`` pick the same atoms.

        Each subprocess builds a real ``AtomicEnvironment`` holding sixteen
        distinct environment ids, calls ``KMC.get_new_environments`` (the real
        method underneath) and ``central_atoms_research`` under
        ``control.seed``, and prints the order and the selection. mpirun gives
        every rank its own random hash seed, so neither may depend on it.
        """
        script = textwrap.dedent(
            """
            from pykmc.atomic_environment import AtomicEnvironment
            from pykmc.config import Config
            from pykmc.kmc import KMC

            class M:
                def broadcast(self, *a, **k):
                    pass

            class L:
                def info(self, *a, **k):
                    pass

            config = Config.from_ini_file("tests/data/input.in")
            control = config.control.model_copy(update={"seed": 20260918})
            kmc = KMC(config.model_copy(update={"control": control}), manager=M())
            kmc.loggers = L()
            kmc.inactive_ae = None
            letters = "abcdefghijklmnop"
            ids = ["hash%02d_%s" % (i, letters[i] * 24) for i in range(16)]
            ae = AtomicEnvironment.__new__(AtomicEnvironment)
            ae.atomic_environment_list = [ids[i % 16] for i in range(160)]
            ae.atomic_environment_list += ["crystal"] * 40
            kmc.atomic_environment = ae
            kmc.visited_environments = {"crystal"}
            new = kmc.get_new_environments()
            picked = kmc.central_atoms_research(new, nsearch=3)
            print("ORDER", " ".join(e[:6] for e in new))
            print("PICKED", " ".join(map(str, picked)))
            """
        )
        root = Path.cwd()
        outputs: list[list[str]] = []
        for hash_seed in ("0", "1", "2"):
            env = dict(os.environ, PYTHONPATH=str(root), PYTHONHASHSEED=hash_seed)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert proc.returncode == 0, proc.stdout + proc.stderr
            lines = [
                line
                for line in proc.stdout.splitlines()
                if line.startswith(("ORDER ", "PICKED "))
            ]
            assert len(lines) == 2, proc.stdout
            outputs.append(lines)
        assert outputs[0][0] == "ORDER " + " ".join(f"hash{i:02d}" for i in range(16))
        assert len(outputs[0][1].split()) == 1 + 16 * 3
        assert outputs[1] == outputs[0] and outputs[2] == outputs[0], outputs


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
        assert manager.prefactor_backward_flags == [False, False]
        summary_lines = [m for n, m in log.messages if "site attempts this step=2" in m]
        assert len(summary_lines) == 1
        assert "(ok=2, rejected=0, no_geometry=0)" in summary_lines[0]
        assert "pending=0 stale=0;" in summary_lines[0]
        assert "hessian requests this step=2" in summary_lines[0]
        assert "prefactor wall=" in summary_lines[0] and summary_lines[0].endswith(" s")
        assert log.step_lines[0]["k"] in (
            rate_from_prefactor(6.0, 0.5, config.rateconstant.T),
            rate_from_prefactor(6.0, 0.7, config.rateconstant.T),
        )

    def test_crop_only_refined_rows_are_counted_as_no_geometry(
        self,
        system_single_type_fcc: Any,
        fcc_neighbors: NeighborsList,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A refined row without its full saddle keeps its estimate; the step says so (F1)."""
        config = _step_config("htst", k0=1.0)
        manager = FakeManager(
            lambda req: event_prefactors(req.event_key, accepted(6e12), accepted(6e12))
        )
        crop_only = _refined(
            system_single_type_fcc,
            fcc_neighbors,
            0,
            0.5,
            refined="T",
            nu0_hz=5e12,
            nu0_status="ok",
        )
        crop_only.full_saddle_positions = None
        sim, log, _ = _run_one_step(
            config, system_single_type_fcc, manager, [crop_only], monkeypatch, tmp_path
        )
        assert manager.prefactor_requests == []
        summary_lines = [m for n, m in log.messages if "HTST prefactors:" in m]
        assert len(summary_lines) == 1
        assert (
            "site attempts this step=1 (ok=0, rejected=0, no_geometry=1)"
            in (summary_lines[0])
        )
        assert "active sources reference=1 site=0 k0=0" in summary_lines[0]
        assert log.step_lines[0]["k"] == rate_from_prefactor(
            5.0, 0.5, config.rateconstant.T
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


class TestReconstructionPurgeLabels:
    """The failed row is dropped by a label that is still valid (contracts section 7d, F4)."""

    def test_dangling_row_is_removed_before_the_reference_purge_relabels(
        self,
        htst_config: Any,
        system_single_type_fcc: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A dangling failed row must not cost a valid row through a stale label.

        Catalogue: row 5 (links to the absent id 7) and row 9. Active rows:
        ref 5, ref 7 (dangling: no catalogue entry) and ref 9. Selecting the
        dangling row first fails; the purge of id 7 removes row 5 by closure
        (7 itself has no row), drops the ref-5 active row and relabels the
        table. The failed ref-7 row must be the one removed explicitly, so
        the ref-9 row survives and is reconstructed.
        """
        kmc = KMC(htst_config, manager=FakeManager())
        kmc.system = copy.deepcopy(system_single_type_fcc)
        kmc.loggers = _Recorder()
        kmc.reference_table = ReferenceEventTable(kmc.config)
        kmc.reference_table.table = pd.DataFrame(
            {"idx_ref": [5, 9], "event_id": ["X", "Y"], "idx_backward": [7, 9]}
        )
        table = ActiveEventTable(kmc.config)
        crop = np.zeros((1, 3))
        table.table = pd.DataFrame(
            {
                "atom_index": [0, 1, 2],
                "saddle_positions": [crop, crop, crop],
                "final_positions": [crop, crop, crop],
                "energy_barrier": [0.5, 0.5, 0.5],
                "k": [1.0, 1.0, 1.0],
                "num_reference_event": [5, 7, 9],
                "refined": ["T", "T", "T"],
                "k_prefactor": [1.0, 1.0, 1.0],
                "nu0": [np.nan, np.nan, np.nan],
                "nu0_status": ["legacy", "legacy", "legacy"],
                "nu0_reason": ["", "", ""],
                "nu0_source": ["k0", "k0", "k0"],
                "nu0_site_attempted": [True, True, True],
            }
        )

        def select(active: ActiveEventTable) -> tuple[int, float, float]:
            refs = active.table["num_reference_event"].astype(int)
            dangling = active.table.index[refs == 7]
            label = int(dangling[0]) if len(dangling) else int(active.table.index[0])
            return label, 1.0, 1.0

        def reconstruct(label: int, active: ActiveEventTable) -> Any:
            ref = int(active.table.loc[label, "num_reference_event"])
            if ref == 7:
                return types.SimpleNamespace(
                    is_ok=lambda: False,
                    err_value=lambda: types.SimpleNamespace(message="boom"),
                )
            return types.SimpleNamespace(is_ok=lambda: True, ok_value=lambda: ref)

        monkeypatch.setattr(kmc, "_select_event", select)
        monkeypatch.setattr(kmc, "_reconstruction_active_event", reconstruct)
        result, _, _, label, err_reference, err_ae = kmc.reconstruction(table)

        assert result.is_ok() and result.ok_value() == 9
        assert list(table.table["num_reference_event"].astype(int)) == [9]
        assert int(table.table.loc[label, "num_reference_event"]) == 9
        assert err_reference == [5] and err_ae == ["X"]
        assert list(kmc.reference_table.table["idx_ref"].astype(int)) == [9]
