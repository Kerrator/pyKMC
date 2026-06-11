from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import concurrent.futures

import numpy as np
import pandas as pd
import pytest

from pykmc.config import Config
from pykmc.enginemanager.lmpi.lammps_operations import reload_potential
from pykmc.enginemanager.lmpi.pool import Manager
from pykmc.eventsearch import EventSearch
from pykmc.kmc import KMC
from pykmc.otfml import OTFMLController
from pykmc.refinement import Refinement
from pykmc.result import (
    Err,
    ErrorInfo,
    ErrorType,
    EventRefinementOutput,
    EventSearchOutput,
    Ok,
    RefinementTask,
)


class DummyLogger:
    def info(self, *_args, **_kwargs):
        return None

    def progress_bar(self, *_args, **_kwargs):
        return None

    def is_enabled_for(self, *_args, **_kwargs):
        return False


class SearchManagerStub:
    def __init__(self, result_batches):
        self.result_batches = list(result_batches)

    def partn_search(self, **_kwargs):
        batch = self.result_batches.pop(0)
        futures = []
        for result in batch:
            future = concurrent.futures.Future()
            future.set_result(result)
            futures.append(future)
        return futures


def build_search_output(atom_index: int, barrier: float) -> Ok:
    positions = np.zeros((3, 3), dtype=float)
    positions[0] = np.array([1.0, 1.0, 1.0])
    return Ok(
        EventSearchOutput(
            central_atom_index=atom_index,
            min1_positions=positions.copy(),
            saddle_positions=positions.copy(),
            min2_positions=positions.copy(),
            dE_forward=barrier,
            dE_backward=barrier + 0.1,
            move_atom_index=0,
        )
    )


@pytest.fixture
def otf_ini_template():
    return """
[Control]
initial_config = ./initial_config.xyz
n_steps = 1
engine = lammps

[Lammps]
pair_style = pair_style_cmd
pair_coeff = pair_coeff_cmd
setup_commands = [fix extra all something, neigh_modify every 1]
reload_commands = [unfix extra]

[AtomicEnvironment]
style = cna/graph
rnei = 2.8
rcut = 6.5

[EventSearch]
style = partn
nsearch = 1

[pARTn]
path_artnso = /tmp/libartn.so

[RateConstant]
style = constant
k0 = 1.0
T = 300.0

[PSR]
style = ira

[IRA]

[OTFML]
retrain_command = true
potential_file = potential.almtp
training_set_file = train.cfg
gamma_tolerance = 1.2
gamma_max = 25.0
enabled_phases = [search, refine, minimize]
"""


def test_otfml_config_parses_from_ini(tmp_path, otf_ini_template):
    ini_path = tmp_path / "input.in"
    ini_path.write_text(otf_ini_template, encoding="utf-8")

    config = Config.from_ini_file(str(ini_path))

    assert config.control.otfml is False
    assert config.otfml.retrain_command == "true"
    assert config.otfml.enabled_phases == ["search", "refine", "minimize"]
    assert config.lammps.setup_commands == [
        "fix extra all something",
        "neigh_modify every 1",
    ]
    assert config.lammps.reload_commands == ["unfix extra"]


def test_otfml_rejects_active_volume(tmp_path, otf_ini_template):
    ini_path = tmp_path / "input.in"
    ini_text = (
        otf_ini_template
        + """
[ActiveVolume]
ract = 8.0
rmov = 4.0
"""
    )
    ini_text = ini_text.replace(
        "engine = lammps", "engine = lammps\nactive_volume = True\notfml = True"
    )
    ini_path.write_text(ini_text, encoding="utf-8")

    with pytest.raises(ValueError, match="OTFML does not support active_volume=True"):
        Config.from_ini_file(str(ini_path))


def test_reload_potential_runs_reload_then_setup_commands():
    commands = []
    engine = SimpleNamespace(command=commands.append)
    config = SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="pair_style_cmd",
            pair_coeff="pair_coeff_cmd",
            reload_commands=["unfix extra"],
            setup_commands=["fix extra all something"],
        )
    )

    reload_potential(engine, config)

    assert commands == [
        "pair_style pair_style_cmd",
        "run 0",
    ]


def test_manager_reload_all_potentials_calls_each_session():
    sessions = [Mock(), Mock()]
    manager = Manager(sessions=sessions, global_session=Mock())
    config = Mock()

    manager.reload_all_potentials(config)

    for session in sessions:
        session.reload_potential.assert_called_once_with(config)


def test_event_search_retries_only_extrapolating_jobs(tmp_path):
    system = SimpleNamespace(
        positions=np.zeros((3, 3), dtype=float),
        cell=np.eye(3) * 10.0,
        types=np.array(["Ni", "Ni", "Ni"]),
    )
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        atomicenvironment=SimpleNamespace(rcut=6.5),
    )
    manager = SearchManagerStub(
        [
            [build_search_output(5, 0.2), build_search_output(6, 0.3)],
            [build_search_output(5, 0.9)],
        ]
    )
    event_search = EventSearch(config, system, manager, DummyLogger())
    otfml = OTFMLController(
        SimpleNamespace(
            config=SimpleNamespace(control=SimpleNamespace(otfml=False), otfml=None)
        )
    )

    event_search.execute([5, 6])
    event_search.results[0] = Err(
        ErrorInfo(
            type=ErrorType.EXTRAPOLATION,
            message="search extrapolated",
            variables={"central_atom_index": 5},
        )
    )
    retry_task_ids = otfml._collect_extrapolation_retry_ids(event_search.results)

    event_search.retry(retry_task_ids)

    assert len(event_search.results) == 2
    by_atom = {
        result.ok_value().central_atom_index: result for result in event_search.results
    }
    assert by_atom[5].ok_value().dE_forward == pytest.approx(0.9)
    assert by_atom[6].ok_value().dE_forward == pytest.approx(0.3)


def test_event_search_retry_removes_the_extrapolating_duplicate():
    system = SimpleNamespace(
        positions=np.zeros((3, 3), dtype=float),
        cell=np.eye(3) * 10.0,
        types=np.array(["Ni", "Ni", "Ni"]),
    )
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        atomicenvironment=SimpleNamespace(rcut=6.5),
    )
    manager = SearchManagerStub(
        [
            [
                build_search_output(5, 0.2),
                build_search_output(6, 0.3),
                build_search_output(5, 0.4),
            ],
            [build_search_output(5, 0.9)],
        ]
    )
    event_search = EventSearch(config, system, manager, DummyLogger())
    otfml = OTFMLController(
        SimpleNamespace(
            config=SimpleNamespace(control=SimpleNamespace(otfml=False), otfml=None)
        )
    )

    event_search.execute([5, 6, 5])
    event_search.results[2] = Err(
        ErrorInfo(
            type=ErrorType.EXTRAPOLATION,
            message="search extrapolated",
            variables={"central_atom_index": 5},
        )
    )

    retry_task_ids = otfml._collect_extrapolation_retry_ids(event_search.results)
    event_search.retry(retry_task_ids)

    barriers_for_atom_5 = [
        result.ok_value().dE_forward
        for result in event_search.results
        if result.is_ok() and result.ok_value().central_atom_index == 5
    ]
    assert barriers_for_atom_5 == [pytest.approx(0.2), pytest.approx(0.9)]


def test_refinement_retry_replaces_only_matching_job(monkeypatch, tmp_path):
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False, refine_thr=0.9999),
        eventsearch=SimpleNamespace(refined_energy_thr=0.05),
        psr=SimpleNamespace(matching_score_thr=0.1),
    )
    refinement = Refinement(
        config=config,
        loggers=DummyLogger(),
        system=None,
        neighbors_list=None,
        atomic_environment=None,
        manager=None,
    )

    dfevent1 = pd.Series({"idx_ref": 1, "k": 1.0})
    dfevent2 = pd.Series({"idx_ref": 2, "k": 2.0})
    execute_calls = []

    def fake_build_tasks(_df, _total_energy):
        return [
            RefinementTask(0, 10, 1, 0, dfevent1, 0.0, 0.0),
            RefinementTask(1, 11, 2, 0, dfevent2, 0.0, 0.0),
        ]

    def fake_run_tasks(tasks):
        execute_calls.append([task.task_id for task in tasks])
        if len(tasks) == 2:
            return {
                0: Err(
                    ErrorInfo(
                        type=ErrorType.EXTRAPOLATION,
                        message="refine extrapolated",
                        variables={
                            "central_atom_index": 10,
                            "num_reference_event": 1,
                            "symmetry_index": 0,
                        },
                    )
                ),
                1: Ok(
                    EventRefinementOutput(
                        11,
                        np.zeros((1, 3)),
                        2.0,
                        num_reference_event=2,
                        symmetry_index=0,
                    )
                ),
            }
        return {
            0: Ok(
                EventRefinementOutput(
                    10,
                    np.zeros((1, 3)),
                    9.0,
                    num_reference_event=1,
                    symmetry_index=0,
                )
            )
        }

    monkeypatch.setattr(refinement, "build_tasks", fake_build_tasks)
    monkeypatch.setattr(refinement, "_run_tasks", fake_run_tasks)
    otfml = OTFMLController(
        SimpleNamespace(
            config=SimpleNamespace(control=SimpleNamespace(otfml=False), otfml=None)
        )
    )

    refinement.execute(pd.DataFrame([dfevent1, dfevent2]), total_energy=0.0)
    retry_task_ids = otfml._collect_extrapolation_retry_ids(refinement.results)

    refinement.retry(retry_task_ids)

    assert execute_calls == [
        [0, 1],
        [0],
    ]
    assert len(refinement.results) == 2
    by_key = {
        (
            result.ok_value().central_atom_index,
            result.ok_value().num_reference_event,
        ): result
        for result in refinement.results
        if result.is_ok()
    }
    assert by_key[(10, 1)].ok_value().E_saddle == pytest.approx(9.0)
    assert by_key[(11, 2)].ok_value().E_saddle == pytest.approx(2.0)


def test_otfml_controller_retrains_reloads_and_retries_search(monkeypatch):
    config = SimpleNamespace(
        control=SimpleNamespace(otfml=True),
        otfml=SimpleNamespace(
            retrain_command="true",
            potential_file="potential.almtp",
            training_set_file="train.cfg",
            gamma_tolerance=1.2,
            gamma_max=25.0,
            launcher="nested",
            batch_args=None,
            runner_args="--oversubscribe",
            extra_args=None,
            sequential_eval=False,
            enabled_phases=["search", "refine", "minimize"],
        ),
    )
    session0 = Mock(session_id=1)
    session1 = Mock(session_id=2)
    global_session = Mock(session_id=0)
    manager = SimpleNamespace(
        using_global=False,
        sleeping_workers=Mock(return_value=nullcontext()),
        sessions=[session0, session1],
        reload_all_potentials=Mock(),
        global_reload_potential=Mock(),
        global_session=global_session,
        use_global=Mock(),
        use_local=Mock(),
        set_all_positions=Mock(),
    )
    kmc = SimpleNamespace(
        config=config,
        manager=manager,
        loggers=DummyLogger(),
        system=SimpleNamespace(positions=np.zeros((2, 3), dtype=float)),
    )
    minimize_mock = Mock()
    kmc._minimize_system_once = minimize_mock
    monkeypatch.setattr("pykmc.otfml.subprocess.run", Mock())

    event_search = SimpleNamespace(
        results=[
            Err(
                ErrorInfo(
                    type=ErrorType.EXTRAPOLATION,
                    message="search extrapolated",
                    variables={"central_atom_index": 5},
                )
            )
        ],
    )

    def retry(task_ids):
        event_search.results = [build_search_output(5, 0.9)]

    event_search.retry = Mock(side_effect=retry)
    controller = OTFMLController(kmc)

    controller.retry_extrapolating("search", event_search)

    retried_task_ids = event_search.retry.call_args.args[0]
    assert retried_task_ids == [0]
    session0.command.assert_any_call("undump extrapolative_structures_dump")
    session0.command.assert_any_call(
        "dump extrapolative_structures_dump all custom 1 extrapolative_dumps/extrapolating_dump.1.lammps id type x y z f_extrapolation_grade"
    )
    session1.command.assert_any_call(
        "dump extrapolative_structures_dump all custom 1 extrapolative_dumps/extrapolating_dump.2.lammps id type x y z f_extrapolation_grade"
    )
    global_session.command.assert_any_call(
        "dump extrapolative_structures_dump all custom 1 extrapolative_dumps/extrapolating_dump.0.lammps id type x y z f_extrapolation_grade"
    )
    manager.reload_all_potentials.assert_called_once()
    manager.global_reload_potential.assert_called_once()
    minimize_mock.assert_called_once()
    assert manager.use_local.call_count == 5
    assert manager.use_global.call_count == 2
    manager.set_all_positions.assert_called_once_with(kmc.system.positions)


def test_kmc_minimize_system_is_unchanged_when_otfml_disabled(
    config_system_single_type, monkeypatch
):
    kmc = KMC(config_system_single_type)
    minimize_once = Mock()
    monkeypatch.setattr(kmc, "_minimize_system_once", minimize_once)

    kmc.minimize_system()

    minimize_once.assert_called_once_with(positions=None, types=None)
