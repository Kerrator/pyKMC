from unittest.mock import Mock

import pykmc.enginemanager.lmpi.pool.factory as factory_module
from pykmc.enginemanager.lmpi.pool import Manager, ManagerFactory
from pykmc import System, Config
from mpi4py import MPI
import pytest
from pytest_lazy_fixtures import lf


def _skip_without_ranks(n_sessions: int, use_rank_0: bool) -> None:
    required_ranks = n_sessions if use_rank_0 else n_sessions + 1
    if MPI.COMM_WORLD.Get_size() < required_ranks:
        pytest.skip(f"requires mpirun with at least {required_ranks} ranks")


class TestManager: 

    def test_factory_rejects_rank0_engines(self, monkeypatch):
        class _FakeWorld:
            def Get_rank(self):
                return 0

            def Get_size(self):
                return 4

        monkeypatch.setattr(factory_module.MPI, "COMM_WORLD", _FakeWorld())

        with pytest.raises(ValueError, match="engine_use_rank_0=True"):
            ManagerFactory(n_sessions=1, use_rank_0=True, has_global=True)

    def test_close_all_uses_global_session_once_in_global_mode(self):
        sessions = [Mock(), Mock()]
        global_session = Mock()
        manager = Manager(sessions=sessions, global_session=global_session)

        manager.close_all()

        global_session.close.assert_called_once_with(wait_status=True)
        for session in sessions:
            session.close.assert_not_called()

    def test_broadcast_command_uses_global_session_in_global_mode(self):
        sessions = [Mock(), Mock()]
        global_session = Mock()
        manager = Manager(sessions=sessions, global_session=global_session)

        manager.broadcast_command("units metal")

        global_session.command.assert_called_once_with("units metal")
        for session in sessions:
            session.command.assert_not_called()

    def test_initialize_manager(self)  : 
        _skip_without_ranks(n_sessions=2, use_rank_0=False)
        factory = ManagerFactory(n_sessions=2, use_rank_0=False, has_global=True)
        manager = factory.launch()

        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.broadcast_command("units metal")
        manager.broadcast_command("log flush")
        manager.global_session.command("dimension 3")
        manager.global_session.command("log flush")
        manager.close_all()


    @pytest.mark.parametrize("system, config", [(lf("system_single_type_fcc"), lf("config_system_single_type"))])
    def test_minimize_manager(self, system: System, config: Config)  : 
        _skip_without_ranks(
            n_sessions=config.control.n_sessions,
            use_rank_0=config.control.engine_use_rank_0,
        )
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions,
            use_rank_0=config.control.engine_use_rank_0,
            has_global=True,
        )
        manager = factory.launch()
        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.initialize_sessions(config, system)
        manager.global_initialize_parameters()
        manager.global_initialize_system(system)
        manager.global_initialize_potential(config)
        f = manager.minimize(config)
        _ = f.result()
        manager.global_minimize(config)
        manager.close_all()
