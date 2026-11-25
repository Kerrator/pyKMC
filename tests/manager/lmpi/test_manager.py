from pykmc.enginemanager.lmpi.pool import ManagerFactory
from pykmc import System, Config
import pytest
from pytest_lazy_fixtures import lf


class TestManager:
    def test_initialize_manager(self):
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

    @pytest.mark.parametrize(
        "system, config",
        [(lf("system_single_type_fcc"), lf("config_system_single_type"))],
    )
    def test_minimize_manager(self, system: System, config: Config):
        factory = ManagerFactory(
            n_sessions=config.control.n_sessions, use_rank_0=True, has_global=True
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
        f.result()
        manager.global_minimize(config)
        manager.close_all()
