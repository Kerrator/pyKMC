from unittest.mock import Mock

import numpy as np
import pytest

from pykmc import System
from pykmc.kmc import KMC
from pykmc.result import BasinOutput, Ok, ReconstructionOutput


def _toy_system(offset: float) -> System:
    return System(
        positions=np.array([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]], dtype=float),
        types=np.array(["Ni", "Ni"]),
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=np.array([True, True, True]),
        index=np.array([0, 1]),
    )


def test_accept_capped_basin_exit_uses_selected_exit_state():
    kmc = KMC(config=Mock())
    kmc.system = _toy_system(0.0)
    kmc.manager = Mock()
    kmc.manager.global_get_total_energy.return_value = -12.34

    exit_system = _toy_system(5.0)
    basin = Mock()
    basin.states = {
        7: Mock(system=exit_system),
    }

    basin_output = BasinOutput(
        initial_system_positions=_toy_system(1.0).positions,
        central_atom=0,
        saddle_positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        final_positions=np.array([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        neighbors=np.array([0, 1]),
        energy_barrier=0.2,
        k_tot=3.0,
        t_exit=4.0,
        exit_state=7,
        from_state=0,
        num_reference_event=11,
    )

    kmc._accept_capped_basin_exit(basin, basin_output)

    assert np.allclose(kmc.system.positions, exit_system.positions)
    assert kmc.total_energy == -12.34
    kmc.manager.global_get_total_energy.assert_called_once()


def test_apply_original_migration_event_restores_positions_and_total_energy():
    kmc = KMC(config=Mock())
    kmc.system = _toy_system(0.0)
    reconstructed_system = _toy_system(3.0)

    result_reconstruction = Ok(
        ReconstructionOutput(
            min1_positions=_toy_system(1.0).positions,
            saddle_positions=_toy_system(2.0).positions,
            min2_positions=reconstructed_system.positions,
            min2_etot=-7.5,
        )
    )

    kmc._apply_original_migration_event(result_reconstruction)

    assert np.allclose(kmc.system.positions, reconstructed_system.positions)
    assert kmc.total_energy == -7.5


def test_restart_file_uses_total_target_steps(tmp_path):
    restart_file = tmp_path / "restart_2.npz"
    np.savez(restart_file, last_step=2, last_time=1.5)

    config = Mock()
    config.control.restart_file = str(restart_file)
    config.control.n_steps = 5

    kmc = KMC(config=config)
    kmc.loggers = Mock()

    last_step, last_time = kmc._load_run_counters()

    assert last_step == 2
    assert last_time == pytest.approx(1.5)
    assert list(kmc._iter_kmc_steps(last_step)) == [3, 4, 5]
