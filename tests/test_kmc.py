from unittest.mock import Mock

import numpy as np

from pykmc import System
from pykmc.kmc import KMC
from pykmc.result import Ok, ReconstructionOutput


def _toy_system(offset: float) -> System:
    return System(
        positions=np.array([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]], dtype=float),
        types=np.array(["Ni", "Ni"]),
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=np.array([True, True, True]),
        index=np.array([0, 1]),
    )


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
