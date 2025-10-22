import pytest 
from pykmc import System, Config
import numpy as np 
from pykmc.basins import StatesConnectivity, BasinStatesConnectivity
from copy import deepcopy
import pandas as pd
import logging
# System Fixtures 

@pytest.fixture(scope="session")
def test_logger():
    """Logger for tests."""
    logger = logging.getLogger("tests")

    if not logger.handlers:
        handler = logging.StreamHandler()  
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@pytest.fixture
def config_system_single_type():
    config = Config.from_ini_file("./tests/data/input.in")
    return config

@pytest.fixture
def system_single_type_fcc() -> System:
    system = System()

    a = 3.52 
    repeat = 4

    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]) * a

    positions = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                shift = np.array([i, j, k]) * a
                for atom in basis:
                    positions.append(atom + shift)

    system.positions = np.array(positions)
    system.types = ['Ni'] * len(system.positions)
    system.cell = np.array([
        [repeat * a, 0.0, 0.0],
        [0.0, repeat * a, 0.0],
        [0.0, 0.0, repeat * a]
    ])
    system.pbc = np.array([True, True, True])
    system.index = np.arange(len(system.positions))

    return system

@pytest.fixture
def system_single_type_fcc_vacancy(system_single_type_fcc: System) -> System:

    system = deepcopy(system_single_type_fcc)
    #remove atom 
    system.positions = np.delete(system.positions, 0, axis=0)
    system.types = np.delete(system.types, 0, axis=0)
    system.index = np.delete(system.index, 0, axis=0)

    return system

@pytest.fixture
def mock_statesconnectivity():
    """Fixture returning a real StatesConnectivity instance with a fake connectivity table."""
    state_connectivity = StatesConnectivity()

    fake_df = pd.DataFrame({
        "state": [0, 0, 1, 2],
        "state_connexion": [1, 2, 0, 0],
        "event_connexion": [12, 34, 13, 35],
        "central_atom": [345, 7, 911, 20],
        "sym": [0, 1, 0, 0],
        "transient": [True, False, True, False],
    })

    state_connectivity.df = fake_df

    return state_connectivity

@pytest.fixture
def mock_basinstatesconnectivity():
    """Fixture returning a real BasinStatesConnectivity instance with a fake connectivity table."""
    state_connectivity = BasinStatesConnectivity()

    fake_df = pd.DataFrame({
        "state": [0, 0, 1, 2],
        "state_connexion": [1, 2, 0, 0],
        "event_connexion": [12, 34, 13, 35],
        "central_atom": [345, 7, 911, 20],
        "sym": [0, 1, 0, 0],
        "transient": [True, False, True, False],
    })

    state_connectivity.df = fake_df

    return state_connectivity