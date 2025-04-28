import pytest 
from pykmc import System
import numpy as np 
from copy import deepcopy
# System Fixtures 


@pytest.fixture
def config_system_single_type():
    return {
        'AtomicEnvironment': {
            'rnei': 3.01,   
            'rcut': 3.5, 
            'radd_cna' : 0,
        }
    }


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
