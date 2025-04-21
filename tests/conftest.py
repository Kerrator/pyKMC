import pytest 
from pykmc import System
import numpy as np 

# System Fixtures 

@pytest.fixture
def system_single_type() -> System : 
    system = System() 
    system.types = ['Ni', 'Ni', 'Ni', 'Ni']
    system.positions = np.array([[0.  , 0.  , 0.  ],
                                 [0.  , 1.76, 1.76],
                                 [1.76, 0.  , 1.76],
                                 [1.76, 1.76, 0.  ]])
    system.cell = np.array([3.52, 3.52, 3.52])
    system.pbc = np.array('True, True, True') 
    system.index = np.array([0,1,2,3])

    return system
