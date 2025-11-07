import pytest 
import os 
from pykmc.basins import BasinStatesConnectivity, StateData
from pykmc import NeighborsList, AtomicEnvironment
import pandas as pd

@pytest.fixture 
def connectivity_table_Cu() :
    """Connectivity table for Copper system with 1SIA and 1Vac, with removed SIA translation event""" 
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    filepath = os.path.join(data_dir, "basin_connectivity_Cu_fake.pickle")

    connectivity_table = BasinStatesConnectivity()
    connectivity_table.df = pd.read_pickle(filepath)

    return connectivity_table

@pytest.fixture
def mock_state_data_Cu(system_Cu):
    """Fixture returning a dummy StateData for testing basin exploration."""
    system = system_Cu
    nl = NeighborsList(system=system, rnei=2.9, rcut=6.5)
    ae = AtomicEnvironment(style='cna/graph', neighbors_list=nl.neighbors_list["rnei"], environment_list=nl.neighbors_list["rcut"])
    state = StateData(
        system=system_Cu,
        environment =ae,
        neighbors_list=nl,
        transient=True,
        visited=False
    )
    return state