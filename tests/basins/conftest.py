import pytest 
import os 
from pykmc.basins import BasinStatesConnectivity
import pandas as pd

@pytest.fixture 
def connectivity_table_Cu() :
    """Connectivity table for Copper system with 1SIA and 1Vac, with removed SIA translation event""" 
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    filepath = os.path.join(data_dir, "basin_connectivity_Cu_fake.pickle")

    connectivity_table = BasinStatesConnectivity()
    connectivity_table.df = pd.read_pickle(filepath)

    return connectivity_table