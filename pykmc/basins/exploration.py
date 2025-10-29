from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd 
from dataclasses import dataclass, field
from pykmc import System, Config, NeighborsList, AtomicEnvironment, ReferenceEventTable
from typing import Optional, TYPE_CHECKING
from .connectivity import StatesConnectivity, BasinStatesConnectivity
from .detection import DetectorThreshold
if TYPE_CHECKING:
    from .basin import StateData


class Explorer(ABC) : 
    """Abstract class for basin exploration algorithms."""

    @abstractmethod
    def explore(self) -> bool:
        """Explore the basins."""
        pass
    
    #@abstractmethod
    #def get_connexion_table(self) -> pd.DataFrame : 
    #    """Get the connexion table in pandas DataFrame format""" 
    #    pass


class BasinGenericEventExplorer(Explorer) : 
    """Explorer that use only generic events of a reference table to explore the basin."""

    def __init__(self, config:Config,  reference_table: ReferenceEventTable, state_index:int = 0, start_index: int = 1) : 
        self.config = config
        self.reference_table: ReferenceEventTable = reference_table
        self.connectivity_table: StatesConnectivity = BasinStatesConnectivity()
        self.detector = DetectorThreshold() 

    def explore(self, state: "StateData", state_index:int = 0, start_index: int = 1) : 
        
        #Find all applicable events on the state 
        df_applicable_events = self.reference_table.has_id_subset_table(state.environment.atomic_environment_list)
        
        #Loop over all applicable events : 
        count = 0
        for idx, df_event in df_applicable_events.iterrows() : #Note : idx is the original index of the self.reference_table.table
            #check if df_event leads to transient state 
            is_transient = self.detector.detect(df_event, self.reference_table.table, self.config.basin.energy_thr)
            #All atoms on which we can apply the event : 
            l_atoms = state.environment.get_atoms_with_id(df_event["event_id"])
            #Loop over all atoms on which we can apply the event : 
            for at in l_atoms : 
                #Loop over symmetries : 
                for i in range(len(df_event.at["sym_matrix"])) : 
                    #for each symmetries add connectivity in table 
                    self.connectivity_table.add_connectivity(state=state_index, state_connexion=start_index+count, event_connexion=idx, central_atom=at, sym=i, transient=is_transient, dE=df_event["energy_barrier"], k=df_event["k"] )

                    #update count 
                    count +=1
    
    def get_connectivity_table(self) : 
        return self.connectivity_table.get_table()

    def clear(self): 
        self.connectivity_table.clear()