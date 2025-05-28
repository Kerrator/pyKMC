from .lammpsengine import LammpsEngine
from abc import ABC, abstractmethod 


class BaseEngine(ABC) : 

    @abstractmethod 
    def minimize(self, system) : 
        pass 

    @abstractmethod 
    def search_event(self, system, central_atom_idx) : 
        pass 

    @abstractmethod
    def refine_event(self, system, central_atom_idx) : 
        pass
    
    @abstractmethod
    def compute_potential_energy(self, system) : 
        pass

    def compute_distances(self, system) : 
        pass
    
    @abstractmethod
    def neighbors(self, system) : 
        pass




class Engine(BaseEngine) : 

    def __init__(self, config: dict) : 

        self.engine_type = config['Control']['engine']
        match self.engine_type : 
            case 'lammps' : 
                self.engine = LammpsEngine(config)
            case _ : 
                raise ValueError("Engine type unknown")

    def minimize(self, system) : 
        result = self.engine.minimize(system)
        return result

    def search_event(self, system, central_atom_idx:int ) : 
        match self.engine.config_event_search['style'] :
            case 'pARTn' : 
                result =  self.engine.pARTn(system, central_atom_idx)
            case _ : 
                raise Exception('Event Search style unknown')
        return result 
    
    def refine_event(self, system, central_atom_idx:int) : 
        match self.engine.config_event_search['style'] :
            case 'pARTn' : 
                result = self.engine.pARTn_refine_event(system, central_atom_idx)
            case _ : 
                raise Exception('Event Search style unknown')
        return result
    def compute_potential_energy(self, system) : 
        potential_energy = self.engine.compute_potential_energy(system)
        return potential_energy
    def compute_distances(self, system) : 
        self.engine.compute_distances(system)

    def neighbors(self, system) : 
        self.engine.neighbors(system)

