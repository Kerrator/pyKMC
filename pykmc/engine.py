from .lammpsengine import LammpsEngine

class Engine() : 

    def __init__(self, config: dict) : 

        self.engine_type = config.control.engine
        match self.engine_type : 
            case 'lammps' : 
                self.engine = LammpsEngine(config)
            case _ : 
                raise ValueError("Engine type unknown")

    def minimize(self, system) : 
        minimized_positions, total_energy = self.engine.minimize(system)
        return minimized_positions, total_energy

    def search_event(self, system, central_atom_idx:int ) : 
        match self.engine.config.eventsearch.style :
            case 'partn' : 
                result =  self.engine.pARTn(system, central_atom_idx)
            case _ : 
                raise Exception('Event Search style unknown')
        return result 
    
    def refine_event(self, system, central_atom_idx:int) : 
        match self.engine.config.eventsearch.style :
            case 'partn' : 
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

