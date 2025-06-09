from .lammpsengine import LammpsEngine
import sys 
import os

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
                original_stdout_fd = os.dup(1)
                devnull = os.open(os.devnull, os.O_WRONLY)
                # Redirect stdout (fd 1) to /dev/null, only way to deal with pARTn error write
                os.dup2(devnull, 1)
                result =  self.engine.pARTn(system, central_atom_idx)
                # Restore original stdout (fd 1)
                os.dup2(original_stdout_fd, 1)
                os.close(original_stdout_fd)
                os.close(devnull)
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

