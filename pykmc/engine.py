from .lammpsengine import LammpsEngine

class Engine : 

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
    def compute_distances(self, system) : 
        self.engine.compute_distances(system)

    def neighbors(self, system) : 
        self.engine.neighbors(system)

