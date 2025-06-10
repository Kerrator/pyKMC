from .result import Result, EventSearchOutput, ErrorInfo
from .utils.geometry import translate
import numpy as np 


class EventSearch() : 

    def __init__(self, system, engine, loggers) : 

        self.system = system 
        self.engine = engine 
        self.loggers = loggers
        self.results = None

    def execute(self, central_atom_research_list: list[int] ) -> list[Result[EventSearchOutput, ErrorInfo]] : 
        self.results = []
        self.loggers.info('log', '\t :=> Searching {} reference events'.format(len(central_atom_research_list))) 
        for i, at_idx in enumerate(central_atom_research_list) : 
            event_search_output = self.engine.search_event(self.system, at_idx)
            self.results.append(event_search_output)
            self.loggers.progress_bar('progress', i+1, len(central_atom_research_list))
    
    def _center_event_positions(self, event_search_output: EventSearchOutput) : 
        #Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr 
        cell = self.system.cell
        ax, ay, az = cell[0][0], cell[1][1], cell[2][2] 
        #displacement 
        move_atom_idx = event_search_output.move_atom_index        
        dx, dy, dz = ax/2 - event_search_output.min1_positions[move_atom_idx][0],  ay/2 - event_search_output.min1_positions[move_atom_idx][1], az/2 - event_search_output.min1_positions[move_atom_idx][2]
        displacement = np.array([dx, dy, dz])
        event_search_output.min1_positions = translate(event_search_output.min1_positions, displacement, cell)
        event_search_output.saddle_positions = translate(event_search_output.saddle_positions, displacement, cell)
        event_search_output.min2_positions = translate(event_search_output.min2_positions, displacement, cell)
        return event_search_output   

    def get_successes_results(self) -> list[EventSearchOutput]: 
        return [self._center_event_positions(e.ok_value()) for e in self.results if e.is_ok()] 
    
