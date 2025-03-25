import pandas as pd


class Catalog : 

    def __init__(self, config) : 
        if config['Control']['catalog'] is None : 
            self._initialize_catalog(config) 
        else : 
            self.catalog = pd.read_pickle(config['Control']['catalog'])
    

    def _initialize_catalog(self, config) : 
        if config['Control']['reconstruction'] : 
            self.catalog = pd.DataFrame(columns=['event_id', 
                                                 'initial_positions', 
                                                 'saddle_positions', 
                                                 'final_positions', 
                                                 'energy_barrier', 
                                                 'k', 
                                                 'id_saddle',
                                                 'id_final', 
                                                 'move_atom_idx'])
        else : 
            self.catalog = pd.DataFrame(columns = ['atom_index', 
                                                   'final_positions',
                                                   'energy_barrier',
                                                   'k'])
            
    def save(self, outfile='catalog.pickle') : 
        self.catalog.to_pickle(outfile)