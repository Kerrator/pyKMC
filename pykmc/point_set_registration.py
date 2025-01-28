import ira_mod 
import random
import numpy as np
import pandas as pd 

#TODO print psr DataFrame infos should be an option, self.ira should return the transformation matrix
#TODO Check if pbc problem is really fixed, and find better solution
#TODO style = 'ira' should not be hardcoded
#TODO typ = 'Ni' should not be hardcoded
#TODO deal with nat1 > nat2
#TODO deal with ira error ---> if error pass new step

class PointSetRegistration() : 
    """
    Define and run the point set registration procedure

    Attributes
    ----------
    system : System Object
        the system
    psr_style : str
        the point set registration style used, can be 'ira'
    idx_cat : int
        index of the event in the catalog on which we gonna perform the point set registration
    central_atom_index : int
        atom index of the system on which we gonna perform the point set registration
    rcutevent : float
        radial cutoff corresponding to the one use in the event search corresponding to en environment around the central atom that have been saved 
    dimension : int
        dimension of the system, by default 3
    nprocs : int, optional
        number of procs available, by default 1
    backend : str, optional
        parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

    Methods 
    -------
    run() 
        run the point set registration
    ira(idx_cat, central_atom_index, rcutevent)
        use IRA to find the transformation matrix between the positions of the event and the central_atom_index environement
    """

    def __init__(self,  system, psr_style, psr_parameters, idx_cat, central_atom_index, rcutevent, dimension, nprocs, backend, save) : 
        self.psr_style = psr_style
        self.system = system
        self.psr_parameters = psr_parameters
        self.idx_cat = idx_cat
        self.central_atom_index = central_atom_index
        self.rcutevent = rcutevent
        self.dimension = dimension
        self.backend = backend
        self.nprocs = nprocs 
        self.save = save

    def run(self) : 
        """ 
        run the point set registration
        """ 
        if self.psr_style == 'ira' : 
            rmat, tr, perm, dh = self.ira(self.idx_cat, self.central_atom_index, self.rcutevent)
            return rmat, tr, perm, dh
        else : 
            print('ERROR')
            return None
        
    
    def ira(self, idx_cat, central_atom_index, rcutevent) : 
        """
        Use IRA to extract rotation, translation, permutation matrix to apply on generic event

        Parameters
        ----------
        idx_cat : int
            index of the event in the catalog
        central_atom_index : int 
           index of the system's central atom 
        rcutevent : float
            radial cutoff used in the event_search
        """
        #Initialize IRA
        ira = ira_mod.IRA() 

        #Event informations : 
        id = self.system.catalog.loc[idx_cat].at['event_id']
        coords2 = self.system.catalog.loc[idx_cat].at["initial_positions"] 
        nat2 = len(coords2)
        typ2 = nat2*['Ni']

        #atom in the rcutevent around the central atom
        ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
        dist = self.system.get_distances(central_atom_index, ind, mic=True)
        neighbor_list = np.where(dist<rcutevent)[0]

        coords1 = self.system.get_positions()[neighbor_list]

        #unwrap if close to cell limits :
        alat = self.system.cell[0][0] 
        for i in range(len(coords1)) : 
            if np.linalg.norm(coords1[i][0] - self.system.positions[central_atom_index][0]) > alat/2 : 
               coords1[i][0] = coords1[i][0] + np.sign(self.system.positions[central_atom_index][0]-coords1[i][0])*alat 
            if np.linalg.norm(coords1[i][1] - self.system.positions[central_atom_index][1]) > alat/2 : 
                coords1[i][1] = coords1[i][1] + np.sign(self.system.positions[central_atom_index][1]-coords1[i][1])*alat
            if np.linalg.norm(coords1[i][2] - self.system.positions[central_atom_index][2]) > alat/2 : 
                coords1[i][2] = coords1[i][2] + np.sign(self.system.positions[central_atom_index][2]-coords1[i][2])*alat

        nat1 = len(coords1)
        typ1 = typ2
        kmax_factor = self.psr_parameters['kmax_factor']
        rmat, tr, perm, dh = ira.match( nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor )

        a = [[rmat, tr, perm, dh]]
        results = pd.DataFrame(a, columns=['R', 
                                        'T', 
                                        'P', 
                                        'dh'])
        if self.save : 
            results.to_pickle('psr_event_'+str(idx_cat)+'.pickle')
        return rmat, tr, perm, dh



