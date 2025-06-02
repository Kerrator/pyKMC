"""Manages Point Set Registration (shape matching) methods"""
import ira_mod 
import numpy as np
from .result import Result, ErrorInfo, PSROutput, Ok, Err, ErrorType
import pandas as pd 

class PointSetRegistration() : 
    """
    Methods 
    -------
    run() 
        run the point set registration
    ira(idx_cat, central_atom_index, rcutevent)
        use IRA to find the transformation matrix between the positions of the event and the central_atom_index environement
    """

    def __init__(self,  config, system, catalog, neighbors_list, idx_cat, central_atom_index) : 

        self.system = system 
        self.config = config        
        self.catalog = catalog
        self.neighbors_list = neighbors_list
        self.idx_cat = idx_cat
        self.central_atom_index = central_atom_index
        self.psr_style = self.config.psr.style


    def run(self) -> Result[PSROutput, ErrorInfo] : 
        """ 
        run the point set registration
        """ 
        match self.psr_style : 
            case 'ira' : 
                return self.ira(self.idx_cat, self.central_atom_index)
            case _: 
                raise Exception('Point set registration style unknown')
        
    
    def ira(self, idx_cat, central_atom_index) -> Result[PSROutput, ErrorInfo]: 
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

        Returns 
        ------- 
        if did not find transformation matrices return (None, None, None, None)
        else 
            rmat : (3,3) numpy.array of float
                rotation matrix 
            tr : (3,) numpy.array of float
                translation matrix 
            perm : list of int 
                list of permutation index
            dh : float 
                Hausdorff distance value of the match

        """
        #Initialize IRA
        ira = ira_mod.IRA() 

        #Event informations : 
        #coords2 = self.catalog.catalog.loc[idx_cat].at["initial_positions"] 
        coords2 = self.catalog.at["initial_positions"] 
        nat2 = len(coords2)

        #atom in the rcutevent around the central atom
        neighbor_list = self.neighbors_list.get_neighbors('rcut', central_atom_index) 

        coords1 = self.system.positions[neighbor_list]
        typ1 = np.array(self.system.types)[neighbor_list]

        typ2 = typ1 #If they have same topology id should be always true ?

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
        kmax_factor = self.config.ira.kmax_factor
        
        #Run ira to find transformation matrices
        try : 
            rmat, tr, perm, dh = ira.match( nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor )
             
            return Ok(PSROutput(rotation_matrix=rmat, 
                                translation_matrix=tr, 
                                permutation_matrix=perm, 
                                matching_score=dh))
        except : 
            return Err(ErrorInfo(type=ErrorType.PSR_NO_MATCH_FOUND, 
                                 message='IRA did not find a match')) 



