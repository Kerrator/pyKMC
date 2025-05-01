import ira_mod
import numpy as np

def unique_symmetries(initial_positions, final_positions, sym_thr) : 
    """ 
    """
    #Find all symmetries of initial_positions 
    nat = len(initial_positions) 
    typ = nat*[1]
    
    sofi = ira_mod.SOFI()
    sym = sofi.compute(nat, typ, initial_positions, sym_thr) #sym data ira object

    #Find unique symmetries
        #Displacment event matrix 
    displacements = initial_positions - final_positions 
        
    unique_displacements = [displacements]
    unique_sym_index = [] 

    for i in range(len(sym.matrix)) : #Loop over all symmetries 
        is_duplicated = False 
        #Apply symmetry to displacements event matrix
        new_displacements = displacements @ sym.matrix[i].T
        new_displacements = new_displacements[sym.perm[i]]

        for disp in unique_displacements : #Check if alreay in unique_displacements 
            if np.allclose(disp, new_displacements) : 
                is_duplicated = True 
                break
        
        if not is_duplicated : #if new unique symmetry 
            unique_sym_index.append(i) #add symmtry to unique 
            unique_displacements.append(new_displacements)

    sym_matrix = np.array([sym.matrix[i] for i in unique_sym_index])
    sym_perm = np.array([sym.perm[i] for i in unique_sym_index])
     
    return sym_matrix, sym_perm 