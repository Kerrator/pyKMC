import ase.geometry 

def transform_positions(positions, transformation_matrix, translation_matrix, permutation_matrix) : 
    transform_positions = positions @ transformation_matrix.T + translation_matrix 
    return transform_positions[permutation_matrix]

def translate( positions, displacement, cell) : 
        positions += displacement 
        positions = ase.geometry.wrap_positions(positions=positions, cell=cell, pbc=True)
        positions[positions < 0 ] = 0
        return positions
    