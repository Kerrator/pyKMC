def transform_positions(positions, transformation_matrix, translation_matrix, permutation_matrix) : 
    transform_positions = positions @ transformation_matrix.T + translation_matrix 
    return transform_positions[permutation_matrix]
    