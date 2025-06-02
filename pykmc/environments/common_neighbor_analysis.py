"""Determine cristalline environments."""

def cna(neighbors_list: list[list[int]]) -> list[str] : 
    """Classify atomic environments by neighbor count.

    Determine if each atom's environment is 'crystal' (12, 8, or 6 neighbors)
    or 'noncrystal'.

    Parameters
    ----------
    neighbors_list : list[list[int]]
        List of first nearest neighbor indices for each atom.

    Returns
    -------
    list[str]
        'crystal' or 'noncrystal' classification for each atom.

    """
    crystals = [12, 8, 6] 
    hash = [] 
    for neighbors in neighbors_list : 
        if len(neighbors) in crystals : 
            hash.append("crystal")
        else : 
            hash.append("noncrystal")
    return hash


