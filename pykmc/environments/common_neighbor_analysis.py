def cna(neighbors_list) : 
    crystals = [12, 8, 6] 
    hash = [] 
    for neighbors in neighbors_list : 
        if len(neighbors) in crystals : 
            hash.append('crystal')
        else : 
            hash.append('noncrystal')
    return cna


