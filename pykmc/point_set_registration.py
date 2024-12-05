import ira_mod 

class PointSetRegistration() : 
    """ 
    """ 
    def __init__(self,  system, psr_style, dimension, nprocs, backend) : 
        """ 
         
        """
        self.psr_style = psr_style
        self.system = system
        self.dimension = dimension
        self.backend = backend
        self.nprocs = nprocs 

    def run(self) : 
        """ 
        """ 
        if self.psr_style == 'ira' : 
            self.ira(0)
        else : 
            print(ERROR)
    
    def ira(self, idx_cat) : 
        """ 
        Use IRA to extract rotation, translation, permutation matrix to apply on generic event
        idx_cat : index in catalog of the selected event 
        """ 
        ira = ira_mod.IRA() 
        #first structure : 
        coords1 = self.system.get_positions() 
        nat1 = len(coords1)
        typ1 = self.system.get_chemical_symbols()
        #second structure 
        coords2 = self.system.catalog.loc[idx_cat].at["initial_positions"] 
        nat2 = len(coords2)
        typ2 = typ1 

        kmax_factor = 10.0
        rmat, tr, perm, dh = ira.match( nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor )
        print( "Hausdorff distance after matching:", dh )
        print( "Rotation matrix:" )
        print( rmat )
        print( "translation vector" )
        print( tr )
        print( "permutation of atoms:" )
        print( perm )
        print("")



