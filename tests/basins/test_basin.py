from pykmc.basins import  BasinsGenericEvents
import logging
import numpy as np
from pykmc.enginemanager.lmpi.pool import ManagerFactory

logger = logging.getLogger("tests")


class TestFingerprint:

    def test_fingerprint_permutation_invariance(self):
        """Fingerprint should be invariant to atom permutation."""
        positions = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = BasinsGenericEvents._compute_fingerprint(positions, cell, pbc)
        fp2 = BasinsGenericEvents._compute_fingerprint(positions[[2,0,3,1]], cell, pbc)
        assert np.allclose(fp1, fp2)

    def test_fingerprint_translation_invariance(self):
        """Fingerprint should be invariant to uniform translation (no boundary crossing)."""
        positions = np.array([[1,1,1],[2,1,1],[1,2,1],[1,1,2]], dtype=float)
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = BasinsGenericEvents._compute_fingerprint(positions, cell, pbc)
        # Translate by 3.0 in each direction — no atoms cross boundary
        fp2 = BasinsGenericEvents._compute_fingerprint(positions + [3.0, 3.0, 3.0], cell, pbc)
        assert np.allclose(fp1, fp2)

    def test_fingerprint_different_structures(self):
        """Different structures should produce different fingerprints."""
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        pos1 = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        pos2 = np.array([[0,0,0],[3,0,0],[0,3,0],[0,0,3]], dtype=float)
        fp1 = BasinsGenericEvents._compute_fingerprint(pos1, cell, pbc)
        fp2 = BasinsGenericEvents._compute_fingerprint(pos2, cell, pbc)
        assert not np.allclose(fp1, fp2, atol=0.3)


class TestBasin :

    def test_connectivity_table_construction(self, test_logger, config_Cu, reference_table_Cu_fake, system_Cu, visited_environments_Cu) :
        
        #Create Manager
        factory = ManagerFactory(n_sessions=config_Cu.control.n_sessions, use_rank_0=True)
        manager = factory.launch()

        if manager is not None: #On rank 0
            manager.initialize_sessions(config_Cu, system_Cu)

            self.basin = BasinsGenericEvents(config=config_Cu, reference_table=reference_table_Cu_fake, known_environments=visited_environments_Cu, manager = None)
            self.basin.manager = manager

            result = self.basin.execute(system=system_Cu)
            if result.is_ok() : 
                test_logger.debug("Find Exit State : ")
                test_logger.debug("Exit time t_exit = {}ps".format(result.ok_value().t_exit))
                test_logger.debug("Exit state n : {}".format(result.ok_value().exit_state))
            else : 
                test_logger.debug("Error: {}".format(result.err_value()))
            
            manager.close_all()
