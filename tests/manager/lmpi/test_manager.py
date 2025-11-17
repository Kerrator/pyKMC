from pykmc.enginemanager.lmpi.engines import MpiApiEngine
from pykmc.enginemanager.lmpi.sessions import MpiApiSession
from pykmc.enginemanager.lmpi.pool import ManagerFactory, Manager
from pykmc.enginemanager.messenger import MpiMessenger
from pykmc import System, Config
from mpi4py import MPI 
import pytest
from pytest_lazy_fixtures import lf
import os
import time
import numpy as np


class TestManager: 

    def test_initialize_manager(self)  : 
        factory = ManagerFactory(n_sessions=2, use_rank_0=False, has_global=True)
        manager = factory.launch()

        if manager is None:
            return  # Engine processes stop here
        # ------------ SESSION CODE (rank 0) ------------
        manager.broadcast_command("units metal")
        manager.broadcast_command("log flush")
        manager.global_session.command("dimension 3")
        manager.global_session.command("log flush")
        manager.close_all()