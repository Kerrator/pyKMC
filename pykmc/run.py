"""Entry point for running a Kinetic Monte Carlo (KMC) simulation.

This script parses command-line arguments to load the simulation configuration
from an input file, initializes the KMC simulation, and runs it.
"""

import argparse
import sys
import traceback

from mpi4py import MPI

from .kmc import KMC
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from .config import Config


def _launch(args: argparse.Namespace) -> None:
    """Load the config, launch the engine pool, and run the KMC simulation.

    On engine ranks ``ManagerFactory.launch`` blocks in the engine service loop and
    returns ``None``; only rank 0 gets a manager and drives the KMC run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (must carry ``input``).

    """
    # Config
    config = Config.from_ini_file(args.input)
    # KMC
    factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=config.control.engine_use_rank_0, has_global=True)
    manager = factory.launch()
    if manager is not None: #On rank 0
        kmc = KMC(config)
        # The manager must be attached BEFORE _initialize: the reference table
        # captures it at construction (htst/rpa nu0 batch fan-out).
        kmc.manager = manager
        kmc._initialize()
        kmc.run()


def main() -> None:
    """Parse input arguments and launch the KMC simulation.

    The function reads a configuration file specified by the user,
    creates a `KMC` instance, and runs the simulation.

    Under MPI (``world size > 1``) any exception escaping ``_launch`` on *any* rank --
    rank 0's KMC driver or an engine rank's service loop -- is a hard failure that
    would otherwise strand the other ranks (rank 0 blocking on a reply, engine ranks
    busy-spinning in ``run_engine_loop``) and hang the whole ``mpirun`` job for the full
    per-run timeout. We print the traceback and call ``MPI.COMM_WORLD.Abort(1)`` so the
    job dies in seconds with a visible cause. A normal finish exits via
    ``KMC._close`` -> ``sys.exit`` (``SystemExit``, not ``Exception``), which is
    deliberately *not* caught, so a clean shutdown never triggers ``Abort``. On a single
    rank the exception is re-raised unchanged, preserving normal tracebacks/exit codes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input", type=str, required=True, help="inputs file")
    args = parser.parse_args()

    if MPI.COMM_WORLD.Get_size() == 1:
        _launch(args)
        return

    try:
        _launch(args)
    except Exception:  # noqa: BLE001 - deliberate MPI-wide failure boundary
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    main()
