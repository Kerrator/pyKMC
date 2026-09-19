"""Entry point for running a Kinetic Monte Carlo (KMC) simulation.

This script parses command-line arguments to load the simulation configuration
from an input file, initializes the KMC simulation, and runs it.
"""

import argparse
from mpi4py import MPI
from .kmc import KMC
from pykmc.factory import EngineManagerFactory
from .config import Config
from .rate_constant import create_rate_constant


def main() -> None:
    """Parse input arguments and launch the KMC simulation.

    The function reads a configuration file specified by the user,
    creates a `KMC` instance, and runs the simulation.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input", type=str, required=True, help="inputs file")
    args = parser.parse_args()

    # Config
    config = Config.from_ini_file(args.input)
    comm = MPI.COMM_WORLD
    group_size = (
        (comm.Get_size() - 1)
        if config.control.group_size == -1
        else config.control.group_size
    )
    # The HTST engine extension is registered only for the htst/rpa styles
    # (contracts section 6); its module is LAMMPS/HTST bound and is never
    # imported on the constant path.
    engine_extensions = None
    if create_rate_constant(config.rateconstant).backend.requires_event_prefactors:
        from pykmc.engine.htst_lammps import LammpsHTSTExtension

        engine_extensions = [LammpsHTSTExtension]
    # KMC
    factory = EngineManagerFactory(
        engine_style=config.control.engine,
        n_workers=config.control.n_sessions,
        comm=comm,
        engine_config=config.lammps,
        group_size=group_size,
        engine_extensions=engine_extensions,
    )
    manager = factory.launch()
    if manager is not None:  # On rank 0
        kmc = KMC(config, manager=manager)
        try:
            kmc._initialize()
            kmc.run()
        except SystemExit:
            # KMC._close chose the status (0 normal completion, 1 aborted
            # simulation) after shutting the workers down; the idempotent
            # shutdown here covers an exit raised anywhere else. The status
            # propagates to ``python -m pykmc`` instead of being swallowed.
            # A shutdown failing here leaves workers un-notified, so it goes
            # through the same abort boundary as any other failure.
            try:
                manager.shutdown()
            except BaseException:
                comm.Abort(1)
                raise
            raise
        except BaseException:
            comm.Abort(1)
            raise


if __name__ == "__main__":
    main()
