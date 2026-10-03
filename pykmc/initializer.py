"""KMC Simulation Initialization Module.

This module contains the `Initializer` class, which takes a reference to a `KMC` object
and sets up its attributes necessary for running the simulation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kmc import KMC
from .log import LogKMC, LOGGING_CONFIG
from .system import System
from .neighbors_list import NeighborsList
from .atomic_environment import AtomicEnvironment
from .event_table import ReferenceEventTable
from .bias import DirectionBias, PointBias, TopoBias
import pickle


class Initializer:
    """Initializer for the KMC class.

    Parameters
    ----------
    kmc : KMC
        KMC object initialized based on its configuration.

    """

    def __init__(self, kmc: "KMC") -> None:
        self.kmc = kmc

    def initialize(self) -> None:
        """Initialize the entire KMC object before starting the simulation."""
        self.initialize_loggers()
        self.initialize_system()
        self.initialize_engine()
        self.initialize_htst()
        self.initialize_neighbors_list()
        self.initialize_atomic_environments()
        self.initialize_reference_table()
        self._initialize_visited_environments()
        self.initialize_bias()

        self.kmc.loggers.new_line("log")
        self.kmc.loggers.info("log", "===========================")
        self.kmc.loggers.info("log", "= Starting KMC simulation =")
        self.kmc.loggers.info("log", "===========================")

        if self.kmc.config.control.restart_file is None:
            self.kmc.loggers.table_line_info_kmc(
                "output", 0, 0.0, 0.0, None, None, None, None, self.kmc.total_energy
            )
        else:
            self.kmc.loggers.info("log", ":=> Restarting")

    def initialize_loggers(self) -> None:
        """Initialize the loggers and create their files."""
        self.kmc.loggers = LogKMC(
            LOGGING_CONFIG, verbosity=self.kmc.config.control.verbosity
        )
        self.kmc.loggers.title("log")
        self.kmc.loggers.write_parameters("log", self.kmc.config)
        if self.kmc.config.control.restart_file is None:
            self.kmc.loggers.output_file_header("output")
            self.kmc.loggers.events_file_header("events")

    def initialize_system(self) -> None:
        """Read and initialize the system from the intial configuration file."""
        self.kmc.loggers.info(
            "log",
            ":=> Reading initial configuration file : {}".format(
                self.kmc.config.control.initial_config
            ),
        )
        self.kmc.system = System.create_from_file(
            self.kmc.config.control.initial_config
        )

    def initialize_engine(self) -> None:
        """Start and initialize the engine workers (local and group)."""
        system = self.kmc.system
        self.kmc.manager.broadcast("start")
        self.kmc.manager.broadcast("initialize_parameters")
        self.kmc.manager.broadcast(
            "initialize_system",
            types=system.types,
            positions=system.positions,
            cell=system.cell,
            pbc=system.pbc,
        )
        self.kmc.manager.broadcast("initialize_potential")
        self.kmc.manager.submit_group("start")
        self.kmc.manager.submit_group("initialize_parameters")
        self.kmc.manager.submit_group(
            "initialize_system",
            types=system.types,
            positions=system.positions,
            cell=system.cell,
            pbc=system.pbc,
        )
        self.kmc.manager.submit_group("initialize_potential")

    def initialize_htst(self) -> None:
        """Check that HTST prefactors can be computed and set the Hessian region.

        Does nothing unless `[RateConstant] style = htst`. A run that could never
        compute a prefactor stops here, with the reason in the log.
        """
        config = self.kmc.config
        if config.rateconstant.style != "htst":
            return
        from pykmc.htst import (
            AV_BUFFER_WARN_A,
            RMOV_WARN_A,
            full_hessian_mb,
            hessian_mode,
            region_record,
        )

        rc, system, loggers = config.rateconstant, self.kmc.system, self.kmc.loggers
        mode = hessian_mode(config)
        species = sorted(set(system.types))
        n_atoms = len(system.types)
        n_sessions = config.control.n_sessions
        shell = None
        try:
            if not system.cell.orthorhombic:
                raise RuntimeError(
                    "HTST: style = htst needs an orthogonal cell; this cell has "
                    "off-diagonal terms. Use an orthogonal cell or set [RateConstant] "
                    "style = constant."
                )
            if mode == "full" and sum(bool(p) for p in system.pbc) < 2:
                pbc = " ".join("T" if p else "F" for p in system.pbc)
                raise RuntimeError(
                    "HTST: free_radius = full needs at least two periodic axes "
                    f"(pbc = {pbc}): with fewer, rotations are zero modes that are "
                    "not removed. Use a numeric free_radius or set [RateConstant] "
                    "style = constant."
                )
            if mode == "full" and full_hessian_mb(n_atoms) > rc.max_full_hessian_mb:
                raise RuntimeError(
                    f"HTST: free_radius = full needs {full_hessian_mb(n_atoms):.0f} MB "
                    f"per Hessian for {n_atoms} atoms (one 3N x 3N matrix of doubles), "
                    f"above max_full_hessian_mb = {rc.max_full_hessian_mb:g}. Up to "
                    f"n_sessions = {n_sessions} Hessians are built at once, each "
                    "needing about twice that in memory and a scratch file of that "
                    "size in hessian_dir. Raise max_full_hessian_mb or use a numeric "
                    "free_radius."
                )
            # Every session checks its own LAMMPS and its access to hessian_dir.
            self.kmc.manager.broadcast(
                "htst_capability_check", config=config, species=species
            )
            info = self.kmc.manager.htst_capability_check(
                config=config, species=species
            ).result()
            if mode == "reduced":
                if rc.shell_width is not None:
                    shell = rc.shell_width
                elif info["force_cutoff"] is None or info["force_cutoff"] <= 0:
                    raise RuntimeError(
                        "HTST: the force cutoff could not be read from LAMMPS "
                        "('Communication cutoff' missing or not positive in info "
                        "communication); set [RateConstant] shell_width."
                    )
                else:
                    shell = 2.0 * info["force_cutoff"]
        except Exception as exc:
            # run.py aborts before the exception is printed: log the reason.
            loggers.error("log", str(exc))
            raise
        self.kmc.htst_shell_width = shell
        self.kmc.htst_region = region_record(config, shell)

        loggers.info(
            "log",
            f"HTST: LAMMPS version {info['lammps_version']}; Hessian scratch files "
            f"go to {info['hessian_dir']}; fd_step = {rc.fd_step:g} A; region record "
            f"'{self.kmc.htst_region}'.",
        )
        if mode == "av":
            rmov, ract = config.activevolume.rmov, config.activevolume.ract
            loggers.info(
                "log",
                "HTST: Hessian region: active volume; atoms within rmov = "
                f"{rmov:g} A of the search or refinement centre are displaced, the "
                "rmov-ract buffer is kept frozen, atoms beyond ract = "
                f"{ract:g} A are left out; free_radius, shell_width and "
                "max_full_hessian_mb are ignored.",
            )
            if rmov < RMOV_WARN_A:
                loggers.warning(
                    "log",
                    f"HTST: rmov = {rmov:g} A is below 8 A: with active_volume = True "
                    "the Hessian region is the active volume, and nu0 is not "
                    "converged at this radius (Ni vacancy hop, against the all-atom "
                    "value: about 50 % low at rmov = 4 A, about 2 % at rmov = 8 A). "
                    "See the HTST page of the documentation.",
                )
            if ract - rmov < AV_BUFFER_WARN_A:
                loggers.warning(
                    "log",
                    "HTST: the active-volume buffer ract - rmov = "
                    f"{ract - rmov:g} A is thinner than 4 A: the frozen buffer may "
                    "not shield the displaced atoms (Ni vacancy hop: crop error up "
                    "to 2 % at 2-3 A, up to 39 % at 1 A).",
                )
        elif mode == "reduced":
            source = (
                "[RateConstant] shell_width"
                if rc.shell_width is not None
                else f"2 x force cutoff {info['force_cutoff']:g} A read from LAMMPS"
            )
            loggers.info(
                "log",
                "HTST: Hessian region: atoms within free_radius = "
                f"{rc.free_radius:g} A of the moving atom's saddle position are "
                f"displaced, a frozen shell of {shell:g} A ({source}) is kept, atoms "
                "beyond are left out.",
            )
        else:
            loggers.info(
                "log",
                f"HTST: Hessian region: the whole system ({n_atoms} atoms, "
                f"{full_hessian_mb(n_atoms):.0f} MB per Hessian; up to n_sessions = "
                f"{n_sessions} Hessians at once, each needing about twice that in "
                "memory and a scratch file of that size in hessian_dir).",
            )
        if config.control.basin:
            loggers.warning(
                "log",
                "HTST: basin = True: inside a basin every rate (internal hops and "
                f"exits) uses the constant prefactor k0 = {rc.k0:g} ps^-1; steps "
                "outside basins use nu0.",
            )
        if config.atomicenvironment.atom_coloring_mode == "grey" and len(species) > 1:
            loggers.warning(
                "log",
                f"HTST: atom_coloring_mode = grey with {len(species)} species: a "
                "reference nu0 is computed with the species of the event it was "
                "found on and is reused by unrefined events on other species; "
                "refined events get their own nu0.",
            )

    def initialize_neighbors_list(self) -> None:
        """Construct a new Neighbors List."""
        self.kmc.loggers.info("log", ":=> Constructing Neighbors Lists")
        self.kmc.neighbors_list = NeighborsList(
            self.kmc.system,
            self.kmc.config.atomicenvironment.rnei,
            self.kmc.config.atomicenvironment.rcut,
        )

    def initialize_atomic_environments(self) -> None:
        """Construct a new Atomic Environment."""
        self.kmc.loggers.info("log", ":=> Computing Atomic Environments")
        self.kmc.atomic_environment = AtomicEnvironment(
            self.kmc.config.atomicenvironment.style,
            self.kmc.neighbors_list.neighbors_list["rnei"],
            self.kmc.neighbors_list.neighbors_list["rcut"],
            self.kmc.config.atomicenvironment.neighbors_add,
            coordination_threshold=self.kmc.config.atomicenvironment.coordination_threshold,
            types=self.kmc.system.types,
            coloring_mode=self.kmc.config.atomicenvironment.atom_coloring_mode,
        )

    def initialize_reference_table(self) -> None:
        """Initialize the Reference Event Table."""
        if self.kmc.config.control.reference_table is not None:
            self.kmc.loggers.info(
                "log",
                ":=> Reading Reference table file {}".format(
                    self.kmc.config.control.reference_table
                ),
            )
        else:
            self.kmc.loggers.info("log", ":=> Generate a empty reference table")
        self.kmc.reference_table = ReferenceEventTable(self.kmc.config)

    def initialize_bias(self) -> None:
        """Instantiate the bias object from the config, or set it to None."""
        bc = self.kmc.config.bias
        if bc is None or not self.kmc.config.control.bias:
            self.kmc.bias = None
            return
        match bc.style:
            case "direction":
                self.kmc.bias = DirectionBias(
                    bc.direction,
                    bc.atom_indices,
                    bc.threshold,
                    mode=bc.mode,
                    bias_weight=bc.bias_weight,
                    pass_unlisted=bc.pass_unlisted,
                )
            case "point":
                self.kmc.bias = PointBias(
                    bc.target_point,
                    bc.atom_indices,
                    bc.threshold,
                    mode=bc.mode,
                    bias_weight=bc.bias_weight,
                    pass_unlisted=bc.pass_unlisted,
                )
            case "topo":
                self.kmc.bias = TopoBias(
                    bc.topo_source,
                    bc.topo_target,
                    mode=bc.mode,
                    bias_weight=bc.bias_weight,
                    pass_unlisted=bc.pass_unlisted,
                )

    def _initialize_visited_environments(self) -> None:
        """Initialize visited environment from file if specified, else initialize as {'crystal'}."""
        if self.kmc.config.control.visited_environments is not None:
            self.kmc.loggers.info(
                "log",
                ":=> Initiating visited environment from file {}".format(
                    self.kmc.config.control.visited_environments
                ),
            )
            try:
                with open(self.kmc.config.control.visited_environments, "rb") as file:
                    loaded_set_environments = pickle.load(file)
                self.kmc.visited_environments = loaded_set_environments
            except Exception as e:
                raise Exception("Can't read visited environment file.") from e
        else:
            self.kmc.visited_environments = set(["crystal"])
        if (
            self.kmc.config.control.visited_environments
            and not self.kmc.config.control.reference_table
        ):
            self.kmc.loggers.warning(
                "log",
                "Visited environments are read from file while no reference table was provided",
            )
