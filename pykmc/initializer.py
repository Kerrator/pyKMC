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
        """Initialize the entire KMC object before starting the simulation.

        Raises
        ------
        RuntimeError
            If ``kmc.manager`` is ``None``: the manager must be injected
            (``KMC(config, manager=...)`` or ``kmc.manager = ...``) before the
            engines, the prefactor service and the reference table are built.

        """
        if self.kmc.manager is None:
            raise RuntimeError(
                "KMC.manager is None: pass the engine manager to "
                "KMC(config, manager=...) (or assign kmc.manager) before "
                "Initializer.initialize()"
            )
        self.initialize_loggers()
        self.initialize_system()
        self.initialize_engine()
        self.initialize_prefactor_service()
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
        from .physics import ResolvedConstraints

        system = self.kmc.system
        # Resolve in the full source ordering before any native mutation.
        self.kmc.global_constraints = ResolvedConstraints.resolve(
            system.positions,
            system.types,
            self.kmc.config.frozen_atoms,
            system.index,
            cell=system.cell,
            pbc=system.pbc,
        )
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
        if self.kmc.uses_event_prefactors:
            # Capability preflight of the HTST extension (contracts section
            # 6): fails fast when LAMMPS lacks PHONON or the potential cannot
            # be initialised in a scratch instance. Constant style: no call.
            self.kmc.manager.broadcast("htst_preflight")
            # ``broadcast`` returns nothing; one worker session's report (the
            # engine's authoritative species/mass map, contracts section 7d,
            # N3) comes back through the Future of the same operation. Every
            # session initialised the same potential, so the map is the same
            # whichever session answers.
            self.kmc.htst_preflight = self._preflight_report(
                self.kmc.manager.submit("htst_preflight").result()
            )
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

    @staticmethod
    def _preflight_report(report: object) -> dict:
        """Validate a worker session's ``htst_preflight`` report.

        Parameters
        ----------
        report : object
            The value the manager returned for ``htst_preflight``.

        Returns
        -------
        dict
            The report, with ``species`` and ``masses`` as tuples.

        Raises
        ------
        RuntimeError
            If the report is not a mapping carrying a non-empty ``species``
            tuple and one mass per species (``None`` is what a non-root rank
            of a session returns; the manager forwards the session root's
            value).

        """
        if not isinstance(report, dict):
            raise RuntimeError(
                "htst_preflight returned no report from the worker session "
                f"(got {type(report).__name__}); the engine species/mass map is "
                "required to build HTST requests"
            )
        species = tuple(str(s) for s in report.get("species", ()))
        masses = tuple(float(m) for m in report.get("masses", ()))
        if not species or len(species) != len(masses):
            raise RuntimeError(
                "htst_preflight report carries an inconsistent species/mass map: "
                f"species {species}, masses {masses}"
            )
        if "engine_physics" in report:
            from .physics import EnginePhysics

            physics = report["engine_physics"]
            if not isinstance(physics, EnginePhysics) or (
                physics.species,
                physics.masses,
            ) != (species, masses):
                raise RuntimeError("htst_preflight descriptor and type map disagree")
        return {**report, "species": species, "masses": masses}

    def initialize_prefactor_service(self) -> None:
        """Build the per-event prefactor service for the htst/rpa styles.

        The constant style leaves ``kmc.prefactor_service`` as ``None`` and
        never imports the HTST modules. Site-specific requests are built from
        the full pARTn-refined saddle (``ActiveEventTable.request_site_prefactors``),
        so ``free_radius`` is independent of the ``rcut`` crop radius. The
        service carries the engine's species/mass map from the preflight
        report kept by :meth:`initialize_engine`, so every live request
        describes the potential's masses (contracts section 7d, N3).

        Raises
        ------
        RuntimeError
            In the htst/rpa styles, if :meth:`initialize_engine` has not
            stored a preflight report (``kmc.htst_preflight`` is ``None``).

        """
        if not self.kmc.uses_event_prefactors:
            self.kmc.prefactor_service = None
            return
        from .rate_constant.prefactors import PrefactorService  # htst path only

        report = self.kmc.htst_preflight
        if report is None:
            raise RuntimeError(
                "initialize_prefactor_service needs the htst_preflight report "
                "(kmc.htst_preflight is None): run Initializer.initialize_engine "
                "first so the service carries the engine's species/mass map"
            )
        species_masses = (tuple(report["species"]), tuple(report["masses"]))
        self.kmc.prefactor_service = PrefactorService(
            self.kmc.config,
            self.kmc.manager,
            self.kmc.rate_constant,
            species_masses=species_masses,
            engine_physics=report.get("engine_physics"),
            global_constraints=getattr(self.kmc, "global_constraints", None),
        )
        settings = self.kmc.prefactor_service.settings
        mass_map = ", ".join(
            "{}={:g}".format(s, m) for s, m in zip(*species_masses, strict=True)
        )
        self.kmc.loggers.info(
            "log",
            ":=> HTST prefactor service ready (style {}, free_radius {} A centred "
            "on the {} geometry, fd_step {} A, nu0 window [{:.3e}, {:.3e}] Hz, "
            "k0 fallback {} ps^-1, engine masses (amu): {})".format(
                self.kmc.config.rateconstant.style,
                settings.free_radius,
                settings.free_region_center,
                settings.fd_step,
                settings.nu0_min_hz,
                settings.nu0_max_hz,
                self.kmc.config.rateconstant.k0,
                mass_map,
            ),
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
        self.kmc.reference_table = ReferenceEventTable(
            self.kmc.config, prefactor_service=self.kmc.prefactor_service
        )

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
