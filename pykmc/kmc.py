"""Module for executing off-lattice kinetic Monte Carlo (KMC) simulations.

This module defines the `KMC` class.
"""

from pykmc import NeighborsList, AtomicEnvironment, ActiveEventTable, Config, Reconstruction
import random
from .result import (
    EventSearchOutput,
    KMCLoopInfo,
    ErrorInfo,
    ErrorType,
    Result,
    AtomicEnvironmentInfo,
    ReferenceEventSearchInfo,
    ReferenceValidEventsInfo,
    RefinementsInfo,
    EventRefinementOutput,
    ReconstructionOutput,
    Err,
    Ok
)
import numpy as np
import os
from ase.io import write
from ase import Atoms
from .algorithms import rejection_free
import sys
import pandas as pd
import pickle
from .initializer import Initializer
from .info_simulation import (
    info_atomic_environments,
    info_reference_event_searches,
    info_is_valid_reference_events,
    info_refinements,
    info_active_events,
    info_basin_events
)
from .eventsearch import EventSearch
from .refinement import Refinement
from .log import Colors
import time
from .utils import push_towards, compute_delr
import copy
from .basins.detection import DetectorThreshold
from .basins import BasinsGenericEvents
from .event_recycling import DistanceRecycling, Recycling
from .bias import Bias
from .dissolution import DissolutionSelection, dissolution_rates, eligible_atoms


# Reconstruction failures that are specific to *this active row* (a per-site IRA
# mapping mismatch or a transient engine hiccup), not evidence that the generic
# reference event is defective. These drop the active row only; they must NOT
# purge the globally-valid reference event (which would silently delete a good
# event -- see findings #2/#9). A wrong-minimum outcome (INVALID_MIN1/MIN2), by
# contrast, means the event does not reconnect and the reference is purged.
_PER_ROW_RECONSTRUCTION_ERRORS = frozenset(
    {
        ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
        ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED,
    }
)


# NOTE can maybe reimplment tries if empty catalog
#TODO: Add reconstruction info

class KMC:
    """Manage and execute the Kinetic Monte Carlo (KMC) simulation.

    This class acts as the central controller coordinating all phases of the simulation:
    initialization, event search, event refinement, event selection, system updates,
    minimization, logging, and termination.

    Attributes
    ----------
    config : Config
        The parameters of the simulation.
    loggers : LogKMC
        Handle logging of simulation progress.
    system : System
        The atomic system.
    engine : Engine
        The E/F engine used.
    neighbors_list : NeighborsList
        Store neighbors of atoms in the system.
    atomic_environment : AtomicEnvironment
        Store atomic environment of atoms in the system
    reference_table : ReferenceEventTable
        Store generic events that can be apply to the system.
    visited_environment : set[str]
        Track atomic environments already explored. Those for which event searches as been previously done.
    total_energy : float
        The total energy of the system.

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.loggers = None
        self.system = None
        self.manager = None
        self.engine = None
        self.neighbors_list = None
        self.atomic_environment = None
        self.reference_table = None
        self.visited_environments = None
        self.total_energy = None
        self.potential_energy = None
        self.active_table: ActiveEventTable | None = None
        self._pre_exec_positions: np.ndarray | None = None
        self.bias: Bias | None = None
        # Per-step dissolution candidates: (indices, rates, coordinations) of the
        # atoms eligible to dissolve this step, or None when the feature is off.
        # Recomputed each step; the rates compete in the same BKL vector.
        self._dissolution_candidates: (
            tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ) = None
        # The recycler decides what to carry over between KMC steps at
        # end-of-step; with no recycler the active table is cleared each
        # step (same observable behavior as prior releases).
        self.recycler: Recycling | None = None
        if self.config.control.recycle:
            if self.config.eventrecycling.style == "displacement":
                self.recycler = DistanceRecycling(
                    movement_thr=self.config.eventrecycling.movement_thr,
                    distance_thr=self.config.eventrecycling.distance_thr,
                )

    def run(self) -> None:
        """Run the simulation, guaranteeing the engine pool is torn down on failure.

        If anything in the body raises before the normal shutdown, the engine ranks
        would otherwise stay busy-spinning in ``run_engine_loop`` waiting for a
        command that never comes -- the production face of the recycle pool-hang (see
        HANDOFF_recycle_pool_hang.md). Closing the pool here unstrands them; the
        original error is re-raised so the failure stays visible. A normal finish goes
        through ``_close()`` -> ``sys.exit()`` (``SystemExit``), which is deliberately
        not caught here, so the pool is closed exactly once.
        """
        try:
            self._run_impl()
        except Exception:
            self.manager.close_all()
            raise

    def _run_impl(self) -> None:
        """Run the simulation."""
        # Initialize the simulation, KMC attributes and minimize the system
        #self._initialize()
        self.manager.initialize_sessions(self.config, self.system)
        self.minimize_system()
        self.neighbors_list = NeighborsList(
                self.system,
                self.config.atomicenvironment.rnei,
                self.config.atomicenvironment.rcut,
            )
        self.atomic_environment = AtomicEnvironment(
                self.config.atomicenvironment.style,
                self.neighbors_list.neighbors_list["rnei"],
                self.neighbors_list.neighbors_list["rcut"],
                self.config.atomicenvironment.neighbors_add,
                types=self.system.types if self.config.atomicenvironment.atom_coloring_mode == "full" else None,
                coordination_threshold=self.config.atomicenvironment.coordination_threshold,
            )
        self.inactive_ae = (
            AtomicEnvironment(
                style="region",
                region=self.config.inactive_atoms,
                positions=self.system.positions,
                atom_types=self.system.types,
            ) if self.config.inactive_atoms is not None else None
        )
        self.frozen_ae = (
            AtomicEnvironment(
                style="region",
                region=self.config.frozen_atoms,
                positions=self.system.positions,
                atom_types=self.system.types,
            ) if self.config.frozen_atoms is not None else None
        )
        #Set new positions to all sessions/engine :
        self.manager.use_local()
        self.manager.set_all_positions(self.system.positions)

        if self.config.control.restart_file is None:
        # Write initial step to file
            self._append_snapshot_to_trajectory()
            last_step = 0
            total_time = 0.0

        else : #read restart file
            self.loggers.info("log", ":=> Reading restart file")
            restart_info = np.load(self.config.control.restart_file)
            last_step = restart_info["last_step"]
            total_time = restart_info["last_time"]
            self.loggers.info("log", ":=> last step = {}, last_end_time = {}ps".format(last_step, total_time))

        # LOOP KMC PARAMETERS
        nkmc_steps = self.config.control.n_steps
        last_step +=1
        nsearch = self.config.eventsearch.nsearch

        # Build the persistent active event table once, with the recycler
        # plugin (built in __init__) and the engine manager (HTST prefactors)
        # attached.
        self.active_table = ActiveEventTable(
            self.config, recycler=self.recycler, manager=self.manager
        )

        # Warn loudly if a configured dissolvable element is absent from the
        # initial system (a likely misconfiguration -- nothing would ever dissolve).
        if self.config.control.dissolution:
            present = set(np.asarray(self.system.types).tolist())
            for el in self.config.dissolution.elements:
                if el not in present:
                    self.loggers.warning(
                        "log",
                        "Dissolution element '{}' is not present in the initial "
                        "system; no atom of it can ever dissolve.".format(el),
                    )

        # KMC LOOP
        for step in range(last_step, nkmc_steps+last_step):
            start_real = time.time()
            start_cpu = time.process_time()

            self.loggers.info(
                "log",
                "{}{}Step : {}{}".format(
                    Colors.BOLD.value, Colors.YELLOW.value, step, Colors.RESET.value
                ),
            )

            # == Find Current atomic environments that has not been visited ==
            new_environments = self.get_new_environments()

            # == Scan for dealloying dissolution candidates (competes in BKL) ==
            self._compute_dissolution_candidates()

            if self.config.control.recycle and len(self.active_table.table) > 0:
                self.loggers.info(
                    "log",
                    "\t :=> Recycling {} events from the previous step".format(
                        len(self.active_table.table)
                    ),
                )

            # == FIND NEW GENERIC EVENTS ==
            ##=>List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(
                new_environments, nsearch
            )

            ##=>Perform event search on each atom in central_atom_research_list
            event_search = self.execute_event_searches(central_atom_research_list)

            # == ADD NEW GENERIC EVENTS TO REFERENCE EVENT TABLE ==
            ##=>Check if the event is valid, ie if not already present and has a valid energy barrier if yes add it to the reference table
            search_results = event_search.get_successes_results()
            if self.inactive_ae is not None:
                inactive_set = set(self.inactive_ae.get_atoms_with_id("in"))
                search_results = [r for r in search_results if r.move_atom_index not in inactive_set]
            results_is_valid_events = self.add_reference_events(search_results)

            ##=>Close simulation if no events in the reference table AND no
            ##  dissolution candidate can carry the step (dissolution needs no
            ##  reference event -- the atom just leaves).
            if len(self.reference_table.table) == 0 and self._n_dissolution() == 0:
                self.loggers.error(
                    "log",
                    "No events have been found, empty reference events table. \n \tTry to increase nsearch or saddle point search algorithm's parameters. \n \tClosing the simulation.",
                )
                self._close()


            # == Update variables ==
            l_ids = list(set(self.atomic_environment.atomic_environment_list))
            self.visited_environments.update(
                set(l_ids).difference(self.visited_environments)
            )
            # == Refinement ==
            ##=>Subset of reference_event_table with generic event that can be apply to the current step (ie event_id in atomic environment)
            subset_reference_event_table = self.reference_table.has_id_subset_table(
                self.atomic_environment.atomic_environment_list
            )
            ##=>Refines all event in subset (skipping (atom, ref_event) pairs already carried over)
            refinement = self.execute_refinements(
                subset_reference_event_table,
                existing_pairs=self.active_table.existing_pairs(),
            )

            # == ADD ACTIVE EVENT TO ACTIVE EVENT TABLE ==
            # The persistent self.active_table is extended in place; recycled
            # rows from the previous step are already present.
            self.add_active_events(refinement.get_successes_results())
            active_table = self.active_table

            active_table.remove_duplicates(self.system.cell, self.neighbors_list)  #To be sure
            self.loggers.info("log", "\t :=> {} active events after removing duplicates.".format(len(active_table.table)))

            # == Site-specific prefactors for refined events (htst/rpa) ==
            # After dedup so duplicates never cost a Hessian; before use_global so
            # the batch fans out over the local session pool.
            active_table.backfill_refined_prefactors(self.system, self.neighbors_list)

            # == Update System ==
            self.manager.use_global()
            result_reconstruction, delta_t, ktot, idx_selected_event, err_reference, err_ae = self.reconstruction(active_table)
            # A dissolution row won the BKL draw: no saddle/reconstruction, the
            # atom just leaves. Detected here so every downstream consumer of the
            # (now out-of-range) idx_selected_event -- the events file, the basin
            # detector, the step log, the prune -- branches on it.
            is_dissolution = isinstance(result_reconstruction, DissolutionSelection)
            events_info = info_active_events(self.system.types, self.reference_table, active_table)
            # Refs purged this step; the orphan eviction is DEFERRED until after the
            # prune block below (see the drop_orphans call there for the rationale).
            removed_refs: set[int] = set()
            if len(err_reference) != 0 :
                self.loggers.info("log", "\t :=> Removing reference event from which reconstruction failed.")
                removed_refs = self.reference_table.remove(list(set(err_reference)))
                self.loggers.info("log", "\t :=> Removing topology from known environments from which reconstruction failed.")
                self.visited_environments = self.visited_environments.difference(set(err_ae))
            events_info = events_info.output_msg()




            #INFO :
            self.loggers.events_file_step_first_line("events", step)
            if is_dissolution :
                # idx_selected_event is a concatenated (past-the-active-rows)
                # index that has no row in the listing below, so write the -1
                # sentinel and a parseable dissolution record instead (consumed by
                # the PyKMC_Analysis events parser).
                self.loggers.events_applicable_info_line("events", -1)
                diss_pos = self.system.positions[result_reconstruction.atom_index]
                self.loggers.info(
                    "events",
                    "#Dissolution: element={} position=[{:.4f}, {:.4f}, {:.4f}] "
                    "coordination={} k={:.6e}".format(
                        self.system.types[result_reconstruction.atom_index],
                        diss_pos[0], diss_pos[1], diss_pos[2],
                        result_reconstruction.coordination,
                        result_reconstruction.rate,
                    ),
                )
            else :
                self.loggers.events_applicable_info_line("events", idx_selected_event)
            self.loggers.info("events", events_info)

                #TODO: Temporary, need to unified kmc main loop and basin operations + ugly
            detector = DetectorThreshold()
            # Pre-execution snapshot for event recycling (needed before update_positions below)
            if self.config.control.recycle:
                self._pre_exec_positions = self.system.positions.copy()
                #IF selected event shows we are in a basin
            if is_dissolution :
                self._execute_dissolution(result_reconstruction, total_time)
                prune_detach_recycler = False
            elif self.config.control.basin and detector.detect(active_table.table.iloc[idx_selected_event], self.reference_table.table, self.config.basin.energy_thr, True) :
                self.loggers.info("log","\t :=> System is in a Basin." )
                self.loggers.info("log","\t :=> Exploring the Basin." )
                #get basin info/explore
                basin = BasinsGenericEvents(self.config, self.reference_table, self.visited_environments, self.manager)
                self.system.update_positions(result_reconstruction.ok_value().min1_positions)
                result_basin = basin.execute(self.system)
                if result_basin.is_ok() : #Basin did no fail
                #move system to a state connected to the exit_state
                    self.system.update_positions(result_basin.ok_value().initial_system_positions)
                    self.neighbors_list = basin.states[result_basin.ok_value().from_state].neighbors_list
                #construct new active table with only event : new_actual_state - > exit_state
                    tmp_active_table = ActiveEventTable(self.config)
                    tmp_event = EventRefinementOutput(central_atom_index=result_basin.ok_value().central_atom,
                                                      saddle_positions=result_basin.ok_value().saddle_positions,
                                                      E_saddle=-1,
                                                      min2_positions=result_basin.ok_value().final_positions,
                                                      dE_forward=result_basin.ok_value().energy_barrier,
                                                      num_reference_event=result_basin.ok_value().num_reference_event,
                                                      neighbors=result_basin.ok_value().neighbors)
                    tmp_active_table.add_events(tmp_event)
                #reconstruct event
                    self.manager.use_global()
                    result_basin_reconstruction = self._reconstruction_active_event(0, tmp_active_table)
                    if result_basin_reconstruction.is_ok() :
                        self.system.update_positions(result_basin_reconstruction.ok_value().min2_positions)
                        self.total_energy = result_basin_reconstruction.ok_value().min2_etot
                        delta_t = result_basin.ok_value().t_exit
                        ktot = result_basin.ok_value().k_tot
                        idx_selected_event = 0
                        active_table.table = tmp_active_table.table

                        #INFO
                        idx_exit_event, basin_info = info_basin_events(self.system.types, self.reference_table, basin.connectivity_table, result_basin.ok_value().exit_state)
                        basin_info = basin_info.output_msg()
                        self.loggers.events_basin_info_line("events",idx_exit_event )
                        self.loggers.info("events", basin_info)


                    else :
                       self.loggers.info("log", "\t :=> Reconstruction Exit State Basin fails with error {}, back to original event".format(result_basin_reconstruction.err_value()))
                       self.system.update_positions(basin.states[0].system.positions)
                       self._apply_original_migration_event(result_reconstruction)
                else :
                    self.loggers.info("log", "\t :=> Basin fails with error : {}, back to original event".format(result_basin.err_value()))
                    self._apply_original_migration_event(result_reconstruction)
                if basin.connectivity_table is not None :
                    basin.connectivity_table.save('basin_connectivity_'+str(step)+'.pickle')
                #update delta_t, ktot (use basin infos)
                # Basin super-event spans many atoms; recycling is deferred (the
                # prune below runs with the recycler detached).
                prune_detach_recycler = True
            else :
                self._apply_original_migration_event(result_reconstruction)
                prune_detach_recycler = False
            total_time += delta_t * 10**-12  # time is in seconds

            ###=> Synchronise all lammps instances with new positions
            self.manager.use_local()
            self.manager.set_all_positions(positions=self.system.positions)
            ##=>Minimize

            # == Log informations ==
            atomic_environment_info = self.get_info_atomic_environments(
                new_environments
            )
            reference_event_searches_info = self.get_info_reference_event_searches(
                event_search.results
            )
            is_valid_events_info = self.get_info_is_valid_reference_events(
                results_is_valid_events
            )
            refinements_info = self.get_info_refinements(refinement.results)
            kmc_loop_info = KMCLoopInfo(
                step=step,
                atomic_environment_info=atomic_environment_info,
                reference_event_searches_info=reference_event_searches_info,
                valid_event_info=is_valid_events_info,
                refinements_info=refinements_info,
            )
            self.loggers.info("info", kmc_loop_info.output_msg())


            elapsed_real = time.time() - start_real
            elapsed_cpu = time.process_time() - start_cpu

            # Dissolution steps have no reference event / active row to read; log
            # honest surrogate columns (ref = -1, Ea = n*E_b, k = the bond-counting
            # rate) so the output table stays parseable.
            if is_dissolution :
                ref_event_col = -1
                ea_col = result_reconstruction.coordination * self.config.dissolution.E_b
                k_evt_col = result_reconstruction.rate
            else :
                ref_event_col = active_table.table.loc[idx_selected_event].at["num_reference_event"]
                ea_col = active_table.table.loc[idx_selected_event].at["energy_barrier"]
                k_evt_col = active_table.table.loc[idx_selected_event].at["k"]
            self.loggers.table_line_info_kmc(
                "output",
                step,
                delta_t * 10**-12,
                total_time,
                ref_event_col,
                ea_col,
                k_evt_col,
                ktot,
                self.total_energy,
                elapsed_cpu,
                elapsed_real
            )

            # == Event recycling: prune the active table for the next step ==
            # Must run AFTER the step log above, which reads the executed event's
            # row; with no recycler (recycle = False, the default) the prune clears
            # the whole table and the lookup would raise KeyError.
            if is_dissolution:
                # The deletion shifted every atom index and dropped an atom, so no
                # active row (its atom_index / neighbours) can be trusted or
                # recycled. Flush the table; the next step rebuilds from scratch.
                self.active_table.table = self.active_table.table.iloc[0:0].reset_index(drop=True)
            elif prune_detach_recycler:
                saved_recycler = self.active_table.recycler
                self.active_table.recycler = None
                self.active_table.prune_for_recycling(
                    idx_selected_event, self.system, self._pre_exec_positions,
                )
                self.active_table.recycler = saved_recycler
            else:
                self.active_table.prune_for_recycling(
                    idx_selected_event, self.system, self._pre_exec_positions,
                )
                if self.config.control.recycle:
                    self.loggers.info(
                        "log",
                        "\t :=> {} events flagged for recycling".format(
                            len(self.active_table.table)
                        ),
                    )

            # == Evict active rows orphaned by this step's reference purge ==
            # Deferred to here (not the purge block above) on purpose: the purge
            # runs before every positional consumer of the POSITIONAL
            # idx_selected_event -- the step log (num_reference_event/energy_barrier/k),
            # detector.detect, and prune_for_recycling's recycling anchor. Evicting
            # + reset_index there would shift labels under idx_selected_event, and
            # if the executed row shares its num_reference_event (or backward
            # sibling) with an earlier row that failed with INVALID_MIN1/MIN2 in the
            # same reconstruction() loop -- routine, since one generic ref maps onto
            # many sites -- the executed row itself would be evicted, turning the
            # .loc[idx_selected_event] lookups into a KeyError on a successful step.
            # Running it after the prune leaves those consumers intact and drops the
            # orphans from the pruned next-step table.
            n_evicted = self.active_table.drop_orphans(removed_refs)
            if n_evicted :
                self.loggers.info("log", "\t :=> Evicted {} orphaned active event(s) after reference purge.".format(n_evicted))

            # == Update variables ==
            self.neighbors_list = NeighborsList(
                self.system,
                self.config.atomicenvironment.rnei,
                self.config.atomicenvironment.rcut,
            )
            self.atomic_environment = AtomicEnvironment(
                self.config.atomicenvironment.style,
                self.neighbors_list.neighbors_list["rnei"],
                self.neighbors_list.neighbors_list["rcut"],
                self.config.atomicenvironment.neighbors_add,
                types=self.system.types if self.config.atomicenvironment.atom_coloring_mode == "full" else None,
                coordination_threshold=self.config.atomicenvironment.coordination_threshold,
            )
            self.inactive_ae = (
                AtomicEnvironment(
                    style="region",
                    region=self.config.inactive_atoms,
                    positions=self.system.positions,
                    atom_types=self.system.types,
                ) if self.config.inactive_atoms is not None else None
            )
            self.frozen_ae = (
                AtomicEnvironment(
                    style="region",
                    region=self.config.frozen_atoms,
                    positions=self.system.positions,
                    atom_types=self.system.types,
                ) if self.config.frozen_atoms is not None else None
            )

            # == Save Reference Table and List visited environment :
            self._save()
            self._append_snapshot_to_trajectory()
            # == Periodic restart save: a killed run resumes from the last interval
            interval = self.config.control.restart_save_interval
            if interval is not None and step % interval == 0:
                self._save_restart_file(step, total_time)
            del active_table
            # == Check if only cristalline environments ==
            if set(list(self.atomic_environment.atomic_environment_list)) == {
                "crystal"
            }:
                self.loggers.info("log", ":=> Only atoms with cristalline environment")
                self._close()
        self._save_restart_file(step, total_time, final=True)
        self._close()

    def get_new_environments(self) -> list[str]:
        """Get atomic environments of the current system that has not been already explored.

        Returns
        -------
        list[str]
            The atomic environments of the current system that are encounter for the first time.

        """
        new_environments = self.atomic_environment.get_new_environments(
            self.visited_environments
        )
        self.loggers.info(
            "log",
            "\t :=> {} new atomic environments found".format(len(new_environments)),
        )
        return new_environments

    def central_atoms_research(
        self, new_environments: list[str], nsearch: int
    ) -> list[int]:
        """Generate list of central atoms on which we gonna perform generic event searches for the reference table.

        For each new environment it adds nseach atoms having that environment to the list.

        Parameters
        ----------
        new_environments : list[str]
            List of atomic environment ID.
        nsearch : int
            Number of searches per atomic environment.

        Returns
        -------
        list[int]
            List of central atoms

        Raises
        ------
        IndexError
            If no atoms are found for a given environment, random.choice will raise an IndexError.

        """
        central_atom_research_list = []
        inactive_set = (
            set(self.inactive_ae.get_atoms_with_id("in"))
            if self.inactive_ae is not None else set()
        )
        # for each atomic environment hash in new_environment
        for env in new_environments:
            # find all index having that hash
            tmp1 = [
                i
                for i, e in enumerate(self.atomic_environment.atomic_environment_list)
                if e == env
            ]
            if inactive_set:
                tmp1 = [i for i in tmp1 if i not in inactive_set]
            if not tmp1:
                continue  # no eligible atoms for this environment
            # Randomly choose nsearch atoms that have that environment
            tmp2 = [random.choice(tmp1) for _i in range(nsearch)]
            central_atom_research_list += tmp2
        return central_atom_research_list

    def execute_event_searches(
        self, central_atom_research_list: list[int]
    ) -> EventSearch:
        """Execute an event search for each atom index in central_atom_research_list.

        Parameters
        ----------
        central_atom_research_list : list[int]
            The list of atom index on which we want to perform and event search.

        Returns
        -------
        EventSearch
            The EventSearch class containing results of the event searches.

        """
        event_search = EventSearch(self.config, self.system, self.manager, self.loggers)
        event_search.execute(central_atom_research_list)
        return event_search

    def add_reference_events(
        self, events: list[EventSearchOutput]
    ) -> list[pd.DataFrame]:
        """Add events to the reference table.

        Parameters
        ----------
        events : list[EventSearchOutput]
            List containing EventSearchOutput dataclass of successful events.

        Returns
        -------
        list[pd.DataFrame]
            List of event dataframe that has been added to the reference event table.

        """
        results_is_valid_events = self.reference_table.add_events(
            events,
            types=list(self.system.types),
            atomic_environment_list=self.atomic_environment.atomic_environment_list,
        )
        self.loggers.info(
            "log",
            "\t :=> Adding {} events to the reference table".format(
                len([e for e in results_is_valid_events if e.is_ok()])
            ),
        )
        return results_is_valid_events

    def execute_refinements(
        self,
        df_reference_events: pd.DataFrame,
        existing_pairs: set[tuple[int, int]] | None = None,
    ) -> Refinement:
        """Refine all events in df_reference_events for all atoms on which they can be apply.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            Subset of the reference table with events that can be apply to the current system.
        existing_pairs : set[tuple[int, int]] | None, optional
            `(atom_index, num_reference_event)` pairs already present in the
            persistent active table (carried over from the previous step).
            These are skipped during refinement.

        Returns
        -------
        Refinement
            The refinement class with results.

        """
        refinement = Refinement(
            self.config,
            self.loggers,
            self.system,
            self.neighbors_list,
            self.atomic_environment,
            self.manager,
        )
        #refinement.execute(df_reference_events, self.potential_energy)
        refinement.execute(df_reference_events, self.total_energy, existing_pairs=existing_pairs)
        return refinement

    def add_active_events(
        self, events: list[EventRefinementOutput]
    ) -> ActiveEventTable:
        """Create a new ActiveEventTable, add active events and return it.

        Parameters
        ----------
        events : list[RefinementsInfo]
            List of events to be added.

        Returns
        -------
        ActiveEventTable
            The active event table object.

        """
        # Extend the persistent active table (initialised once in `run()`).
        # Any rows surviving from the previous step are already present and
        # are not re-added because Refinement skipped them via existing_pairs.
        self.active_table.add_events(events)
        return self.active_table

    def _select_event(
        self,
        active_table: ActiveEventTable,
    ) -> tuple[int, float, float]:
        """Select an event in the active table based on the refection free algorithm.

        Uses ``self.bias`` when set and enabled; otherwise performs a standard
        unbiased rejection-free selection.  ``delta_t`` and ``ktot`` are always
        derived from the rates of the pool at the moment of acceptance.

        Parameters
        ----------
        active_table : ActiveEventTable
            The ActiveEventTable object with active events.

        Returns
        -------
        tuple[int, float, float]
            A typle containing :
            - int: Index of the selected event in the ActiveEventTable table.
            - float: time increment associated with the event.
            - float: total rate constant of the active events.

        """
        l_k = np.array(
            [active_table.table.loc[i].at["k"] for i in range(len(active_table.table))],
            dtype=float,
        )
        if self.bias is None:
            # Dissolution rows compete in the SAME rejection-free vector, appended
            # after the active-event rows (so a selected index >= len(table) maps
            # to a dissolution). ktot then includes them and delta_t is correct.
            if self._n_dissolution() > 0:
                l_k = np.concatenate([l_k, self._dissolution_candidates[1]])
            idx_selected_event, delta_t, ktot = rejection_free(l_k)
        else:
            idx_selected_event, delta_t, ktot = self.bias.select(
                rejection_free, l_k, active_table,
                self.system, self.reference_table, self.atomic_environment
            )
        return idx_selected_event, delta_t, ktot

    def _n_dissolution(self) -> int:
        """Return the number of dissolution candidates eligible this step.

        Returns
        -------
        int
            Number of atoms eligible to dissolve, or 0 when the feature is off.

        """
        if self._dissolution_candidates is None:
            return 0
        return len(self._dissolution_candidates[0])

    def _compute_dissolution_candidates(self) -> None:
        """Scan the current system for atoms eligible to dissolve and rate them.

        Populates ``self._dissolution_candidates`` with an aligned
        ``(indices, rates, coordinations)`` tuple (``None`` when the feature is
        off). Coordination is the per-atom first-shell (rnei) neighbour count of
        the current neighbour list, so the scan reflects the live configuration.
        """
        if not self.config.control.dissolution:
            self._dissolution_candidates = None
            return
        coordination = np.array(
            [len(n) for n in self.neighbors_list.neighbors_list["rnei"]], dtype=int
        )
        diss = self.config.dissolution
        idx = eligible_atoms(
            self.system.types, coordination, diss.elements, diss.coord_max
        )
        # Respect the region gates: an atom the run may not search on
        # (inactive_atoms) or move (frozen_atoms) must not dissolve either.
        excluded: set[int] = set()
        inactive_ae = getattr(self, "inactive_ae", None)
        if inactive_ae is not None:
            excluded |= set(inactive_ae.get_atoms_with_id("in"))
        frozen_ae = getattr(self, "frozen_ae", None)
        if frozen_ae is not None:
            excluded |= set(frozen_ae.get_atoms_with_id("in"))
        if excluded:
            idx = np.array([i for i in idx if int(i) not in excluded], dtype=int)
        coords = coordination[idx]
        rates = dissolution_rates(
            coords, diss.nu_d, diss.E_b, self.config.rateconstant.T
        )
        self._dissolution_candidates = (idx, rates, coords)

    def _delete_atom(self, atom_idx: int) -> None:
        """Remove one atom from every n_atoms-sized ``System`` array.

        ``positions``, ``types`` and ``index`` are all length-N; each is trimmed
        so the system stays internally consistent, and ``index`` is renumbered to
        ``arange(N-1)`` to match the positional numbering used everywhere else.
        Indices after ``atom_idx`` shift down by one.

        Parameters
        ----------
        atom_idx : int
            Index (current numbering) of the atom to remove.

        """
        n = len(self.system.types)
        keep = np.ones(n, dtype=bool)
        keep[atom_idx] = False
        self.system.positions = np.ascontiguousarray(self.system.positions[keep])
        # Keep types an ndarray per the System contract (create_from_file seeds a
        # list; np.asarray tolerates both).
        self.system.types = np.asarray(self.system.types)[keep]
        self.system.index = np.arange(n - 1)

    def _execute_dissolution(
        self, selection: DissolutionSelection, total_time: float
    ) -> None:
        """Delete a dissolved atom, rebuild the engines, and relax the surface.

        A dissolution event has no saddle/reconstruction: the atom simply leaves.
        Its removal shrinks the system and shifts every later atom index, so all
        n_atoms-sized structures are trimmed (:meth:`_delete_atom`), every LAMMPS
        engine (local + global instances) is rebuilt to the reduced count before
        any positions are scattered, and the system is re-minimised so the total
        energy stays honest. Neighbour lists and the classifier are rebuilt at the
        end of the step from the updated system; the active table is flushed by
        the caller (indices shifted).

        Parameters
        ----------
        selection : DissolutionSelection
            The chosen dissolution event (atom index, coordination, rate).
        total_time : float
            Simulation clock (s) at this step, for the log line.

        """
        atom_idx = selection.atom_index
        element = self.system.types[atom_idx]
        position = np.asarray(self.system.positions[atom_idx], dtype=float).copy()
        self._delete_atom(atom_idx)
        # Position (not index) is the stable identity of the dissolved atom --
        # indices shift with each deletion. Report the remaining count of each
        # dissolvable species so dealloying curves fall out of the log directly.
        remaining = ", ".join(
            "{}={}".format(el, int(np.count_nonzero(np.asarray(self.system.types) == el)))
            for el in self.config.dissolution.elements
        )
        self.loggers.info(
            "log",
            "\t :=> [dissolution] {} atom index {} coordination {} at "
            "[{:.4f}, {:.4f}, {:.4f}] dissolves; remaining {} (t = {:.6e} s)".format(
                element,
                atom_idx,
                selection.coordination,
                position[0],
                position[1],
                position[2],
                remaining,
                total_time,
            ),
        )
        # Every engine still holds the old atom count; rebuild both the local and
        # global LAMMPS instance from the reduced system (clear + reinit, the
        # _ensure_full_system(force=True) sequence) before any positions are set.
        self.manager.reinitialize_system(self.config, self.system)
        # Relax the surface after the atom left and refresh the total energy.
        # apply_positions=True so a resumed run (restart_file set) also applies
        # the relaxed positions instead of scattering the stale unrelaxed ones.
        self.minimize_system(apply_positions=True)

    def reconstruction(self, active_table) :
            #TODO make a Result

            err_reference = []
            err_ae = []
            # Dissolution rows keep the pool non-empty even after every active
            # event has been removed, so a dissolution stays selectable rather
            # than dying via the "all reconstructions failed" path below.
            while len(active_table.table) > 0 or self._n_dissolution() > 0 :
                ##=>Select event
                idx_selected_event, delta_t, ktot = self._select_event(active_table)
                ##=>A dissolution row won (index past the active rows): no event to
                ##  reconstruct, the selection always "succeeds". Return the chosen
                ##  atom so run() deletes it (delta_t/ktot already include its rate).
                n_active = len(active_table.table)
                if idx_selected_event >= n_active :
                    indices, rates, coords = self._dissolution_candidates
                    local = idx_selected_event - n_active
                    selection = DissolutionSelection(
                        atom_index=int(indices[local]),
                        coordination=int(coords[local]),
                        rate=float(rates[local]),
                    )
                    return selection, delta_t, ktot, idx_selected_event, err_reference, err_ae
                ##=>Reconstruct event
                self.loggers.info("log", "\t :=> Event Reconstruction")
                result_reconstruction = self._reconstruction_active_event(idx_selected_event, active_table)
                if result_reconstruction.is_ok() :
                    break
                else :
                    num_ref_event = active_table.table.loc[idx_selected_event].at['num_reference_event']
                    err_type = result_reconstruction.err_value().type
                    self.loggers.info("log", "\t :=> Reconstruction fails (reference event {}) :  {}".format(num_ref_event, result_reconstruction.err_value().message))
                    if err_type in _PER_ROW_RECONSTRUCTION_ERRORS :
                        # Per-row / transient miss: keep the (globally-valid)
                        # reference event, drop only this active row.
                        self.loggers.info("log", "\t :=> Per-row/transient failure; keeping reference event, dropping active event only.")
                    else :
                        # Genuine reconstruction defect: schedule the reference
                        # event (and its topology) for purge. Guard the lookup:
                        # an orphaned recycled row may point at an already-purged
                        # ref, in which case there is nothing left to remove.
                        event_ids = self.reference_table.table[self.reference_table.table['idx_ref'] == num_ref_event]['event_id'].values
                        if len(event_ids) > 0 :
                            err_reference.append(num_ref_event)
                            err_ae.append(event_ids[0])
                        else :
                            self.loggers.info("log", "\t :=> Reference event {} already purged; dropping orphaned active event.".format(num_ref_event))

                    self.loggers.info("log", "\t :=> Removing active event.")
                    active_table.remove(idx_selected_event)
            else :
                self.loggers.error("log", "All event reconstuctions failed.")
                # Non-silent death: every active event this step failed to
                # reconstruct, so there is no move to apply. Exit NONZERO (via
                # the same close/teardown path) so a truncated campaign is not
                # mistaken for a completed run (production rc=0 at half-steps).
                self._close(exit_code=1)
            return result_reconstruction, delta_t, ktot, idx_selected_event, err_reference, err_ae

    def _reconstruction_active_event(self, idx_selected_event: int, active_table: AtomicEnvironment) :
        central_atom = active_table.table.loc[idx_selected_event].at["atom_index"]
        stored_neighbors = active_table.table.loc[idx_selected_event].at["neighbors"]
        saddle_positions = copy.deepcopy(active_table.table.loc[idx_selected_event].at["saddle_positions"])
        supposed_final_positions = copy.deepcopy(active_table.table.loc[idx_selected_event].at["final_positions"])

        # Fail fast: the stored neighbour ordering is authoritative for the
        # saddle/final coordinate rows. A missing column (None) or a length
        # mismatch would scatter coords onto the wrong absolute atoms.
        if (
            stored_neighbors is None
            or saddle_positions is None
            or supposed_final_positions is None
        ):
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
                    message="Active event row is missing required reconstruction "
                    "columns (neighbors/saddle_positions/final_positions).",
                    variables={
                        "idx_selected_event": int(idx_selected_event),
                        "central_atom": int(central_atom),
                    },
                )
            )
        neighbors = np.asarray(stored_neighbors, dtype=int)
        if not (
            len(neighbors) == len(saddle_positions) == len(supposed_final_positions)
        ):
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
                    message="Active event row has an inconsistent 'neighbors' column.",
                    variables={
                        "idx_selected_event": int(idx_selected_event),
                        "central_atom": int(central_atom),
                        "n_neighbors": int(len(neighbors)),
                        "n_saddle": int(len(saddle_positions)),
                        "n_final": int(len(supposed_final_positions)),
                    },
                )
            )

        supposed_initial_positions = copy.deepcopy(self.system.positions[neighbors])




        #Move the system to the saddle point
        self.system.update_positions(new_positions= saddle_positions, atom_idx = neighbors)

        #try to reconstruct
        result = Reconstruction(self.config, self.manager, types=self.system.types).reconstruct(supposed_initial_positions, supposed_final_positions, self.system.positions, self.system.cell, neighbors, central_atom=central_atom)
        #result with min1, saddle, min2 pos

        #Back to original positions, in case reconstruction fails
        self.system.update_positions(new_positions = supposed_initial_positions, atom_idx = neighbors)
        return result

    def _apply_event(
        self, idx_selected_event: int, active_table: ActiveEventTable
    ) -> None:
        """Apply an active event to the system.

        Parameters
        ----------
        idx_selected_event : int
            index of the selected event in the active_table's table
        active_table : ActiveEventTable
            The ActiveEventTable okbject with active events.

        """
        new_positions = active_table.table.loc[idx_selected_event].at["final_positions"]
        self.system.update_positions(new_positions)

    def _apply_original_migration_event(self, result_reconstruction: Ok[ReconstructionOutput]) -> None:
        reconstruction_output = result_reconstruction.ok_value()
        self.system.update_positions(reconstruction_output.min2_positions)
        self.total_energy = reconstruction_output.min2_etot

    def minimize_system(self, positions = None, apply_positions: bool | None = None) -> None:
        """Minimize the system and update its positions.

        Parameters
        ----------
        positions : np.ndarray | None, optional
            Positions to minimise from; ``None`` uses the engine's current state.
        apply_positions : bool | None, optional
            Whether to write the minimised positions back onto ``self.system``.
            ``None`` (default) keeps the legacy init-time gate (apply only on a
            fresh run, i.e. ``restart_file is None`` -- a resumed run trusts its
            own restart positions). ``True`` always applies: the dissolution path
            passes this so a resumed run relaxes the post-deletion surface instead
            of scattering stale positions with a freshly-minimised energy.

        """
        if apply_positions is None:
            apply_positions = self.config.control.restart_file is None
        if self.config.control.restart_file is None or apply_positions:
            self.loggers.info("log", ":=> Minimizing the system")
        else :
            self.loggers.info("log", ":=> Computing energies")
        new_positions, total_energy = self.manager.global_minimize_with_results(self.config, positions=positions, types=self.system.types)
        #TEST
        #future = self.manager.minimize_with_results(self.config, positions=positions)
        #new_positions, total_energy = future.result()
        #np.savetxt('before_min.dat', self.system.positions)
        #np.savetxt('after_min.dat', new_positions)
        if apply_positions :
            self.system.update_positions(new_positions)
        self.total_energy = total_energy
        self.potential_energy = self.manager.global_get_potential_energy()

    def get_info_atomic_environments(
        self, new_environments: list[str]
    ) -> AtomicEnvironmentInfo:
        """Get atomic environments informations for outputs.

        See :func:`pykmc.info_simulation.info_atomic_environments`.

        Parameters
        ----------
        new_environments : list[str]
            List of new environments detected.

        Returns
        -------
        AtomicEnvironmentInfo
            The Dataclass with atomic environments informations.

        """
        return info_atomic_environments(self, new_environments)

    def get_info_reference_event_searches(
        self,
        results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]],
    ) -> ReferenceEventSearchInfo:
        """Get reference event searches informations for outputs.

        See :func:`pykmc.info_simulation.info_reference_event_searches`.

        Parameters
        ----------
        results_reference_event_searches : list[Result[EventSearchOutput, ErrorInfo]]
            The list of Result from event searches.

        Returns
        -------
        ReferenceEventSearchInfo
            The Dataclass with reference event searches informations.

        """
        return info_reference_event_searches(results_reference_event_searches)

    def get_info_is_valid_reference_events(
        self, results_is_valid_events: list[Result[pd.DataFrame, ErrorInfo]]
    ) -> ReferenceValidEventsInfo:
        """Get informations on whether or not an event is valid.

        See :func:`pykmc.info_simulation.info_is_valid_reference_events`.

        Parameters
        ----------
        results_is_valid_events : list[Result[pd.DataFrame, ErrorInfo]]
            List of Results from ReferenceEventTable.is_valid_event().

        Returns
        -------
        ReferenceValidEventsInfo
            The Dataclass with information on whether an event is valid or not.

        """
        return info_is_valid_reference_events(results_is_valid_events)

    def get_info_refinements(
        self, results_refinements: list[Result[EventSearchOutput, ErrorType]]
    ) -> RefinementsInfo:
        """Get informations on refined events.

        See :func:`pykmc.info_simulation.info_refinements`.

        Parameters
        ----------
        results_refinements : list[Result[EventSearchOutput, ErrorType]]
           List of Results from the refinements.

        Returns
        -------
        RefinementsInfo
           The dataclass with refinements informations.

        """
        return info_refinements(results_refinements)

    def _initialize(self) -> None:
        """Initialize the KMC attributes.

        See :func:`pykmc.Initializer.initialize()`.

        """
        Initializer(self).initialize()

    def _append_snapshot_to_trajectory(self) -> None:
        """Append the configurations positions to the trajectory file."""
        atoms = Atoms(
            self.system.types,
            positions=self.system.positions,
            cell=self.system.cell,
            pbc=self.system.pbc,
        )
        write(self.config.control.trajectory_output, atoms, append=True)

    def _save(self) -> None:
        """Save the reference event table and the list of visited environments."""
        self.reference_table.save(self.config.control.reference_table_output or "reference_table.pickle")
        with open(self.config.control.visited_environments_output, "wb") as file:
            pickle.dump(self.visited_environments, file)

    def _save_restart_file(self, last_step, last_time, final: bool = False) -> None :
        """Atomically write restart info plus a resume-ready snapshot.

        restart_latest.npz (last_step/last_time) and restart_latest.xyz (current
        minimized positions) are written via tmp + os.replace so a kill mid-write
        cannot leave a truncated file. Resume with:
            restart_file = restart_latest.npz
            initial_config = restart_latest.xyz
        With final=True the legacy end-of-run restart_<step>.npz is also written.
        """
        #Write BOTH tmp files first, then rename xyz before npz: the npz (which
        #the resume path trusts for step/time) must never point ahead of the
        #snapshot it describes; a kill between the two renames at worst re-runs
        #one interval.
        np.savez("restart_latest.tmp.npz", last_step=last_step, last_time=last_time)
        atoms = Atoms(
            self.system.types,
            positions=self.system.positions,
            cell=self.system.cell,
            pbc=self.system.pbc,
        )
        write("restart_latest.tmp.xyz", atoms)
        os.replace("restart_latest.tmp.xyz", "restart_latest.xyz")
        os.replace("restart_latest.tmp.npz", "restart_latest.npz")

        if final:
            np.savez("restart_"+str(last_step)+".npz",
                     last_step = last_step,
                     last_time = last_time)


    def _close(self, exit_code: int = 0) -> None:
        """Close the simulation.

        Parameters
        ----------
        exit_code : int, optional
            Process exit status passed to ``sys.exit`` (default 0, a healthy
            finish). The reconstruction-exhausted failure path passes a nonzero
            code so a truncated campaign is not silently reported as success.
            ``manager.close_all()`` (the MPI/engine-pool teardown) runs first
            regardless -- exactly as the healthy path already does -- so only the
            exit status differs; ``run()`` catches ``Exception`` but not the
            resulting ``SystemExit``, so the pool is still closed exactly once.

        """
        self.loggers.info("log", ":=> End of simulation")
        self.manager.close_all()
        sys.exit(exit_code)

