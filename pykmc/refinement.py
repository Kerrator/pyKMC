"""Module implementing the Refinement class that deals with the event refinement procedure."""

from .result import Result, EventRefinementOutput, ErrorInfo, ErrorType, Err, Ok
from .point_set_registration import PointSetRegistration, check_match
from .utils import geometry
from .config import Config
from .system import System
from .neighbors_list import NeighborsList
from .log import LogKMC
from .atomic_environment import AtomicEnvironment
from .enginemanager.lmpi.pool import Manager
import ase.geometry
import numpy as np
import pandas as pd
import concurrent.futures


class Refinement:
    """Perfrom event refinements and deal with results.

    Parameters
    ----------
    config : Config
        The configuration of the simulation.
    loggers : LogKMC
        The logger of the KMC simulation.
    system : System
        The atomic system.
    neighbors_list : NeighborsList
        The neighbors lists of the system.
    atomic_environment : AtomicEnvironment
        The atomic environment of the system.
    engine : Engine
        The engine to use for the refinement.

    """

    def __init__(
        self,
        config: Config,
        loggers: LogKMC,
        system: System,
        neighbors_list: NeighborsList,
        atomic_environment: AtomicEnvironment,
        manager: Manager,
    ) -> None:
        self.config = config
        self.loggers = loggers
        self.system = system
        self.neighbors_list = neighbors_list
        self.atomic_environment = atomic_environment
        self.manager = manager
        self.results = None

    def execute(self, df_reference_events: pd.DataFrame, total_energy) -> None:
        """Execute event refinements for each reference event in the df_reference_events dataframe.

        It stores the results of the event refinements in self.results.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            dataframe of reference events to refine.

        """
        self.results = []

        total_refinements = self.get_total_refinements_todo(df_reference_events)
        self.loggers.info("log", "\t :=> Refining {} events".format(total_refinements))
        #count = 0

        all_futures = []
        future_context = {}  # mapping future -> contexte
        #Launch all refine Jobs 
        for idx, dfevent in df_reference_events.iterrows():
            ###=>Find atoms with same atomic environment as the generic event
            atoms_refine_idx = self.atomic_environment.get_atoms_with_id(
                dfevent["event_id"]
            )
            for at_idx in atoms_refine_idx:
                ###=>refine single generic
                futures = self.refine_single(
                    at_idx, dfevent, idx, total_refinements,  future_context
                )
                if isinstance(futures,list): #If symmetries
                    all_futures.extend(futures)
                else:
                    all_futures.append(futures)
                #self.results += result_single
                #count = len(all_futures)

        #Get results and update values : 
        for f in all_futures:
            #get results
            res = f.result()
            job_queue_len = self.manager.job_queue.qsize()
            #if job_queue_len != 0 : 
                #self.loggers.progress_bar("progress", total_refinements-job_queue_len, total_refinements)
            #get specific results info
            ctx = future_context[f]

            _ = future_context.pop(f)

            self.loggers.progress_bar("progress", total_refinements-len(future_context), total_refinements)
            #update result 
            if res.is_ok() : 
                res.ok_value().min2_positions = ctx["min2_positions"]
                res.ok_value().num_reference_event = ctx["num_reference_event"]
                res.ok_value().dE_forward = res.ok_value().E_saddle - total_energy 
                res.ok_value().saddle_positions = res.ok_value().saddle_positions[ctx["neighbors"]]
                #Now check if energy barrier consistent with generic one 
                res = self.check_refinement_energy(res,
                            abs(
                                res.ok_value().dE_forward
                                - ctx["reference_energy_barrier"] 
                            ),
                            self.config.eventsearch.refined_energy_thr,
                        )
            self.results.append(res)


    def refine_single(
        self,
        at_idx: int,
        dfevent: pd.Series,
        cat_idx: int,
        total_refinements: int,
        future_context: dict
    ) -> list[Result[EventRefinementOutput, ErrorInfo]]:
        """Perform a single reference event refinement.

        If a reference event has symmetries, it also refine those symmetric events.

        Parameters
        ----------
        at_idx : int
            index of the central atom for which we perform the refinement.
        dfevent : pd.Series
            a Series of the reference event to refine.
        cat_idx : int
            index of the event in the reference event table.
        total_refinements : int
            total refinements to do.
        count : int
            number of refinements already done.

        Returns
        -------
        list[Result[EventRefinementOutput, ErrorInfo]]
            list of Result of the refinements.

        """
        ##=>PSR between generic event and at_idx environments
        result_psr = PointSetRegistration(
            self.config, self.system, dfevent, self.neighbors_list, at_idx
        ).match()
        ##=>Check results if match or match < matching_score
        result_psr = check_match(result_psr, self.config.psr.matching_score_thr)
        if not result_psr.is_ok():
            result_psr.err_value().variables = {
                "n_sym_associated": len(dfevent.at["sym_matrix"])
            }
            f = concurrent.futures.Future()
            f.set_result(result_psr)
            future_context[f] = {}
            return f
#            return [result_psr]  # Err()
        else:
            output_psr = result_psr.ok_value()

            displacement = (
                dfevent.at["saddle_positions"] - dfevent.at["initial_positions"]
            )

            #all_results = []
            futures = []
            # Apply symmetries :

            current_positions = self.system.positions.copy()
            #initial_potential_energy = self.system.total_energy
            for sym_matrix, perm_matrix in zip(
                dfevent.at["sym_matrix"], dfevent.at["sym_perm"], strict=False
            ):
                new_displacement = geometry.transform_positions(
                    displacement, sym_matrix, 0, perm_matrix
                )
                saddle_positions = dfevent.at["initial_positions"] + new_displacement
                new_positions = geometry.transform_positions(
                    saddle_positions,
                    output_psr.rotation_matrix,
                    output_psr.translation_matrix,
                    output_psr.permutation_matrix,
                )
                neighbors = self.neighbors_list.get_neighbors("rcut", at_idx)

                self.system.update_positions(new_positions, atom_idx=neighbors)
                #add a job to manager queue
                f = self.manager.partn_refine(self.config, at_idx, self.system.positions.copy()) #send copy not reference !
                futures.append(f)


                #NOTE: TEMPORARY, NEED TO FIND A BETTER WAY
                #Update Result : 
                #        #Apply psr to generic final positions
                final_positions = dfevent.at["final_positions"] + new_displacement
                new_positions = geometry.transform_positions(final_positions, output_psr.rotation_matrix, output_psr.translation_matrix, output_psr.permutation_matrix)
                # self.system.update_positions(new_positions, atom_idx=neighbors )
                future_context[f] = {
                    #"min2_positions": self.system.positions.copy()[neighbors],
                    "min2_positions": final_positions,
                    "num_reference_event": cat_idx, 
                    "reference_energy_barrier": dfevent["energy_barrier"],
                    "neighbors": neighbors
                }


                #if result_refine.is_ok():
                #    #Update Result : 
                #        #Apply psr to generic final positions
                #    final_positions = dfevent.at["final_positions"] + new_displacement
                #    new_positions = geometry.transform_positions(final_positions, output_psr.rotation_matrix, output_psr.translation_matrix, output_psr.permutation_matrix)
                #    self.system.update_positions(new_positions, atom_idx=neighbors )
                #    result_refine.ok_value().min2_positions = self.system.positions 
                #        #Compute dE 
                #    result_refine.ok_value().dE_forward = result_refine.ok_value().E_saddle - initial_potential_energy  

                #    result_refine.ok_value().num_reference_event = cat_idx

                #        #Check if energy barrier consistent with generic one
                #    result_refine = self.check_refinement_energy(
                #            result_refine,
                #            abs(
                #                result_refine.ok_value().dE_forward
                #                - dfevent["energy_barrier"]
                #            ),
                #            self.config.eventsearch.refined_energy_thr,
                #        )
                self.system.update_positions(current_positions)
                #all_results.append(result_refine)
            return futures



    def check_refinement_energy(
        self,
        result_refine: Result[EventRefinementOutput, ErrorInfo],
        energy_mismatch: float,
        refined_energy_thr: float,
    ) -> Result[EventRefinementOutput, ErrorInfo]:
        """Check if the energy barrier of the refinement correspond the one of the reference event.

        Parameters
        ----------
        result_refine : Result[EventRefinementOutput, ErrorInfo]
            Results of the refinement procedure.
        energy_mismatch : float
            Difference between the reference event energy barrier and the refine one.
        refined_energy_thr : float
            maximum allowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy

        Returns
        -------
        Result[EventRefinementOutput, ErrorInfo]
            list of results of the procedure.

        """
        if energy_mismatch > refined_energy_thr:
            return Err(
                ErrorInfo(
                    type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER,
                    message="refinement energy barrier does not match reference one",
                )
            )
        else:
            return result_refine

    def get_total_refinements_todo(self, df_reference_events: pd.DataFrame) -> int:
        """Give the total number of refinements to do.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            dataframe with reference events to refine.

        Returns
        -------
        int
            total number of refinements to do.

        """
        total = 0
        for _idx, dfevent in df_reference_events.iterrows():
            ###=>Find atoms with same atomic environment as the generic event
            total += len(
                self.atomic_environment.get_atoms_with_id(dfevent["event_id"])
            ) * len(dfevent["sym_matrix"])
        return total

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return successful results.

        Returns
        -------
        list[EventRefinementOutput]
            list of EventRefinementOutpout dataclass with refine event's informations.

        """
        return [e.ok_value() for e in self.results if e.is_ok()]
