from .result import Result, EventRefinementOutput, ErrorInfo, ErrorType, Err, Ok
from .point_set_registration import PointSetRegistration, check_match
from .utils import geometry
import ase.geometry
import numpy as np
import pandas as pd 

class Refinement() : 

    def __init__(self, config, loggers, system, neighbors_list, atomic_environment, engine): 
        self.config = config
        self.loggers = loggers
        self.system = system
        self.neighbors_list = neighbors_list
        self.atomic_environment = atomic_environment
        self.engine = engine
        self.results = None


    def execute(self, df_reference_events: pd.DataFrame) : 
        self.results = [] 

        total_refinements = self.get_total_refinements_todo(df_reference_events)
        self.loggers.info('log', "\t :=> Refining {} events".format(total_refinements)) 
        count = 0 

        for idx, dfevent in df_reference_events.iterrows() :  
            ###=>Find atoms with same atomic environment as the generic event
            atoms_refine_idx = self.atomic_environment.get_atoms_with_id(dfevent["event_id"])
            for at_idx in atoms_refine_idx : 
            ###=>refine single generic
                result_single = self.refine_single(at_idx, dfevent, idx, total_refinements, count) 
                self.results += result_single
                count += len(result_single)

    def refine_single(self, at_idx, dfevent, cat_idx, total_refinements, count) -> Result[EventRefinementOutput, ErrorInfo] : 
        ##=>PSR between generic event and at_idx environments 
        result_psr = PointSetRegistration(self.config, self.system, dfevent, self.neighbors_list, 0, at_idx).match()
        ##=>Check results if match or match < matching_score
        result_psr = check_match(result_psr, self.config.psr.matching_score_thr)
        if not result_psr.is_ok() : 
            result_psr.err_value().variables = {"n_sym_associated" : len(dfevent.at['sym_matrix'])}
            return [result_psr] #Err()
        else : 
            output_psr = result_psr.ok_value() 

            displacement = dfevent.at['saddle_positions'] - dfevent.at['initial_positions']

            all_results = [] 
            #Apply symmetries : 

            current_positions = self.system.positions.copy()
            for sym_matrix, perm_matrix in zip(dfevent.at['sym_matrix'],dfevent.at['sym_perm']):
                new_displacement = geometry.transform_positions(displacement, sym_matrix, 0, perm_matrix) 
                saddle_positions = dfevent.at['initial_positions']+new_displacement
                new_positions = geometry.transform_positions(saddle_positions, output_psr.rotation_matrix, output_psr.translation_matrix, output_psr.permutation_matrix)
                neighbors = self.neighbors_list.get_neighbors('rcut', at_idx)
                self.system.update_positions(new_positions, atom_idx=neighbors)

                result_refine = self.engine.refine_event(self.system, at_idx)

                count +=1 
                self.loggers.progress_bar('progress', count, total_refinements )

                if result_refine.is_ok() :  
                    result_refine.ok_value().num_reference_event = cat_idx
                    result_refine = self.check_refinement_minima(result_refine.ok_value(), current_positions, at_idx, self.config.eventsearch.refined_minimum_delr_thr)
                    if result_refine.is_ok() : 
                        result_refine = self.check_refinement_energy(result_refine, abs(result_refine.ok_value().dE_forward-dfevent['energy_barrier']), self.config.eventsearch.refined_energy_thr)
                self.system.update_positions(current_positions)
                all_results.append(result_refine)
            return all_results
        

    def check_refinement_minima(self, result_refine: EventRefinementOutput, current_positions, at_idx: int, minimum_delr_thr: float ) -> Result[EventRefinementOutput, ErrorInfo] : 
        """Find if min1 or min2 is initial positions """ 
        #To deal with pbc problem and lammps slighlty over/under box positions 
        dr1_vec, _ = ase.geometry.find_mic(current_positions[at_idx] - result_refine.min1_positions[at_idx],cell=self.system.cell,pbc=self.system.pbc)
        dr2_vec, _ = ase.geometry.find_mic(current_positions[at_idx] - result_refine.min2_positions[at_idx],cell=self.system.cell,pbc=self.system.pbc)
        #compare only atom that move 
        dr1 = np.sum(np.abs(dr1_vec))
        dr2 = np.sum(np.abs(dr2_vec))

        if dr1 > minimum_delr_thr and dr2 > minimum_delr_thr : 
            return Err(ErrorInfo(type=ErrorType.REFINEMENT_INVALID_MINIMA, 
                                 message="Mismatch between current positions and minima positions of the refined event."))

        elif dr1 < dr2 : 
            return Ok(result_refine)
        else : 
            result_refine.min1_positions, result_refine.min2_positions = result_refine.min2_positions, result_refine.min1_positions
            return Ok(result_refine)


    def check_refinement_energy(self, result_refine: Result[EventRefinementOutput, ErrorInfo], energy_mismatch: float, refined_energy_thr: float) -> Result[EventRefinementOutput, ErrorInfo] : 
        if energy_mismatch > refined_energy_thr : 
            return Err(ErrorInfo(type=ErrorType.REFINEMENT_INVALID_ENERGY_BARRIER, 
                             message = "refinement energy barrier does not match reference one"))
        else : 
            return result_refine
        
    def get_total_refinements_todo(self, df_reference_events: pd.DataFrame ) -> int : 
        total = 0 
        for idx, dfevent in df_reference_events.iterrows() :  
            ###=>Find atoms with same atomic environment as the generic event
            total += len(self.atomic_environment.get_atoms_with_id(dfevent["event_id"]))*len(dfevent["sym_matrix"])
        return total

    def get_successes_results(self) -> list[EventRefinementOutput]: 
        return [e.ok_value() for e in self.results if e.is_ok()]