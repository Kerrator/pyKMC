"""Module to reconstruct an event from saddle positions"""
from pykmc.enginemanager.lmpi.pool import Manager
from pykmc import Config
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
import numpy as np
import copy 
from pykmc.utils.geometry import (
    push_towards,
    per_atom_displacement,
    minimum_image_distance,
)
import ase.geometry

#TODO: Use it in KMC
#TODO: Clean reconstruct/split the method

class Reconstruction: 

    def __init__(self, config: Config, manager: Manager, types=None) -> None :
        self.config = config
        self.manager = manager #Manager objet that can perform minimization and return minimized positions
        self.types = types

    def reconstruct(self, supposed_min1_positions, supposed_min2_positions, saddle_positions, cell, delr_thr, neighbors = None, central_atom = None) :
        """From a saddle point, try to reconstruct the event to see if it matches the 
        supposed min1 and min2 positions, and that the to minima are connected.

        Since we generaly save only the atomic environment of the central atom 
        we can specified neighbors which correspond to the list index of atoms in
        saddle positions that we need to modifie to go toward min pos.

        The reconstruction procede as follow : 
        From the saddle positions 
        Move the system toward the first minimum (with fraction)
        Minimize and compare minimized positions with supposed min1 positions 
        same for min2


        Parameters
        ----------
        supposed_min1_positions : _type_
            _description_
        supposed_min2_positions : _type_
            _description_
        saddle_positions : _type_
            _description_
        central_atom : _type_
            _description_
        cell : 
        neighbors : _type_, optional
            _description_, by default None
            typically the neighors list of the in the atomic environment of the atom on which we apply the event
        """

        if neighbors is None : #len min1 == len min2 == len saddle pos
            neighbors = np.arange(len(saddle_positions))

        matching_thr = self.config.psr.matching_score_thr

        #The atoms that actually participate in the event (largest min1->min2
        #displacement) decide whether the reconstruction landed on the right
        #state. Peripheral atoms that barely move must not veto an otherwise
        #correct reconstruction, so the acceptance check is restricted to these
        #top-n movers rather than the maximum over the whole rcut neighbourhood.
        event_disp = per_atom_displacement(supposed_min1_positions, supposed_min2_positions, cell)
        significant = np.where(event_disp > matching_thr)[0]
        if len(significant) == 0 : #degenerate event: fall back to the single largest mover
            significant = np.array([int(np.argmax(event_disp))])
        order = significant[np.argsort(event_disp[significant])[::-1]]
        movers = order[: self.config.reconstruction.n_movers]

        #Radius-containment guard: if a top mover sits in the outer rcut shell,
        #the event reaches the edge of the stored neighbourhood and the frozen
        #far field would truncate it, so reject before the expensive minimize.
        if central_atom is not None :
            central_rows = np.where(np.asarray(neighbors) == central_atom)[0]
            if len(central_rows) > 0 :
                central_pos = supposed_min1_positions[central_rows[0]]
                max_mover_r = max(
                    minimum_image_distance(central_pos, supposed_min1_positions[m], cell)
                    for m in movers
                )
                rcut_limit = self.config.atomicenvironment.rcut - self.config.reconstruction.containment_margin
                if max_mover_r > rcut_limit :
                    return Err(
                        ErrorInfo(
                            type=ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED,
                            message="event not contained in rcut : max mover radius {} > {}".format(max_mover_r, rcut_limit),
                            variables={"max_mover_r": float(max_mover_r), "rcut_limit": float(rcut_limit)},
                        )
                    )

        #Saddle positions
        tmp_positions = copy.deepcopy(saddle_positions)

        #Move toward min1 positions
        saddle_toward_min1_pos = push_towards(saddle_positions[neighbors], supposed_min1_positions, fraction=self.config.reconstruction.push_fraction, cell = cell)
        tmp_positions[neighbors] = saddle_toward_min1_pos
        #A LAMMPS error during the minimize (e.g. an unstable pushed geometry that loses
        #atoms) must drop this reconstruction, not crash the run. The engine ranks have
        #already handled the error symmetrically and are back in their service loop, so
        #the manager stays usable for the next reconstruction.
        try:
            min1_pos, _ = self.manager.global_minimize_with_results(self.config, positions=tmp_positions, types=self.types)
        except RuntimeError as exc:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
                    message="min1 reconstruction minimize failed: {}".format(exc),
                    variables={},
                )
            )

        #compare min1_pos with system current positions, restricted to the movers
        t1 = ase.geometry.wrap_positions(positions = min1_pos, cell = cell, pbc = True)
        disc1 = per_atom_displacement(supposed_min1_positions, t1[neighbors], cell)
        delr1 = float(disc1[movers].max())
        if delr1 > matching_thr :
            return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                        message="did not retreive initial minimum : delr1 = {}".format(delr1),
                        variables={"delr1": delr1},
                    )
                )
        else :
            #positions towards min2 :
            saddle_toward_min2_pos = push_towards(saddle_positions[neighbors],supposed_min2_positions, fraction=self.config.reconstruction.push_fraction, cell = cell)
            tmp_positions[neighbors] = saddle_toward_min2_pos
            try:
                min2_pos, min2_etot = self.manager.global_minimize_with_results(self.config, positions=tmp_positions, types=self.types)
            except RuntimeError as exc:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
                        message="min2 reconstruction minimize failed: {}".format(exc),
                        variables={},
                    )
                )

            #compare min2_pos with expected final positions, restricted to the movers
            t2 = ase.geometry.wrap_positions(positions = min2_pos, cell = cell, pbc = True)
            disc2 = per_atom_displacement(supposed_min2_positions, t2[neighbors], cell)
            delr2 = float(disc2[movers].max())
            if delr2 > matching_thr :
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message="did not retreive expected final minimum : delr2 = {}".format(delr2),
                        variables={"delr2": delr2},
                    )
                )

            else :
                return Ok(
                    ReconstructionOutput(
                        min1_positions=min1_pos,
                        saddle_positions=saddle_positions,
                        min2_positions=min2_pos,
                        min2_etot=min2_etot
                    )
                )

