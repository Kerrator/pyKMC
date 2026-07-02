"""Module to reconstruct an event from saddle positions"""
from pykmc.enginemanager.lmpi.pool import Manager
from pykmc import Config
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
import numpy as np
import copy 
from pykmc.utils.geometry import (
    push_towards,
    per_atom_displacement,
    event_movers,
    reconstruction_matches,
    event_contained,
)
import ase.geometry

#TODO: Use it in KMC
#TODO: Clean reconstruct/split the method

class Reconstruction: 

    def __init__(self, config: Config, manager: Manager, types=None) -> None :
        self.config = config
        self.manager = manager #Manager objet that can perform minimization and return minimized positions
        self.types = types

    def reconstruct(self, supposed_min1_positions, supposed_min2_positions, saddle_positions, cell, neighbors = None, central_atom = None, pbc: "np.ndarray | list[bool] | bool | None" = None, from_positions: "np.ndarray | None" = None) :
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
        pbc : np.ndarray or list[bool] or None, optional
            Per-axis periodicity of the system. ``None`` (default) reproduces the
            historical full-PBC behaviour. Threading the runtime ``pbc`` here
            makes the serial wrap/push and the acceptance displacement metric use
            the same geometry rule as the engine (MPI) basin path, so a
            near-boundary mover on a slab (``pbc=[True, True, False]``) is
            wrapped/pushed identically on both paths.
        from_positions : np.ndarray or None, optional
            Full-system unpushed from-state geometry (mirrors the engine's
            ``from_positions``). Passing it also flags this call as the basin
            reconstruction path: the active-volume outer-sphere freeze is applied
            only when ``from_positions`` is not ``None`` (in addition to the AV
            config), matching the engine which freezes only inside its basin op.
            When the freeze engages this array selects the frozen atom set. The
            main KMC loop caller leaves this ``None`` and therefore keeps its
            historical plain unconstrained minimize even under active volume.
        """

        if neighbors is None : #len min1 == len min2 == len saddle pos
            neighbors = np.arange(len(saddle_positions))

        matching_thr = self.config.psr.matching_score_thr

        #Active-volume outer-sphere freeze radius. On the engine (MPI) side the
        #freeze is applied ONLY inside the basin reconstruction op
        #(_basin_reconstruct_impl -> _minimize_freeze_outer_sphere); the main KMC
        #loop reconstruction (kmc.py _reconstruction_active_event) uses a plain
        #unconstrained minimize even under active_volume. To keep the serial path
        #engine-canonical the freeze must be scoped to the basin path only, not to
        #every active_volume run. The basin caller uniquely marks itself by passing
        #from_positions (the unpushed from-state geometry); the main-loop caller
        #never does. So gate on from_positions is not None in addition to the AV
        #config, otherwise the main KMC reconstruction would silently switch to an
        #outer-sphere-frozen minimize (dropping the frozen_atoms group too) and its
        #accept/reject verdicts would change. None => plain unconstrained global
        #minimize with the historical frozen-group (types).
        av_rmov = None
        if (
            from_positions is not None
            and self.config.control.active_volume
            and self.config.activevolume is not None
        ) :
            av_rmov = self.config.activevolume.rmov
        freeze_positions = from_positions if from_positions is not None else saddle_positions

        #The atoms that actually participate in the event (largest min1->min2
        #displacement) decide whether the reconstruction landed on the right
        #state. Peripheral atoms that barely move must not veto an otherwise
        #correct reconstruction, so the acceptance check is restricted to these
        #top-n movers rather than the maximum over the whole rcut neighbourhood.
        event_disp = per_atom_displacement(supposed_min1_positions, supposed_min2_positions, cell, pbc)
        movers = event_movers(event_disp, self.config.reconstruction.n_movers, matching_thr)
        shell_thr = self.config.reconstruction.shell_tolerance

        #Degenerate/empty event (no rcut shell, no movers): reject gracefully
        #rather than crash on the max()/argmax over an empty array. The serial
        #caller does not wrap reconstruct() in a try/except, so this MUST be an
        #Err, not a raise.
        if len(movers) == 0 :
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
                    message="degenerate event: no movers to reconstruct (empty shell/displacement)",
                    variables={},
                )
            )

        #Radius-containment guard: if a mover sits in the outer rcut shell at
        #ANY point of the path (min1, saddle, or min2) the event reaches the edge
        #of the stored neighbourhood and the frozen far field would truncate it,
        #so reject before the expensive minimize. An outward event that is inside
        #at min1 but crosses rcut at the saddle/min2 must also be caught. The
        #saddle rows are the mover-shell subset of the full-system saddle
        #(saddle_positions[neighbors]); an absent central row fails closed.
        contained, max_mover_r, rcut_limit = event_contained(
            central_atom,
            neighbors,
            movers,
            supposed_min1_positions,
            saddle_positions[neighbors],
            supposed_min2_positions,
            cell,
            self.config.atomicenvironment.rcut,
            self.config.reconstruction.containment_margin,
        )
        if not contained :
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
        saddle_toward_min1_pos = push_towards(saddle_positions[neighbors], supposed_min1_positions, fraction=self.config.reconstruction.push_fraction, cell = cell, pbc = pbc)
        tmp_positions[neighbors] = saddle_toward_min1_pos
        #A LAMMPS error during the minimize (e.g. an unstable pushed geometry that loses
        #atoms) must drop this reconstruction, not crash the run. The engine ranks have
        #already handled the error symmetrically and are back in their service loop, so
        #the manager stays usable for the next reconstruction.
        try:
            min1_pos, _ = self._minimize(tmp_positions, freeze_positions, central_atom, av_rmov, cell, pbc)
        except RuntimeError as exc:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
                    message="min1 reconstruction minimize failed: {}".format(exc),
                    variables={},
                )
            )

        #compare min1_pos with system current positions, restricted to the movers
        t1 = ase.geometry.wrap_positions(positions = min1_pos, cell = cell, pbc = pbc if pbc is not None else True)
        disc1 = per_atom_displacement(supposed_min1_positions, t1[neighbors], cell, pbc)
        ok1, delr1, shell1 = reconstruction_matches(disc1, movers, matching_thr, shell_thr)
        if not ok1 :
            return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                        message="did not retreive initial minimum : delr1 = {} shell = {} (shell_thr {})".format(delr1, shell1, shell_thr),
                        variables={"delr1": delr1, "delr_shell1": shell1},
                    )
                )
        else :
            #positions towards min2 :
            saddle_toward_min2_pos = push_towards(saddle_positions[neighbors],supposed_min2_positions, fraction=self.config.reconstruction.push_fraction, cell = cell, pbc = pbc)
            tmp_positions[neighbors] = saddle_toward_min2_pos
            try:
                min2_pos, min2_etot = self._minimize(tmp_positions, freeze_positions, central_atom, av_rmov, cell, pbc)
            except RuntimeError as exc:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_MINIMIZE_FAILED,
                        message="min2 reconstruction minimize failed: {}".format(exc),
                        variables={},
                    )
                )

            #compare min2_pos with expected final positions, restricted to the movers
            t2 = ase.geometry.wrap_positions(positions = min2_pos, cell = cell, pbc = pbc if pbc is not None else True)
            disc2 = per_atom_displacement(supposed_min2_positions, t2[neighbors], cell, pbc)
            ok2, delr2, shell2 = reconstruction_matches(disc2, movers, matching_thr, shell_thr)
            if not ok2 :
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message="did not retreive expected final minimum : delr2 = {} shell = {} (shell_thr {})".format(delr2, shell2, shell_thr),
                        variables={"delr2": delr2, "delr_shell2": shell2},
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

    def _minimize(
        self,
        positions: np.ndarray,
        freeze_positions: np.ndarray,
        central_atom: "int | None",
        av_rmov: "float | None",
        cell: np.ndarray,
        pbc: "np.ndarray | list[bool] | bool | None",
    ) -> "tuple[np.ndarray, float]" :
        """Global-pool minimize, freezing the AV outer sphere when active volume is on.

        Routes through the Manager's global session so the serial basin path and
        the engine (MPI) basin path relax the same geometry. When ``av_rmov`` is
        ``None`` (active volume off, central atom unavailable, or the main KMC
        loop path where ``from_positions`` was not supplied) this is the
        historical unconstrained ``global_minimize_with_results`` call carrying
        the frozen-group ``types``, so those runs are byte-for-byte unchanged.
        The freeze branch only fires on the basin path, where ``freeze_positions``
        is always the real from-state geometry (never the ``saddle_positions``
        fallback).

        Parameters
        ----------
        positions : np.ndarray
            (N, 3) pushed geometry to load and minimize.
        freeze_positions : np.ndarray
            (N, 3) unpushed from-state geometry used to select the frozen set.
        central_atom : int or None
            0-based central atom index (centre of the active volume).
        av_rmov : float or None
            Outer-sphere freeze radius; ``None`` disables the freeze.
        cell : np.ndarray
            3x3 simulation cell.
        pbc : np.ndarray or list[bool] or None
            Per-axis periodicity for the minimum-image frozen-set selection.

        Returns
        -------
        tuple[np.ndarray, float]
            ``(minimized_positions, total_energy)``.

        """
        if av_rmov is not None and central_atom is not None :
            return self.manager.global_minimize_freeze_outer_sphere_with_results(
                self.config,
                positions=positions,
                freeze_positions=freeze_positions,
                central_atom=central_atom,
                rmov=av_rmov,
                cell=cell,
                pbc=pbc,
            )
        return self.manager.global_minimize_with_results(
            self.config, positions=positions, types=self.types,
        )

