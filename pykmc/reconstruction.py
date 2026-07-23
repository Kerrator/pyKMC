"""Reconstruct an event from its saddle point and verify it connects two minima.

Reconstruction is the final validation step before a selected event is applied:
starting from the saddle positions mapped onto the current configuration, the
system is pushed a fraction of the way toward each supposed minimum and
re-minimised. The event is accepted only if both relaxations recover the
expected minima within the matching threshold.
"""

from pykmc.enginemanager.lmpi.pool import Manager
from pykmc import Config
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
import numpy as np
import copy
from pykmc.utils.geometry import push_towards, compute_delr
import ase.geometry

# TODO: Use it in KMC
# TODO: Clean reconstruct/split the method


class Reconstruction:
    """Validate a mapped event by relaxing from its saddle point.

    Parameters
    ----------
    config : Config
        The simulation configuration; ``config.reconstruction.push_fraction``
        sets how far the system is pushed from the saddle toward each minimum
        and ``config.psr.matching_score_thr`` is the acceptance threshold.
    manager : Manager
        Engine session pool used to minimise the pushed configurations
        (global mode).
    types : np.ndarray, optional
        Atom types of the full system, forwarded to the minimisation engine.
    """

    def __init__(self, config: Config, manager: Manager, types=None) -> None:
        self.config = config
        self.manager = manager  # Manager objet that can perform minimization and return minimized positions
        self.types = types

    def reconstruct(
        self,
        supposed_min1_positions,
        supposed_min2_positions,
        saddle_positions,
        cell,
        delr_thr,
        neighbors=None,
    ):
        """Check that relaxing from the saddle recovers both expected minima.

        From the saddle positions, the system is pushed a fraction of the way
        toward the first supposed minimum, minimised, and the result compared
        with ``supposed_min1_positions``; the same is then done for the second
        minimum. The event is valid only if both comparisons fall below
        ``config.psr.matching_score_thr``.

        Since generally only the atomic environment of the central atom is
        stored, ``neighbors`` gives the indices in the full system of the atoms
        described by the supposed minima.

        Parameters
        ----------
        supposed_min1_positions : np.ndarray
            Expected positions of the ``neighbors`` atoms at the initial
            minimum.
        supposed_min2_positions : np.ndarray
            Expected positions of the ``neighbors`` atoms at the final
            minimum.
        saddle_positions : np.ndarray
            Positions of the full system with the event's atoms at the saddle
            point.
        cell : np.ndarray
            Simulation cell, used for periodic wrapping and minimum-image
            distances.
        delr_thr : float
            Matching threshold argument (currently unread — the implementation
            uses ``config.psr.matching_score_thr`` directly).
        neighbors : np.ndarray, optional
            Indices of the atoms described by the supposed minima, typically
            the ``rcut`` neighbour list of the event's central atom. Defaults
            to all atoms.

        Returns
        -------
        Result[ReconstructionOutput, ErrorInfo]
            ``Ok`` with the relaxed ``min1``/``min2`` positions, the saddle
            positions, and the final minimum's total energy; ``Err`` with
            ``RECONSTRUCTION_INVALID_MIN1`` / ``RECONSTRUCTION_INVALID_MIN2``
            when the corresponding minimum is not recovered.
        """

        if neighbors is None:  # len min1 == len min2 == len saddle pos
            neighbors = np.arange(len(saddle_positions))

        # Saddle positions
        tmp_positions = copy.deepcopy(saddle_positions)

        # Move toward min1 positions
        saddle_toward_min1_pos = push_towards(
            saddle_positions[neighbors],
            supposed_min1_positions,
            fraction=self.config.reconstruction.push_fraction,
            cell=cell,
        )
        tmp_positions[neighbors] = saddle_toward_min1_pos
        # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
        min1_pos, _ = self.manager.global_minimize_with_results(
            self.config, positions=tmp_positions, types=self.types
        )
        #        min1_pos, _ = future.result()

        # compaire min1_pos with system current positions
        t1 = ase.geometry.wrap_positions(positions=min1_pos, cell=cell, pbc=True)
        delr1 = compute_delr(
            supposed_min1_positions, t1[neighbors], cell
        )  # I guess we need to be carefull here, if atom_modify sort 0 it's ok
        if delr1 > self.config.psr.matching_score_thr:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                    message="did not retreive initial minimum : delr1 = {}".format(
                        delr1
                    ),
                    variables={"delr1": delr1},
                )
            )
        else:
            # positions towards min2 :
            saddle_toward_min2_pos = push_towards(
                saddle_positions[neighbors],
                supposed_min2_positions,
                fraction=self.config.reconstruction.push_fraction,
                cell=cell,
            )
            tmp_positions[neighbors] = saddle_toward_min2_pos
            # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
            min2_pos, min2_etot = self.manager.global_minimize_with_results(
                self.config, positions=tmp_positions, types=self.types
            )
            #            min2_pos, _ = future.result()

            # Compare min2pos with expected final_positions
            t2 = ase.geometry.wrap_positions(positions=min2_pos, cell=cell, pbc=True)
            # delr2 = compute_delr(supposed_min2_positions, min2_pos[neighbors], cell)
            delr2 = compute_delr(supposed_min2_positions, t2[neighbors], cell)
            if delr2 > self.config.psr.matching_score_thr:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message="did not retreive expected final minimum : delr2 = {}".format(
                            delr2
                        ),
                        variables={"delr2": delr2},
                    )
                )

            else:
                return Ok(
                    ReconstructionOutput(
                        min1_positions=min1_pos,
                        saddle_positions=saddle_positions,
                        min2_positions=min2_pos,
                        min2_etot=min2_etot,
                    )
                )
