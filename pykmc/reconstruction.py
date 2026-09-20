"""Module to reconstruct an event from saddle positions"""

from pykmc.manager import Manager
from pykmc import Config
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
from pykmc.physics import ConstraintViolationError, overlay_tolerance
import numpy as np
import copy
from pykmc.utils.geometry import (
    push_towards,
    compute_delr,
    normalize_pbc,
    wrap_positions,
)

# TODO: Use it in KMC
# TODO: Clean reconstruct/split the method


class Reconstruction:
    def __init__(
        self, config: Config, manager: Manager, types=None, constraints=None, pbc=True
    ) -> None:
        self.config = config
        self.manager = manager  # Manager objet that can perform minimization and return minimized positions
        self.types = types
        self.constraints = constraints
        self.pbc = normalize_pbc(pbc)

    def reconstruct(
        self,
        supposed_min1_positions,
        supposed_min2_positions,
        saddle_positions,
        cell,
        delr_thr,
        neighbors=None,
    ):
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

        axes = normalize_pbc(self.pbc)
        if neighbors is None:  # len min1 == len min2 == len saddle pos
            neighbors = np.arange(len(saddle_positions))

        if self.constraints is not None:
            neighbors = np.asarray(neighbors)
            if (
                neighbors.ndim != 1
                or neighbors.dtype.kind not in "iu"
                or len(np.unique(neighbors)) != len(neighbors)
                or np.any(neighbors < 0)
                or np.any(neighbors >= len(saddle_positions))
            ):
                raise ValueError(
                    "reconstruction neighbors must be unique local indices"
                )
            for endpoint in (supposed_min1_positions, supposed_min2_positions):
                if np.shape(endpoint) != (len(neighbors), 3):
                    raise ValueError(
                        "reconstruction endpoint shape does not match neighbors"
                    )
            # A claimed stationary event must already obey its USER-fixed
            # references at the PSR tolerance; projecting an incompatible event
            # would silently change its physics, so it is reported as an Err
            # and the caller purges the reference. The AV shell is a transport
            # restriction that protect_positions re-clamps below (contracts 7f
            # policy 5).
            tolerance = overlay_tolerance(self.config)
            try:
                self.constraints.validate_positions(
                    saddle_positions, tolerance=tolerance, user_only=True
                )
                for endpoint in (supposed_min1_positions, supposed_min2_positions):
                    full = np.array(saddle_positions, copy=True)
                    full[neighbors] = endpoint
                    self.constraints.validate_positions(
                        full, tolerance=tolerance, user_only=True
                    )
            except ConstraintViolationError as exc:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
                        message="event changes a user-fixed reference "
                        "coordinate: {}".format(exc),
                    )
                )

        # Saddle positions
        tmp_positions = copy.deepcopy(saddle_positions)

        # Move toward min1 positions
        saddle_toward_min1_pos = push_towards(
            saddle_positions[neighbors],
            supposed_min1_positions,
            fraction=self.config.reconstruction.push_fraction,
            cell=cell,
            pbc=axes,
        )
        tmp_positions[neighbors] = saddle_toward_min1_pos
        if self.constraints is not None:
            tmp_positions = self.constraints.protect_positions(tmp_positions)
        constraint_kwargs = (
            {} if self.constraints is None else {"constraints": self.constraints}
        )
        # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
        min1_pos, _ = self.manager.group_minimize_with_results(
            config=self.config,
            positions=tmp_positions,
            types=self.types,
            **constraint_kwargs,
        )
        #        min1_pos, _ = future.result()

        # compaire min1_pos with system current positions
        t1 = wrap_positions(positions=min1_pos, cell=cell, pbc=axes)
        delr1 = compute_delr(
            supposed_min1_positions, t1[neighbors], cell, pbc=axes
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
                pbc=axes,
            )
            tmp_positions[neighbors] = saddle_toward_min2_pos
            if self.constraints is not None:
                tmp_positions = self.constraints.protect_positions(tmp_positions)
            # future = self.manager.minimize_with_results(self.config, positions=tmp_positions)
            min2_pos, min2_etot = self.manager.group_minimize_with_results(
                config=self.config,
                positions=tmp_positions,
                types=self.types,
                **constraint_kwargs,
            )
            #            min2_pos, _ = future.result()

            # Compare min2pos with expected final_positions
            t2 = wrap_positions(positions=min2_pos, cell=cell, pbc=axes)
            # delr2 = compute_delr(supposed_min2_positions, min2_pos[neighbors], cell)
            delr2 = compute_delr(supposed_min2_positions, t2[neighbors], cell, pbc=axes)
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
