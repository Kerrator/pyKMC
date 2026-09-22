"""Module to reconstruct an event from saddle positions"""

from pykmc.manager import Manager
from pykmc import Config
from pykmc.result import Result, Ok, Err, ReconstructionOutput, ErrorInfo, ErrorType
from pykmc.physics import ConstraintViolationError, overlay_tolerance
import numpy as np
import copy
from pykmc.utils.geometry import (
    push_towards,
    normalize_pbc,
    wrap_positions,
    minimum_image_displacement,
    event_movers,
    reconstruction_matches,
    event_contained,
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
        central_atom: int | None = None,
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

        Acceptance is a two-tier rule focused on the atoms that actually
        participate in the event: every event mover must land within
        ``psr.matching_score_thr``, and the whole rcut shell within the looser
        ``reconstruction.shell_tolerance`` -- a peripheral atom that merely
        settled during the minimize no longer vetoes an otherwise-correct
        reconstruction, while one that relaxed into a distinct site still
        rejects it. When ``central_atom`` is given, a radius-containment guard
        additionally rejects events whose movers reach the edge of the stored
        rcut neighbourhood before the (expensive) minimize runs.

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
        central_atom : int, optional
            Index of the event's central atom, located by value in
            ``neighbors``; enables the rcut containment guard. ``None``
            (default) disables the guard.
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

        def displacement(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
            # Per-atom minimum-image distance on the declared periodic axes only.
            return np.linalg.norm(
                minimum_image_displacement(np.asarray(post) - pre, cell, axes), axis=1
            )

        matching_thr = self.config.psr.matching_score_thr
        shell_thr = self.config.reconstruction.shell_tolerance

        # The atoms that actually participate in the event (largest min1->min2
        # displacement) decide whether the reconstruction landed on the right
        # state.
        event_disp = displacement(supposed_min1_positions, supposed_min2_positions)
        movers = event_movers(
            event_disp, self.config.reconstruction.n_movers, matching_thr
        )

        # Degenerate/empty event (no rcut shell, no movers): reject gracefully
        # rather than crash on the max()/argmax over an empty array.
        if len(movers) == 0:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA,
                    message="degenerate event: no movers to reconstruct (empty shell/displacement)",
                    variables={},
                )
            )

        # Radius-containment guard: if a CORE mover sits in the outer rcut shell
        # at ANY point of the path (min1, saddle, or min2) the event reaches the
        # edge of the stored neighbourhood and the un-stored far field would
        # truncate it, so reject before the expensive minimize. The guard is
        # measured over the top-n_movers core only (the largest displacers, the
        # atoms actually transported by the event), NOT over every atom above
        # matching_thr: a collective event's small elastic ripple
        # (0.1-0.2 A displacements) legitimately extends to the shell edge, and
        # truncating a ripple that size is harmless -- measured on a 4000-atom
        # Ni SIA system, such events reconstruct exactly, yet a whole-participant
        # guard rejects them and purges the catalogue. The saddle rows are the
        # mover-shell subset of the full-system saddle
        # (saddle_positions[neighbors]); an absent central row fails closed.
        # Without a centre the guard is disabled and needs no rcut or margin.
        if central_atom is not None:
            core_movers = movers[: self.config.reconstruction.n_movers]
            contained, max_mover_r, rcut_limit = event_contained(
                central_atom,
                neighbors,
                core_movers,
                supposed_min1_positions,
                np.asarray(saddle_positions)[neighbors],
                supposed_min2_positions,
                cell,
                self.config.atomicenvironment.rcut,
                self.config.reconstruction.containment_margin,
                pbc=axes,
            )
            if not contained:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_EVENT_NOT_CONTAINED,
                        message="event not contained in rcut : max mover radius {} > {}".format(
                            max_mover_r, rcut_limit
                        ),
                        variables={
                            "max_mover_r": float(max_mover_r),
                            "rcut_limit": float(rcut_limit),
                        },
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

        # compare min1_pos with the supposed initial positions, restricted to
        # the movers (tight) and the whole shell (loose)
        t1 = wrap_positions(positions=min1_pos, cell=cell, pbc=axes)
        disc1 = displacement(supposed_min1_positions, t1[neighbors])
        ok1, delr1, shell1 = reconstruction_matches(
            disc1, movers, matching_thr, shell_thr
        )
        if not ok1:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                    message="did not retreive initial minimum : delr1 = {} shell = {} (shell_thr {})".format(
                        delr1, shell1, shell_thr
                    ),
                    variables={"delr1": delr1, "delr_shell1": shell1},
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

            # Compare min2_pos with expected final positions, same two-tier rule
            t2 = wrap_positions(positions=min2_pos, cell=cell, pbc=axes)
            disc2 = displacement(supposed_min2_positions, t2[neighbors])
            ok2, delr2, shell2 = reconstruction_matches(
                disc2, movers, matching_thr, shell_thr
            )
            if not ok2:
                return Err(
                    ErrorInfo(
                        type=ErrorType.RECONSTRUCTION_INVALID_MIN2,
                        message="did not retreive expected final minimum : delr2 = {} shell = {} (shell_thr {})".format(
                            delr2, shell2, shell_thr
                        ),
                        variables={"delr2": delr2, "delr_shell2": shell2},
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
