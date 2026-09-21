from __future__ import annotations

import numpy as np
import pandas as pd

from .exit_time_solver import BisectionSolver
from .connectivity import StatesConnectivity
from .utils import solve_master_equation
from pykmc.result import (
    Result,
    Ok,
    Err,
    ErrorInfo,
    ErrorType,
    BasinSelectorOutput,
    BasinExitTimeSolverOutput,
)

# TODO : Use a Abstract Selector (if implement a new one, eg MRT)
# TODO : For the moment spectral decomposition=True is hardcoded, and it is assumed that we use BisectionSolver, need to modify if use different (and use builder)

#: Relative roundoff tolerated in an occupation or flux vector. An imaginary or
#: negative part no larger than this fraction of the vector's largest positive
#: entry is spectral-solver noise and is discarded; anything larger is rejected.
FLUX_ROUNDOFF_RTOL = 64.0 * np.finfo(float).eps


class BasinExitFluxError(ValueError):
    """The exit flux at the sampled time is not a valid channel distribution.

    Raised by the direct selector calls (``select_absorbing_state`` and the
    flux accessors). ``select_from_connectivity`` transports it as
    ``Err(ErrorType.BASIN_INVALID_EXIT_FLUX)`` so the KMC loop falls back to
    the originally selected event; no approximate or uniform channel is ever
    substituted.
    """

    def __init__(self, message: str, variables: dict | None = None) -> None:
        super().__init__(message)
        self.variables = dict(variables or {})


def validate_flux(values, what: str, t_exit: float) -> np.ndarray:
    """Return ``values`` as a finite, real, non-negative vector with a positive total.

    Policy (contracts 7f, R11): a non-finite entry, a materially negative
    entry, a material imaginary part or the absence of any positive entry is
    an error, never silently repaired. Only roundoff within
    ``FLUX_ROUNDOFF_RTOL`` of the largest positive entry is discarded
    (imaginary parts dropped, negative entries clamped to zero). Tiny positive
    values are valid: the scale is the vector's own maximum.

    Parameters
    ----------
    values : array_like
        Occupation or flux vector, possibly complex from the spectral solver.
    what : str
        Name used in the error message.
    t_exit : float
        Sampled exit time (ps), reported in the error.

    Raises
    ------
    BasinExitFluxError
        If the vector is not a valid non-negative distribution up to roundoff.
    """
    values = np.asarray(values)
    if values.size == 0:
        raise BasinExitFluxError(
            f"{what} at t_exit={t_exit!r} ps is empty: no exit transition",
            {"t_exit": t_exit},
        )
    if not np.all(np.isfinite(values)):
        raise BasinExitFluxError(
            f"{what} at t_exit={t_exit!r} ps is not finite: {values.tolist()}",
            {"t_exit": t_exit, what: values.tolist()},
        )
    real = np.asarray(values.real if np.iscomplexobj(values) else values, dtype=float)
    scale = float(real.max())
    if not scale > 0.0:
        raise BasinExitFluxError(
            f"{what} at t_exit={t_exit!r} ps has no positive entry: {real.tolist()}",
            {"t_exit": t_exit, what: real.tolist()},
        )
    tolerance = FLUX_ROUNDOFF_RTOL * scale
    if np.iscomplexobj(values):
        imaginary = float(np.abs(values.imag).max())
        if imaginary > tolerance:
            raise BasinExitFluxError(
                f"{what} at t_exit={t_exit!r} ps is not real: largest imaginary "
                f"part {imaginary!r} exceeds {tolerance!r}",
                {"t_exit": t_exit, what: values.tolist()},
            )
    smallest = float(real.min())
    if smallest < -tolerance:
        raise BasinExitFluxError(
            f"{what} at t_exit={t_exit!r} ps is materially negative: {smallest!r} "
            f"below -{tolerance!r}",
            {"t_exit": t_exit, what: real.tolist()},
        )
    return np.where(real > 0.0, real, 0.0)


def normalized_weights(flux: np.ndarray) -> np.ndarray:
    """Normalise a validated flux vector to unit total, scaled by its maximum first."""
    scaled = np.asarray(flux, dtype=float) / float(np.max(flux))
    return scaled / scaled.sum()


def draw_channel(weights: np.ndarray) -> int:
    """Draw one channel index from normalised weights with a single uniform draw.

    The cumulative distribution is renormalised so that its last entry is
    exactly one, and the draw is placed with ``side="right"`` so a channel of
    zero weight is never selected (in particular the first one at draw 0).
    """
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    r2 = np.random.random()
    return int(np.searchsorted(cumulative, r2, side="right"))


class FPTASelector:
    """
    Selector implementing First Passage Time Analysis (FPTA) to determine the exit time and absorbing state of a basin.

    This class follows the procedure described in Ref. [1, 2]:

        1. Build the full generator matrix.
        2. Construct a reduced generator matrix where all absorbing states are collapsed into a single effective absorbing state.
        3. Use a numerical solver to compute the exit time from the reduced system.
        4. Given the exit time, select the exit transition from the instantaneous
           flux at that time (see :meth:`select_exit_transition`).

    Attributes
    ----------
    M_abs : np.ndarray or None
        Full absorbing generator matrix (transient + absorbing states).
    M_abs_reduced : np.ndarray or None
        Reduced matrix where all absorbing states are merged into a single one.

    Notes
    -----
    The exit channel is conditioned on the sampled exit time ``T``: with the
    transient occupation ``q(T) = exp(-M_T T) e_0`` (``M_T`` the transient
    block of ``M_abs``), the instantaneous flux through a transition
    ``e = (i -> a)`` of rate ``k_e`` is ``q_i(T) k_e`` and the flux into an
    absorbing state is ``R q(T)`` with ``R = -M_abs[n:, :n]``. The cumulative
    absorption up to ``T`` answers a different conditioning question and is
    not used.

    References
    ----------
    [1] doi.org/10.1063/1.3369627
    [2] doi.org/10.1063/5.0015039
    """

    def __init__(self) -> None:

        self.M_abs = None  # Absorbing Markov chain generator matrix
        self.M_abs_reduced = None  # Reduced absorbing markoc chain generator matrix

    @property
    def n_transient(self) -> int:
        """Number of transient states (the reduced matrix minus its merged absorber)."""
        return len(self.M_abs_reduced) - 1

    def select_from_connectivity(
        self, connectivity_table: StatesConnectivity
    ) -> Result[BasinSelectorOutput, ErrorInfo]:
        """
        Find both an exit time and an exit transition from a `StatesConnectivity` object.

        Parameters
        ----------
        connectivity_table : StatesConnectivity
            StatesConnectivity object.

        Returns
        -------
        Result[BasinSelectorOutput, ErrorInfo]
            - Ok(BasinSelectorOutput(t_exit, exit_state, exit_row, from_state))
              on success; ``exit_row`` is the connectivity-table index label of
              the selected transition.
            - Err(ErrorInfo) if the exit time solver failed
              (``BASIN_TEXIT_NOT_FOUND``) or the flux at the sampled time is
              not a valid channel distribution (``BASIN_INVALID_EXIT_FLUX``).
        """

        # Number of transient states
        n_transient_states = len(set(connectivity_table.df["state"]))

        # Build generator matrix
        self.build_absorbing_matrix_from_connectivity(connectivity_table)
        # Build reduced matrix (all absorbing states as one)
        self.build_reduced_matrix(n_transient_states)

        # Find exit time :
        result = self.get_exit_time()
        if not result.is_ok():  # Solver Err when determining t_exit
            return result
        t_exit = result.ok_value().t_exit

        # Find exit transition from the instantaneous flux at t_exit
        try:
            exit_row, from_state, exit_state = self.select_exit_transition(
                connectivity_table, t_exit
            )
        except (BasinExitFluxError, np.linalg.LinAlgError) as exc:
            variables = dict(getattr(exc, "variables", None) or {})
            variables.setdefault("t_exit", t_exit)
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_INVALID_EXIT_FLUX,
                    message="Basin: exit channel selection failed: {}".format(exc),
                    variables=variables,
                )
            )

        return Ok(
            BasinSelectorOutput(
                t_exit=t_exit,
                exit_state=exit_state,
                exit_row=exit_row,
                from_state=from_state,
            )
        )

    def build_absorbing_matrix_from_connectivity(
        self, connectivity_table: StatesConnectivity
    ) -> None:
        """
        Construct the full generator matrix M_abs from a `StatesConnectivity` object.

        The matrix is defined as:
            - M_ij = -k_ji for i ≠ j
            - M_ii = -sum_{j≠i} M_ij
        where k are the rates.

        We force the absorbing -> transient rate to be equal to 0.

        Parameters
        ----------
        connectivity_table : StatesConnectivity
            StatesConnectivity object with forward/backward rates.

        Returns
        -------

        None
        """

        # Build empty Absorbin markoc chain transition matrix
        n_states = (
            max(
                set(connectivity_table.df["state"])
                | set(connectivity_table.df["state_connexion"])
            )
            + 1
        )
        self.M_abs = np.zeros((n_states, n_states))

        # Non diagonal elements : M_ij = -k_ji
        for _, row in connectivity_table.df.iterrows():
            # for each row we find
            i = row["state"]
            j = row["state_connexion"]

            self.M_abs[j, i] -= row["k_forward"]
        # Absorbing columns will always be O since we initialize M as a Null matrix and absorbing state are never in ['state']

        # Diagonal elements : M_ii = sum_j k_ij
        for i in range(
            len(set(connectivity_table.df["state"]))
        ):  # only diag for transient states
            # since M_ij has kj->i elements
            self.M_abs[i, i] = -sum(
                [self.M_abs[j, i] for j in range(n_states) if j != i]
            )

    def build_reduced_matrix(self, n_transient_states: int) -> None:
        """
        Build the reduced generator matrix where all absorbing states are collapsed into a single absorbing state.

        Parameters
        ----------
        n_transient_states : int
            Number of transient states.

        Returns
        -------
        None

        Notes
        -----
        Reducing the absorbing block reduces the matrix size and accelerates computation of exp(-M_abs * t).
        """

        self.M_abs_reduced = np.zeros((n_transient_states + 1, n_transient_states + 1))
        # Copy the transient part
        self.M_abs_reduced[:n_transient_states, :n_transient_states] = self.M_abs[
            :n_transient_states, :n_transient_states
        ]

        ## Sum the rates of absorbing states only line is affected, last row should be = 0
        for i in range(n_transient_states):
            self.M_abs_reduced[-1, i] = self.M_abs[n_transient_states:, i].sum()

    def get_exit_time(self) -> Result[BasinExitTimeSolverOutput, ErrorInfo]:
        """
        Use Solver to find the exit time form the reduced matrix.

        Returns
        -------
        Result[BasinExitTimeSolverOutput, ErrorInfo]
            - Ok(result) containing t_exit on success.
            - Err(ErrorInfo) if solver failed.
        """

        # Initialize
        p0 = np.zeros(len(self.M_abs_reduced))
        p0[0] = 1  # we are always in state 0 when entering the basin

        # Pick random number between [0,1) representing the probability of being in an absorbing states after time t
        r1 = np.random.random()

        # Use solver :
        exit_time_solver = BisectionSolver(self.M_abs_reduced, p0, r1)
        result = exit_time_solver.solve()

        return result

    def transient_occupation(self, t_exit: float) -> np.ndarray:
        """
        Probability of occupying each transient state at ``t_exit``, entered in state 0.

        ``q(t) = exp(-M_T t) e_0`` with ``M_T`` the transient block of ``M_abs``;
        the vector is unnormalised and its total is the survival probability.

        Parameters
        ----------
        t_exit : float
            Exit time (ps).

        Returns
        -------
        np.ndarray
            Length ``n_transient`` vector, validated by :func:`validate_flux`.

        Raises
        ------
        BasinExitFluxError
            If the solver output is not a valid non-negative vector.
        """
        n = self.n_transient
        p0 = np.zeros(n)
        p0[0] = 1.0  # always at state 0 when entering the basin
        occupation = solve_master_equation(self.M_abs[:n, :n], t_exit, p0)
        return validate_flux(occupation, "transient occupation", t_exit)

    def absorbing_flux(self, t_exit: float) -> np.ndarray:
        """
        Instantaneous flux into each absorbing state at ``t_exit``.

        ``flux(t) = R q(t)`` with ``R = -M_abs[n:, :n]`` (the transient ->
        absorbing rates) and ``q`` from :meth:`transient_occupation`. Its
        total is the exit-time density; divided by the survival probability
        it is the conditional hazard at ``t_exit``.

        Parameters
        ----------
        t_exit : float
            Exit time (ps).

        Returns
        -------
        np.ndarray
            One entry per absorbing state, in the numbering of ``M_abs``.
        """
        n = self.n_transient
        flux = (-self.M_abs[n:, :n]) @ self.transient_occupation(t_exit)
        return validate_flux(flux, "absorbing flux", t_exit)

    def exit_transition_flux(
        self, connectivity_table: StatesConnectivity, t_exit: float
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Instantaneous flux through each transient -> absorbing transition at ``t_exit``.

        A transition ``e = (i -> a)`` of rate ``k_e`` carries ``q_i(t_exit) k_e``.
        Transitions are listed as :meth:`StatesConnectivity.absorbing_transitions`
        orders them (by destination, then insertion), so that a single draw over
        them reaches each absorbing state with exactly the probability
        :meth:`select_absorbing_state` gives it.

        Parameters
        ----------
        connectivity_table : StatesConnectivity
            Table whose generator was built by
            :meth:`build_absorbing_matrix_from_connectivity`.
        t_exit : float
            Exit time (ps).

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            The ordered exit transitions (original index labels kept) and
            their validated flux.
        """
        exits = connectivity_table.absorbing_transitions()
        occupation = self.transient_occupation(t_exit)
        sources = exits["state"].to_numpy(dtype=int)
        rates = exits["k_forward"].to_numpy(dtype=float)
        flux = occupation[sources] * rates
        return exits, validate_flux(flux, "exit transition flux", t_exit)

    def select_exit_transition(
        self, connectivity_table: StatesConnectivity, t_exit: float
    ) -> tuple:
        """
        Draw the exit transition at ``t_exit`` from the instantaneous transition flux.

        Parameters
        ----------
        connectivity_table : StatesConnectivity
            Table whose generator was built by
            :meth:`build_absorbing_matrix_from_connectivity`.
        t_exit : float
            Exit time (ps).

        Returns
        -------
        tuple
            ``(exit_row, from_state, exit_state)``: the connectivity-table index
            label of the selected transition, its transient source state and
            its absorbing destination.

        Raises
        ------
        BasinExitFluxError
            If the flux at ``t_exit`` is not a valid channel distribution.
        """
        exits, flux = self.exit_transition_flux(connectivity_table, t_exit)
        position = draw_channel(normalized_weights(flux))
        chosen = exits.iloc[position]
        label = exits.index[position]
        if isinstance(label, np.integer):
            label = int(label)
        return label, int(chosen["state"]), int(chosen["state_connexion"])

    def select_absorbing_state(self, t_exit: float) -> int:
        """
        Find which absorbing state is reached at the given exit time.

        The state is drawn from the instantaneous flux :meth:`absorbing_flux`
        at ``t_exit`` (not from the cumulative absorption up to ``t_exit``).

        Parameters
        ----------
        t_exit : float
            Exit time.

        Returns
        -------
        int
            Index of the absorbing state selected (matching the original
            numbering of the full matrix M_abs).

        Raises
        ------
        BasinExitFluxError
            If the flux at ``t_exit`` is not a valid channel distribution.
        """
        flux = self.absorbing_flux(t_exit)
        return self.n_transient + draw_channel(normalized_weights(flux))
