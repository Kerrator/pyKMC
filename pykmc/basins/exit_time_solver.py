import logging

import numpy as np
from numpy.linalg import eig, inv
from pykmc.result import Result, Ok, Err, ErrorInfo, ErrorType,  BasinExitTimeSolverOutput
from .utils import solve_master_equation_last_value

logger = logging.getLogger("log")

#TODO : Use an abstract Solver (if implement new one)
#TODO : Might need to move solve_master_equation and solve_master_equation_last_value has independant function (if implement new solver)
#TODO : max_iteration, tolerance are parameters but never used, so kind of hardcoded (could be added to config)

class BisectionSolver() : 
    """Find the exit time `t_exit` such that:

        p_abs(t_exit) = p[-1](t_exit) = r

    where p(t) = exp(-M t) p0 is the solution of the master equation. 
    The method proceeds in two steps:

        1. Determine a finite upper bound t_max for which p_abs(t_max) >= r
        2. Use bisection on [0, t_max] to solve p_abs(t) = r

    Assumptions
    -----------
    - M is a generator matrix: diagonal entries >= 0, off-diagonal entries <= 0.
    - All absorbing probabilities are grouped into the last state. 
      Meaning M sould be a (n_transient_states+1)x(n_transient_states+1) matrix.
      i.e. p_abs(t) = p[-1] is the probability of being absorbed.

    Parameters
    ----------
    M : np.ndarray
        Generator matrix of shape (n, n).
    p0 : np.ndarray
        Initial probability distribution vector of size n.
    r : float
        Target absorbed probability (0 <= r < 1).
    spectral_decomposition : bool, default=True
        If True, computes exp(-Mt)p0 using eigen decomposition of M,
        else falls back to scipy.linalg.expm.
    tolerance : float, default=1e-5
        Relative tolerance on interval width for bisection.

    Notes
    -----
    The exit time is stored internally in `self.t_exit` and is returned, by solve(),
    wrapped in a `Result` object.

    """

    def __init__(self, M: np.ndarray, p0: np.ndarray, r: float,  spectral_decomposition = True, tolerance:float = 1e-3) -> None:

        self.M = M 
        self.p0 = p0 
        self.spectral_decomposition = spectral_decomposition
        self.r = r
        self.tolerance = tolerance

        #Initialization
        self.t_max = 0 
        self.t_min = 0
        self.t_exit = -1

        #Compute only one time eigen values/vector of M when using spectral decomposition.
        #The expm path must still carry the (unused) attributes: solve_master_equation_last_value
        #receives them unconditionally, and a missing attribute crashed spectral_decomposition=False.
        self.Valeig: "np.ndarray | None" = None
        self.Veceig: "np.ndarray | None" = None
        self.Veceiginv: "np.ndarray | None" = None
        if self.spectral_decomposition == True : 
            self.Valeig, self.Veceig = eig(self.M)
            self.Veceiginv = inv(self.Veceig)


    def solve(self) -> Result[BasinExitTimeSolverOutput, ErrorInfo]: 
        """Compute the exit time `t_exit` using:

        1. determine_tmax()
        2. determine_texit()

        Returns
        -------
        Result[BasinExitTimeSolverOutput, ErrorInfo]
            - Ok(BasinExitTimeSolverOutput) on success, where BasinExitTimeSolverOutput.t_exit contains the exit time
            - Err(ErrorInfo) if any step fails

        """
        result = self.determine_tmax()        
        if not result.is_ok() :  
            return result #Determine t_max Err

        result = self.determine_texit()
        if not result.is_ok() : 
            return result #Determine t_exit Err
        
        return Ok(BasinExitTimeSolverOutput(t_exit=self.t_exit))


    def determine_tmax(self, max_iterations:int = 2000) -> Result[None, ErrorInfo]: 
        """Determine a finite upper bound t_max such that:

            p_abs(t_max) >= r

        where p_abs(t) is the absorbing probability p(t)[-1].

        Strategy:
        ----------
        - Start from t_max = 1 / sum(diag(M))
        - Double t_max until p_abs(t_max) >= r or max_iterations reached

        Parameters
        ----------
        max_iterations : int
            Maximum number of doubling steps.

        Returns
        -------
        Result[None, ErrorInfo]
            - Ok(None) if t_max successfully found
            - Err(ErrorInfo) if no suitable t_max is found

        """
        #first guess 
        self.t_max = 1.0/np.sum(np.diag(self.M))

        iterations = 0
        while iterations < max_iterations : 
            p_abs = solve_master_equation_last_value(M=self.M, t=self.t_max, p0=self.p0, spectral_decomposition=self.spectral_decomposition, Valeig=self.Valeig, Veceig=self.Veceig, Veceiginv=self.Veceiginv)
            if p_abs - self.r > 0 : 
                break 
            else : 
                self.t_max *= 2
            iterations +=1 
        else : #No breack so we reached max_iterations
            return Err(ErrorInfo(type=ErrorType.BASIN_TEXIT_NOT_FOUND, message=("Basin: could not find t_max using bisection method after {} iterations".format(iterations)))
                       )
        return Ok(None)
        
    
    def determine_texit(self, max_iterations: int = 50000) -> Result[None, ErrorInfo]: 
        """Compute t_exit such that p_abs(t_exit) = r using bisection.

        The algorithm assumes:
        - p_abs(t_min) <= r
        - p_abs(t_max) >= r
        which is guaranteed if determine_tmax() succeeded.

        Parameters
        ----------
        max_iterations : int
            Maximum number of bisection iterations.

        Returns
        -------
        Result[None, ErrorInfo]
            - Ok(None) on success (t_exit stored in self.t_exit)
            - Err(ErrorInfo) if tolerance not reached after max_iterations

        """
        iterations = 0

        while iterations < max_iterations : 
            t_mid = (self.t_min + self.t_max) / 2

            if abs(self.t_max - self.t_min) / ((self.t_max + self.t_min) / 2) < self.tolerance: #tmax and tmin good
                break
            
            p_abs = solve_master_equation_last_value(M=self.M, t=t_mid, p0=self.p0, spectral_decomposition=self.spectral_decomposition, Valeig=self.Valeig, Veceig=self.Veceig, Veceiginv=self.Veceiginv)

            if p_abs-self.r < 0:
                self.t_min = t_mid
            else:
                self.t_max = t_mid
                
            iterations += 1
        else : #No break so we reached max_iterations
            return Err(ErrorInfo(type=ErrorType.BASIN_TEXIT_NOT_FOUND, message=("Basins: could not find t_exit using bisection method after {} iterations".format(iterations)), variables={"tmin":self.t_min, "tmax": self.t_max, "tmid": t_mid, "r": self.r}))

        self.t_exit = t_mid
        return Ok(None)
#Relative residual |Q tau - p0| / |p0| above which the direct MFPT solve is
#considered untrustworthy and the cancellation-free reduction takes over. For an
#M-matrix solved by LU the residual measured this way tracks the forward relative
#error (it is ~eps * cond(Q) when p0 is a unit vector), so the threshold is a
#conditioning test, not a stiffness test. Measured on the two-state chain of
#tests/basins/test_exit_time_solver.py the direct error is 5e-5 at a residual of
#7e-5 and 5.5e-4 at 7e-4, so 1e-5 keeps the direct path at least 20x inside the
#1e-3 accuracy requirement while leaving well-conditioned generators - including
#the stiffness-1e7 slow-passage case, residual 6e-7 - on the fast path.
MFPT_RESIDUAL_TOL = 1.0e-5


def occupation_times_gth(M: np.ndarray, p0: np.ndarray) -> "np.ndarray | None":
    """Transient occupation times by GTH state reduction, without cancellation.

    Solves the same system as ``tau = Q^-1 p0`` (expected time spent in each
    transient state before absorption) but never forms a difference of two
    positive numbers, so it is accurate to ~n*eps at *any* stiffness. Every step
    is an addition, multiplication or division of non-negative quantities:

    - the rates are read from the off-diagonals only, ``k(i->j) = -M[j, i]`` and
      ``gamma_i = -M[-1, i]``, and each total out-rate is rebuilt as
      ``d_i = gamma_i + sum_j k(i->j)`` - never from ``M[i, i]``, whose stored
      value has already lost every escape rate smaller than ``eps * d_i``;
    - eliminating state ``n`` folds the paths through it into the survivors,
      ``k'(i->j) = k(i->j) + k(i->n) k(n->j) / d_n``,
      ``gamma'_i = gamma_i + k(i->n) gamma_n / d_n``,
      ``p0'_j = p0_j + k(n->j) p0_n / d_n``, and drops the resulting self-loop by
      rebuilding ``d'_i`` as the sum of the surviving out-rates (Grassmann,
      Taksar and Heyman; equal to ``d_i - k(i->n) k(n->i) / d_n`` but computed
      additively);
    - back-substitution recovers each eliminated ``tau_n`` from the level it was
      removed at, ``tau_n = (p0_n + sum_i k(i->n) tau_i) / d_n``.

    Cost is O(n^3) in a Python-level loop over the eliminations: 13-23 s at
    n = 2000 (the basins `max_states` cap) against 0.2-0.5 s for
    ``np.linalg.solve``, 7 ms at n = 100. Hence the fallback path, not the
    default one.

    Parameters
    ----------
    M : np.ndarray
        Reduced generator matrix of shape (n_transient+1, n_transient+1); the
        last row holds the (negated) escape rates into the merged absorbing state.
    p0 : np.ndarray
        Initial probability distribution; only the first n entries are used.

    Returns
    -------
    np.ndarray or None
        The occupation times ``tau``, or None when some state has no outgoing
        rate at all (no absorbing escape is reachable, so no exit time exists).

    """
    n = len(M) - 1
    if n < 1:
        return None

    #k[i, j] = rate i -> j. The generator invariant makes the off-diagonals <= 0;
    #clamp so the reduction's non-negativity precondition holds by construction.
    k = np.maximum(-np.asarray(M[:n, :n], dtype=np.float64).T, 0.0)
    np.fill_diagonal(k, 0.0)
    gamma = np.maximum(-np.asarray(M[-1, :n], dtype=np.float64), 0.0)
    p = np.maximum(np.asarray(p0[:n], dtype=np.float64), 0.0).copy()
    d = gamma + k.sum(axis=1)

    #Values of the level each state is eliminated at, for the back-substitution.
    kin_at: "list[np.ndarray | None]" = [None] * n
    p_at = np.zeros(n)
    d_at = np.zeros(n)

    for m in range(n - 1, 0, -1):
        d_m = d[m]
        if not (d_m > 0.0) or not np.isfinite(d_m):
            return None
        kin = k[:m, m].copy()      #k(i -> m)
        kout = k[m, :m].copy()     #k(m -> j)
        kin_at[m], p_at[m], d_at[m] = kin, p[m], d_m

        k = k[:m, :m] + np.outer(kin, kout) / d_m
        np.fill_diagonal(k, 0.0)   #self-loops are absorbed by rebuilding d below
        gamma = gamma[:m] + kin * (gamma[m] / d_m)
        p = p[:m] + kout * (p[m] / d_m)
        d = gamma + k.sum(axis=1)

    if not (d[0] > 0.0) or not np.isfinite(d[0]):
        return None

    tau = np.empty(n)
    tau[0] = p[0] / d[0]
    for m in range(1, n):
        kin_m = kin_at[m]
        if kin_m is None:  #unreachable: every m in [1, n) was eliminated above
            return None
        tau[m] = (p_at[m] + float(kin_m @ tau[:m])) / d_at[m]

    if not np.all(np.isfinite(tau)):
        return None
    return tau


class QSDSolver():
    """Analytical exit time solver based on the mean first-passage time (MFPT).

    Used when the reduced generator matrix is stiff: transient mixing rates
    are many orders of magnitude larger than absorbing escape rates, making
    the numerical matrix exponential unreliable.

    With Q the transient block of M (columns = out-rates, dp/dt = -Q p), the
    expected time spent in each transient state before absorption when starting
    from p0 is tau = Q^-1 p0[:n], and the MFPT is sum(tau). The exit time is
    drawn from an exponential of mean MFPT (exact in the stiff limit, where the
    first-passage-time distribution is exponential), i.e. k_eff = 1 / MFPT.
    ``qsd`` holds tau normalised to one: in the stiff limit it is the
    quasi-stationary distribution, and in general weighting the absorbing
    rates by it (as ``FPTASelector.select_absorbing_state`` does) gives the
    exact absorption probabilities sum_i tau_i k_{i->j}.

    This replaces the earlier closed-chain stationary distribution, which was
    only valid when *every* transient state mixes faster than *any* escape:
    a slow internal passage (fast 0<->1, slow 1<->2, escape only from 2) made
    it underestimate the exit time by orders of magnitude.

    The MFPT itself is computed twice over: ``np.linalg.solve`` first, then, if
    that solve is ill-conditioned, the cancellation-free GTH state reduction of
    ``occupation_times_gth``. The direct solve carries the escape rate only
    inside Q's diagonal ``d_i = (internal + escape)``; once ``escape / internal``
    drops below eps that diagonal rounds to the internal rate alone, Q becomes
    numerically singular and the solve returns garbage or raises. The reduction
    never forms that sum-then-difference, so it is exact at any stiffness and,
    unlike the closed-chain form, also correct when the escape sits behind a slow
    internal passage. Which path ran is recorded in ``method``, the relative
    residual of the direct solve in ``residual``.

    The switch is a conditioning test on the direct solve (relative residual,
    non-finite or negative occupation times, LinAlgError), never a stiffness
    heuristic: a stiff generator whose direct solve is accurate stays on the
    fast path.

    Parameters
    ----------
    M : np.ndarray
        Reduced generator matrix of shape (n_transient+1, n_transient+1).
        Last row/column corresponds to the merged absorbing state.
    p0 : np.ndarray
        Initial probability distribution vector (unused, kept for interface compatibility).
    r : float
        Target absorbed probability (0 <= r < 1).

    """

    def __init__(self, M: np.ndarray, p0: np.ndarray, r: float) -> None:
        self.M = M
        self.p0 = p0
        self.r = r
        self.t_exit = -1.0
        self.qsd: np.ndarray | None = None
        self.k_eff: float | None = None
        self.method: str = "none"     #'mfpt-direct' | 'mfpt-gth' | 'none'
        self.residual: float = np.inf  #relative residual of the direct MFPT solve

    def solve(self) -> Result[BasinExitTimeSolverOutput, ErrorInfo]:
        """Compute the exit time using the QSD approach.

        Runs the direct MFPT solve, tests its conditioning, and falls back to the
        cancellation-free GTH reduction when the direct solve cannot be trusted.

        Returns
        -------
        Result[BasinExitTimeSolverOutput, ErrorInfo]
            - Ok(BasinExitTimeSolverOutput) on success
            - Err(ErrorInfo) if neither path yields a positive finite MFPT

        """
        n = len(self.M) - 1  # number of transient states

        # Open transient generator (diagonal keeps the absorbing leakage) and the
        # transient part of the entry distribution.
        Q = np.asarray(self.M[:n, :n], dtype=np.float64)
        p0_t = np.array(self.p0[:n], dtype=np.float64)

        # Mean occupation times tau = Q^-1 p0, fast path.
        tau: "np.ndarray | None"
        try:
            tau = np.linalg.solve(Q, p0_t)
        except np.linalg.LinAlgError:
            tau = None

        # Conditioning test on that solve. |Q tau - p0| / |p0| is ~eps*cond(Q)
        # here, so it estimates the forward relative error; a negative or
        # non-finite occupation time is garbage outright.
        if tau is not None:
            norm_p0 = float(np.linalg.norm(p0_t))
            self.residual = float(np.linalg.norm(Q @ tau - p0_t) / norm_p0) if norm_p0 > 0 else np.inf
            mfpt = float(np.sum(tau))
            trustworthy = (
                np.all(np.isfinite(tau))
                and np.isfinite(mfpt) and mfpt > 0
                and not np.any(tau < -1e-12 * abs(mfpt))
                and self.residual <= MFPT_RESIDUAL_TOL
            )
        else:
            trustworthy = False

        if trustworthy:
            self.method = "mfpt-direct"
        else:
            # Ill-conditioned (typically Q's diagonal has rounded away the escape
            # rates): redo the same MFPT without any cancellation.
            logger.info("[FPTA] QSD solver: direct MFPT solve rejected "
                        "(residual=%.3e > %.1e or non-positive tau); using GTH reduction",
                        self.residual, MFPT_RESIDUAL_TOL)
            tau = occupation_times_gth(self.M, self.p0)
            self.method = "mfpt-gth"

        mfpt = float(np.sum(tau)) if tau is not None else np.nan
        if tau is None or not np.isfinite(mfpt) or mfpt <= 0 or np.any(tau < -1e-12 * abs(mfpt)):
            self.qsd = None
            self.k_eff = 0.0
            self.method = "none"
            return Err(ErrorInfo(
                type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                message="QSD solver: k_eff <= 0, no absorbing escape possible"))

        self.qsd = np.maximum(tau, 0.0) / mfpt
        self.k_eff = 1.0 / mfpt

        logger.info("[FPTA] QSD solver (%s): k_eff=%.6e (MFPT=%.6e), qsd_min=%.6e, qsd_max=%.6e",
                    self.method, self.k_eff, mfpt, np.min(self.qsd), np.max(self.qsd))

        # Exit time from exponential distribution: P(t < T) = 1 - exp(-k_eff * T)
        # Solving for T: T = -ln(1 - r) / k_eff
        self.t_exit = -np.log(1.0 - self.r) / self.k_eff

        return Ok(BasinExitTimeSolverOutput(t_exit=self.t_exit))
