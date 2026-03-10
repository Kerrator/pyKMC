import logging
import numpy as np
from scipy.sparse.linalg import expm
from numpy.linalg import eig, inv
from pykmc.result import Result, Ok, Err, ErrorInfo, ErrorType,  BasinExitTimeSolverOutput
from .utils import solve_master_equation_last_value

logger = logging.getLogger("log")

#TODO : Use an abstract Solver (if implement new one)
#TODO : Might need to move solve_master_equation and solve_master_equation_last_value has independant function (if implement new solver)
#TODO : max_iteration, tolerance are parameters but never used, so kind of hardcoded (could be added to config)

class BisectionSolver() :
    """
    Find the exit time `t_exit` such that:

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

        #Compute only one time eigen values/vector of M when using spectral decomposition
        if self.spectral_decomposition == True :
            self.Valeig, self.Veceig = eig(self.M)
            self.Veceiginv = inv(self.Veceig)


    def solve(self) -> Result[BasinExitTimeSolverOutput, ErrorInfo]:
        """
        Compute the exit time `t_exit` using:

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
        """
        Determine a finite upper bound t_max such that:

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
        """
        Compute t_exit such that p_abs(t_exit) = r using bisection.

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


class QSDSolver():
    """
    Analytical exit time solver based on quasi-stationary distribution (QSD).

    Used when the reduced generator matrix is stiff: transient mixing rates
    are many orders of magnitude larger than absorbing escape rates, making
    the numerical matrix exponential unreliable.

    In this regime the system rapidly equilibrates among transient states
    to a quasi-stationary distribution pi_qs, then escapes exponentially
    with effective rate k_eff = pi_qs . gamma, where gamma[i] is the total
    absorbing escape rate from transient state i.

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

    def solve(self) -> Result[BasinExitTimeSolverOutput, ErrorInfo]:
        """
        Compute the exit time using the QSD approach.

        Returns
        -------
        Result[BasinExitTimeSolverOutput, ErrorInfo]
            - Ok(BasinExitTimeSolverOutput) on success
            - Err(ErrorInfo) if k_eff <= 0
        """
        n = len(self.M) - 1  # number of transient states

        # Absorbing escape rates: gamma[i] = -M[-1, i] (positive)
        gamma = -self.M[-1, :n]

        # Build closed transient generator Q_tt
        # Start from the transient block, then adjust diagonal to remove
        # the absorbing leakage (make columns sum to zero within transient block)
        Q_tt = self.M[:n, :n].copy()
        for i in range(n):
            Q_tt[i, i] -= gamma[i]

        # Find QSD: null vector of Q_tt via SVD (numerically stable)
        U, s, Vh = np.linalg.svd(Q_tt)
        null_vec = np.real(Vh[-1, :])  # last row = smallest singular value
        null_vec = np.abs(null_vec)
        self.qsd = null_vec / np.sum(null_vec)

        # Effective escape rate
        self.k_eff = float(np.dot(self.qsd, gamma))

        logger.info("[FPTA] QSD solver: k_eff=%.6e, qsd_min=%.6e, qsd_max=%.6e",
                    self.k_eff, np.min(self.qsd), np.max(self.qsd))

        if self.k_eff <= 0:
            return Err(ErrorInfo(
                type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                message="QSD solver: k_eff <= 0, no absorbing escape possible"))

        # Exit time from exponential distribution: P(t < T) = 1 - exp(-k_eff * T)
        # Solving for T: T = -ln(1 - r) / k_eff
        self.t_exit = -np.log(1.0 - self.r) / self.k_eff

        return Ok(BasinExitTimeSolverOutput(t_exit=self.t_exit))
