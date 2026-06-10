import logging

import numpy as np
from .exit_time_solver import BisectionSolver, QSDSolver
from .connectivity import StatesConnectivity
from .utils import solve_master_equation
from pykmc.result import Result, Ok, ErrorInfo, BasinSelectorOutput, BasinExitTimeSolverOutput

logger = logging.getLogger("log")

#TODO : Use a Abstract Selector (if implement a new one, eg MRT) 
#TODO : For the moment spectral decomposition=True is hardcoded, and it is assumed that we use BisectionSolver, need to modify if use different (and use builder)

class FPTASelector(): 
    """Selector implementing First Passage Time Analysis (FPTA) to determine the exit time and absorbing state of a basin.

    This class follows the procedure described in Ref. [1, 2]:

        1. Build the full generator matrix.
        2. Construct a reduced generator matrix where all absorbing states are collapsed into a single effective absorbing state.
        3. Use a numerical solver to compute the exit time from the reduced system.
        4. Given the exit time, compute the probability distribution over the original absorbing states and select the exit state.

    Attributes
    ---------- 
    M_abs : np.ndarray or None
        Full absorbing generator matrix (transient + absorbing states).
    M_abs_reduced : np.ndarray or None
        Reduced matrix where all absorbing states are merged into a single one. 

    References
    ---------- 
    [1] doi.org/10.1063/1.3369627 
    [2] doi.org/10.1063/5.0015039

    """

    def __init__(self, solver: str = "auto") -> None :

        self.M_abs = None #Absorbing Markov chain generator matrix
        self.M_abs_reduced = None #Reduced absorbing markoc chain generator matrix
        self.solver = solver #'auto' | 'bisection' | 'qsd' (see get_exit_time)
        self._use_qsd = False #set by get_exit_time; selects the exit-state weighting
        self._qsd = None #quasi-stationary distribution when the QSD solver was used

    def select_from_connectivity(self, connectivity_table: StatesConnectivity) -> Result[BasinSelectorOutput, ErrorInfo] : 
        """Find both an exit time and an exit absorbing state from a `StatesConnectivity` object.

        Parameters
        ----------
        connectivity_table : StatesConnectivity 
            StatesConnectivity object. 

        Returns
        -------
        Result[BasinSelectorOutput, ErrorInfo]
            - Ok(BasinSelectorOutput(t_exit, exit_state) ) on success.
            - Err(ErrorInfo) if exit time solver failed.

        """
        #Number of transient states
        n_transient_states = len(set(connectivity_table.df["state"]))

        #Build generator matrix
        self.build_absorbing_matrix_from_connectivity(connectivity_table)
        #Build reduced matrix (all absorbing states as one)
        self.build_reduced_matrix(n_transient_states)

        #Find exit time :
        result = self.get_exit_time()
        if not result.is_ok() : #Solver Err when determining t_exit
            return result 
        t_exit = result.ok_value().t_exit

        #Find exit state
        exit_state = self.select_absorbing_state(t_exit=t_exit)

        return Ok(BasinSelectorOutput(t_exit=t_exit, exit_state=exit_state))

    def build_absorbing_matrix_from_connectivity(self, connectivity_table: StatesConnectivity) -> None: 
        """Construct the full generator matrix M_abs from a `StatesConnectivity` object.

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
        #Build empty Absorbin markoc chain transition matrix
        n_states = max(set(connectivity_table.df["state"]) | set(connectivity_table.df["state_connexion"])) +1
        self.M_abs = np.zeros((n_states, n_states))

        #Non diagonal elements : M_ij = -k_ji
        for _, row in connectivity_table.df.iterrows() : 
            #for each row we find 
            i = row["state"] 
            j = row["state_connexion"]

            self.M_abs[j,i] -=  row["k_forward"]
        #Absorbing columns will always be O since we initialize M as a Null matrix and absorbing state are never in ['state']

        #Diagonal elements : M_ii = sum_j k_ij
        for i in range(len(set(connectivity_table.df["state"]))): #only diag for transient states
            self.M_abs[i, i] = -sum([self.M_abs[i, j] for j in range(n_states) if j != i])


    def build_reduced_matrix(self, n_transient_states: int) -> None: 
        """Build the reduced generator matrix where all absorbing states are collapsed into a single absorbing state.

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
        self.M_abs_reduced = np.zeros((n_transient_states+1, n_transient_states+1)) 
        # Copy the transient part
        self.M_abs_reduced[:n_transient_states, :n_transient_states] = self.M_abs[:n_transient_states, :n_transient_states]

        ## Sum the rates of absorbing states only line is affected, last row should be = 0
        for i in range(n_transient_states) : 
            self.M_abs_reduced[-1, i] = self.M_abs[n_transient_states:,i].sum()

    def get_exit_time(self) -> Result[BasinExitTimeSolverOutput, ErrorInfo]:
        """Use a solver to find the exit time from the reduced matrix.

        The solver is selected by ``self.solver``:

        - ``'auto'`` (default): use the QSD solver when the reduced generator is stiff
          (max transient rate / max absorbing rate > 1e6), otherwise bisection. The
          numerical matrix-exponential bisection is unreliable for stiff generators.
        - ``'bisection'``: always use the bisection solver.
        - ``'qsd'``: always use the QSD solver.

        Returns
        -------
        Result[BasinExitTimeSolverOutput, ErrorInfo]
            - Ok(result) containing t_exit on success.
            - Err(ErrorInfo) if the solver failed.

        """
        #Initialize
        p0 = np.zeros(len(self.M_abs_reduced))
        p0[0] = 1 #we are always in state 0 when entering the basin

        # Pick random number between [0,1) representing the probability of being in an absorbing states after time t
        r1 = np.random.random()

        #Decide which solver to use
        n = len(self.M_abs_reduced) - 1  #number of transient states
        max_transient_rate = np.max(np.diag(self.M_abs_reduced)[:n])
        max_absorbing_rate = np.max(np.abs(self.M_abs_reduced[-1, :n]))
        stiffness = max_transient_rate / max_absorbing_rate if max_absorbing_rate > 0 else np.inf

        use_qsd = self.solver == "qsd" or (self.solver == "auto" and stiffness > 1e6)

        if use_qsd:
            logger.info("[FPTA] Using QSD solver (solver=%s, stiffness=%.2e)", self.solver, stiffness)
            exit_time_solver = QSDSolver(self.M_abs_reduced, p0, r1)
            result = exit_time_solver.solve()
            if result.is_ok():
                self._use_qsd = True
                self._qsd = exit_time_solver.qsd
            return result

        self._use_qsd = False
        self._qsd = None
        exit_time_solver = BisectionSolver(self.M_abs_reduced, p0, r1)
        return exit_time_solver.solve()

    def select_absorbing_state(self, t_exit: float) -> int: 
        """Find which absorbing state is reached at the given exit time.

        Parameters
        ----------
        t_exit : float
            Exit time. 

        Returns
        -------
        int
            Index of the absorbing state selected (matching the original
            numbering of the full matrix M_abs).

        """
        n_transient = len(self.M_abs_reduced) - 1

        if self._use_qsd and self._qsd is not None:
            # QSD path: exit-state probabilities from the quasi-stationary distribution.
            # The matrix exponential is unreliable for the stiff generator that triggered
            # the QSD solver, so weight each absorbing target j by the QSD-averaged rate
            # into it: p_j ∝ sum_i pi_qs[i] * |M_abs[j, i]|.
            n_absorbing = len(self.M_abs) - n_transient
            p_absorbing = np.zeros(n_absorbing)
            for j_idx in range(n_absorbing):
                j = n_transient + j_idx
                p_absorbing[j_idx] = np.sum(self._qsd * np.abs(self.M_abs[j, :n_transient]))
        else:
            #Compute full probability vector
                #initial vector
            p0 = np.zeros(len(self.M_abs))
            p0[0] = 1 #always at state 0 when entering the basin

            #compute P = ext(-Mt)p0
            p = solve_master_equation(self.M_abs, t_exit, p0)

            #Select only absorbing state
            p_absorbing = p[n_transient:]

        #adjust so sum gives 1 (clamp numerical negatives first)
        p_absorbing = np.real(p_absorbing)
        p_absorbing = np.maximum(p_absorbing, 0)
        total = np.sum(p_absorbing)
        if total > 0:
            p_absorbing = p_absorbing / total
        else:
            p_absorbing = np.ones(len(p_absorbing)) / len(p_absorbing)

        #choose exit state
        p_absorbing_cumul = np.cumsum(p_absorbing)
        r2 = np.random.random()
        state_exit = np.searchsorted(p_absorbing_cumul, r2)

        return state_exit + n_transient 
    

