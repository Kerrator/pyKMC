import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from numpy.linalg import eig, inv


class FTPASelector(): 
    """
    """

    def __init__(self) : 
        self.M_abs = None #Absorbing Markov chain transition matrix
        self.M_abs_reduced = None #Reduced absorbing markoc chain transition matrix
        
        
        
    def select_from_connectivity(self, connectivity_table) : 
        """"""

        n_transient_states = len(set(connectivity_table.df['state']))

        #Build transition matrices
        self.build_absorbing_matrix_from_connectivity(connectivity_table)
        self.build_reduced_matrix(n_transient_states)
        #solve P(texit) = (.,.,.,....,.,r1) = exp(-M_abs * texit)P(0) using bisection method
        t_exit = self.solve_exit_time_bisection()
        print("HEREREHRERE")
        print(t_exit)
        exit_state = self.select_absorbing_state(t_exit=t_exit)
        return exit_state

    def build_absorbing_matrix_from_connectivity(self, connectivity_table) : 
        """"""
    
        #Build empty Absorbin markoc chain transition matrix
        n_states = max(set(connectivity_table.df["state"]) | set(connectivity_table.df["state_connexion"])) +1
        self.M_abs = np.zeros((n_states, n_states))

        #Non diagonal elements : M_ij = -k_ji
        for _, row in connectivity_table.df.iterrows() : 
            #for each row we find 
            i = row['state'] 
            j = row['state_connexion']

            self.M_abs[i,j] = -row["k_backward"]
            self.M_abs[j,i] = -row["k_forward"]

        #Diagonal elements : M_ii = sum_j k_ij
        for i in range(n_states):
            self.M_abs[i, i] = -np.sum(self.M_abs[i, j] for j in range(n_states) if j != i)


    def build_reduced_matrix(self, n_transient_states: int) : 
        """Build reduce matrix of M_abs considering all absorbing state as one (make exponential matrix computation faster)"""

        self.M_abs_reduced = np.zeros((n_transient_states+1, n_transient_states+1)) 
        # Copy the transient part
        self.M_abs_reduced[:n_transient_states, :n_transient_states] = self.M_abs[:n_transient_states, :n_transient_states]

        ## Sum the rates of absorbing states
        for i in range(n_transient_states) : 
            self.M_abs_reduced[i,-1] = self.M_abs[i,n_transient_states:].sum()
            self.M_abs_reduced[-1, i] = self.M_abs[n_transient_states:,i].sum()

        # Recalculate the last term of the diagonal
        self.M_abs_reduced[-1, -1] = abs(self.M_abs_reduced[-1, :].sum())


    def solve_exit_time_bisection(self) : 
        
        #initial 
        p0 = np.zeros(len(self.M_abs_reduced))
        p0[0] = 1 #we are always in state 0 when entering the basin

        # Pick random number between [0,1) representing the probability of being in an absorbing states after time t
        r1 = np.random.random()
        r1=0.9
        
         #TETEETSTESTSTSET
        
        #we normalized 
        norm = np.linalg.norm(self.M_abs_reduced, ord=np.inf)
        print("NORM", norm)
        self.M_abs_reduced = self.M_abs_reduced/norm
 

        #initial t0 guess 
        t0 = 1.0 / np.sum(np.diag(self.M_abs_reduced))  
        print("t0 =", t0) 
        # Initial bisection values
        t_min, t_max = 0.0, t0
        tolerance = 1e-5

        prob_absorbing = 0 #value for tmin

        #adjust upper bound to be sure that for t_max prob_absorping > r1
        #TETEETSTESTSTSET
        
        #compute one time eigenvalues/vectors : 
        vals, vecs = eig(self.M_abs_reduced)
        vecs_inv = np.linalg.inv(vecs) 
        #we only need p[-1]
        vec_last_row = vecs[-1,:]

        print("calcul inv:", vecs @ vecs_inv)
        print("vec", vecs)
        print("vecinv", vecs_inv)
        print("vals", vals)


        while True:

            #difference_proddb_t_min_and_r = prob_absorbing - r1

            #p = expm(-t_max * self.M_abs_reduced) @ p0
            #p = expm_multiply(-t_max * self.M_abs_reduced, np.eye(len(self.M_abs_reduced))) @ p0
            #p = expm_multiply(-t_max * self.M_abs_reduced,p0)

            #Compute exponential
            #e = exponential_spectral_decomposition(self.M_abs_reduced, t_max)
            #e = normalized_expm(self.M_abs_reduced, t_max)
            #p = e @ p0

            #TEST
            #Compute one using spectral decomposition and vals,vecs eigen
            prob_absorbing = compute_p_absorbing(vals, vecs_inv,vec_last_row, t_max, p0) 


            #prob_absorbing = p[-1]
            print("new step")
            print("random =", r1)
            print("p_absorbing = ", prob_absorbing)
            print("tmax = ",t_max)
            print(self.M_abs_reduced)

            difference_prob_t_max_and_r = prob_absorbing - r1

            #if difference_prob_t_min_and_r < 0 and difference_prob_t_max_and_r < 0 :
            if  difference_prob_t_max_and_r < 0 :
                t_max = t_max * 2
                print("ajusting tmax")
            else:
                print("we break")
                break
        #end norm 
        t_max = t_max*norm
        print("tmmax = ", t_max)
        print("End Adjusting")
        #bisection method
        iteration = 0
        while True :

            t_mid = (t_min + t_max) / 2

            p = expm(-t_mid * self.M_abs_reduced) @ p0
            prob_absorbing = p[-1]
            
            if abs(t_max - t_min) / ((t_max + t_min) / 2) < tolerance: #tmax and tmin good
                break
            
            #else adjust
            print("adjusting tmid")
            print(iteration)
            if prob_absorbing < r1:
                t_min = t_mid
            else:
                t_max = t_mid
                
            iteration += 1
            if iteration > 100: #TODO : need to deal with it
                print("  Stop the bisection after 100 iterations")
                break

        t_exit = t_mid
        return t_exit


    def select_absorbing_state(self, t_exit) : 

        #Compute full probability vector 
            #initial vector
        p0 = np.zeros(len(self.M_abs))
        p0[0] = 1 #always at state 0 when entering the basin 

        #p = expm_multiply(-t_exit*self.M_abs, p0)
        p = exponential_spectral_decomposition(self.M_abs, t_exit) @ p0

        p_absorbing = p[len(self.M_abs_reduced)-1:]
        #asjust so sum gives 1 
        p_absorbing = p_absorbing/np.sum(p_absorbing)

        #choose exit state 
        p_absorbing_cumul = np.cumsum(p_absorbing)
        r2 = np.random.random()
        state_exit = np.searchsorted(p_absorbing_cumul, r2)
        print("EXIT STATE ABSORBING= ", state_exit)
        print("EXIT STATE = ", state_exit+len(self.M_abs_reduced)-1)
        return state_exit+len(self.M_abs_reduced) -1 


def exponential_spectral_decomposition(M, t) : 
    """compute exopnential exp(-Mt) using spectral decomposition, more stable than expm when low values
    uses exp(-Mt) = V exp(-lambda *t)V-1
    """
    #compute eigvector, eigenvalues
    vals, vecs = eig(M)
    #inverse eigenvector
    Vinv = inv(vecs)

    exp_vals = np.array([
        np.exp(-t * val) 
        for val in vals
    ])

    result = vecs @ np.diag(exp_vals) @ Vinv
    return np.real_if_close(result)


def normalized_expm(M, t):
    s = np.linalg.norm(M, ord=np.inf)

    M_scaled = M / s

    return expm(-t * s * M_scaled) 


def compute_p_absorbing(eigenval, eigenvecinv, last_row_eigenvec, t, p0):
        """Calcule uniquement p[-1] = probabilité d'absorption"""
        # exp(-t*M) @ p0 = V @ exp(t*vals) @ Vinv @ p0
        exp_vals = np.exp(-t * eigenval)
        temp = eigenvecinv @ p0
        print("p0=", p0)
        print("eigenvalues=", eigenval)
        print("t*eigenval = ", -t*eigenval)
        print("exp_vals = ", exp_vals)
        print("temp=", temp)
        print("eigenvec=", eigenvecinv)
        print("last_row_eigenvec=", last_row_eigenvec)
        return np.real(np.dot(last_row_eigenvec, exp_vals * temp))
