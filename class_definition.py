from scipy.io import loadmat

  

class Problem:
    def __init__(self,name,x0,compute_f,compute_g,compute_H=None):
        self.name = name
        self.x0 = x0
        #self.n = n
        self.compute_f_ = compute_f
        self.compute_g_ = compute_g
        self.compute_H = compute_H
        self.f_eval = 0
        self.g_eval = 0
    
    def compute_f(self, x):
        self.f_eval += 1
        return self.compute_f_(x)
    
    def compute_g(self, x):
        self.g_eval += 1
        return self.compute_g_(x)
        
        
class Method:
    def __init__(self,name,step_type=None,constant_step_size=1e-3):
        self.name = name
        # step_type: 'None' or 'backtracking' or ''wolfe'
        self.step_type = step_type
        self.constant_step_size = constant_step_size

class Options:
    def __init__(self,term_tol=1e-6, max_iterations=1e3, 
                 tau=0.5, c1=1e-4, alpha_bar=1.0, c2_curve=0.9,
                 beta=1e-6,
                 epsilon=1e-6,
                 H0=None, m=5,
                 c_1_tr=1e-3, c_2_tr=0.7,
                 term_tol_CG=1e-6,
                 delta_init=1,
                 c_3_sr1 = 1e-4,
                 B0_sr1 = None):
        
        # termination constants
        self.term_tol = term_tol
        self.max_iterations = max_iterations
        
        # backtracking method constants
        self.tau = tau
        self.c1 = c1
        self.alpha_bar = alpha_bar 

        # Wolfe line search (Curvature condition)
        self.c2_curve = c2_curve

        # subroutine for modifed Newton contants
        self.beta = beta
        
        # BFGS/LBFGS constants
        self.epsilon = epsilon
        self.H0 = H0
        
        # LBFGS constant
        self.m = m
        
        # Parameters for TR radius update
        self.c_1_tr = c_1_tr
        self.c_2_tr = c_2_tr
        
        # Parameters for TR-CG subproblem
        self.term_tol_CG = term_tol_CG
        self.delta_init = delta_init # Parameters for initial TR range

        # Parameters for determing update B or not in SR1
        self.c_3_sr1 = c_3_sr1
        # Initial Hessian approximation for SR1
        self.B0_sr1 = B0_sr1
