from scipy.io import loadmat

def test_f(a):
    return(a)

def test_g():
    
    return(None)
    
class Problem:
    def __init__(self,name,x0,compute_f,compute_g,compute_H=None):
        self.name = name
        self.x0 = x0
        #self.n = n
        self.compute_f = compute_f
        self.compute_g = compute_g
        self.compute_H = compute_H
    
    def get_data(self):
        data = loadmat('data/'+ self.name +'.mat')
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        
class Method:
    def __init__(self,name,step_type,constant_step_size=1e-3,loss='LS'):
        self.name = name
        self.step_type = step_type
        self.constant_step_size = constant_step_size
        self.loss = loss # LS(liear leadt square) or LR(logitic regression)

class Options:
    def __init__(self,term_tol = 1e-6, max_iterations = 1e3
                 , tau = 0.5,c1 = 1e-4,alpha_bar = 1.0
                 , beta = 1e-6
                 , epsilon = 1e-6
                 , m = 5, H0 = None
                 , b = 1
                 , max_gradevaluation=20):
        # termination constants
        self.term_tol = term_tol
        self.max_iterations = max_iterations
        self.max_gradevaluation = max_gradevaluation #the multipilier for gradient evaluation for MLoptsolver
        # backtracking method constants
        self.tau = tau
        self.c1 = c1
        self.alpha_bar = alpha_bar 
        # subroutine for modifed Newton contants
        self.beta = beta
        # BFGS/LBFGS constants
        self.epsilon = epsilon
        self.H0 = H0
        # LBFGS constant
        self.m = m
        #batch size for SG
        self.b = b