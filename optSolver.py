# IOE 511/MATH 562, University of Michigan
# Code written by: Team NaN

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)
import numpy as np
import matplotlib.pyplot as plt
import algorithms 


def optSolver(problem,method,options):
    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    norm_g = np.linalg.norm(g,ord= np.inf)
    # parameter for terminiation
    term_tol = options.term_tol
    max_iterations  = options.max_iterations 
    
    norm_g_x0 = np.linalg.norm(problem.compute_g(problem.x0), ord=np.inf)

    # import the initial value for TR method
    if method.name == 'TRNewtonCG':
        delta = options.delta_init

    # set initial iteration counter
    k = 0

    while (k < max_iterations) and (norm_g > term_tol*max(norm_g_x0, 1)) :
        if method.name == 'TRNewtonCG':
            x_new, f_new, g_new, delta = algorithms.TRNewtonCGStep(x,problem,method,options,delta)

        else:
            print('Warning: method is not implemented yet')
    
        # update old and new function values        
        #x_old = x; f_old = f; g_old = g; norm_g_old = norm_g
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g,ord=np.inf)
        print(f'f: {f}, norm_g: {norm_g}, delta: {delta}', end='\n')

        # increment iteration counter
        k = k + 1 

    return x,f