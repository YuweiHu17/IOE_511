# IOE 511/MATH 562, University of Michigan
# Code written by: Team NaN

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)
import numpy as np
import matplotlib.pyplot as plt
import algorithms 
from class_definition import *
import time

def optSolver(problem: Problem, method: Method, options: Options):
    # Records
    cpu_times = []
    
    
    # compute initial function/gradient/Hessian
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    H = problem.compute_H(x) if method.name == 'Newton' else None
    #if H is not None:
        #print("Hessian matrix shape:", H.shape)
    #else:
        #print("Hessian matrix is not used for this method")

    norm_g = np.linalg.norm(g,ord= np.inf)
    # parameter for termination
    term_tol = options.term_tol
    max_iterations  = options.max_iterations 
    
    norm_g_x0 = np.linalg.norm(problem.compute_g(problem.x0), ord=np.inf)

    # import the initial value for TR method
    if method.name == 'TRNewtonCG' or method.name == 'TRSR1CG':
        delta = options.delta_init
    
    if method.name == 'TRSR1CG':
    # B is the Hessian approximation used in SR1
        if options.B0_sr1 is  not None:
            B_sr1 = options
        else:
            B_sr1 = np.eye(len(x))

    if method.name in ['BFGS', 'BFGSW', 'DFP', 'DFPW']:
        H_BFGS = options.H0

    # Initialization for L-BFGS(or k=0 case)
    if method.name == 'L-BFGS':
        y_stored = []
        s_stored = []
        d = -g
        if method.step_type == 'wolfe':
            alpha = algorithms.wolfe_line_search(x, d, problem, options)
        elif method.step_type == 'backtracking':
            alpha = algorithms.backtracking(x, d, problem, options)
        s_stored.append(alpha*d)
        x_new = x+alpha*d
        y_stored.append(problem.compute_g(x_new)-g)

    # set initial iteration counter
    k = 0
    start = time.time()
    while (k < max_iterations) and (norm_g > term_tol*max(norm_g_x0, 1)) :
        if method.name == 'TRNewtonCG':
            x_new, f_new, g_new, delta = algorithms.TRNewtonCGStep(x,problem,method,options,delta,f,g)
        elif method.name == 'Newton':
            x_new, f_new, g_new, H = algorithms.newton_step(x, f, g, H, problem, method, options)
            #print(f'Iteration {k}, f: {f}, norm_g: {norm_g}, delta: {delta}')
        elif method.name == 'Modified Newton':
            x_new,f_new,g_new, l_computeEta = algorithms.Modified_NewtonStep(x,problem,method,options,g)
            #print(f'f: {f_new}, norm_g: {np.linalg.norm(g,ord=np.inf)}, sub_iter_eta: {l_computeEta}', end='\n')
        elif method.name == 'GradientDescent':
            x_new, f_new, g_new = algorithms.gradient_descent_step(x, f, g, problem, method, options)
        elif method.name == 'TRSR1CG':
            x_new, f_new, g_new, delta, B_sr1 = algorithms.TRSR1CGStep(x,problem,method,options,delta,f,g,B_sr1)
            #print(f'Iteration {k}, f: {f}, norm_g: {norm_g}, delta: {delta}')
        elif method.name == 'BFGS':
            x_new, f_new, g_new, H_BFGS = algorithms.BFGS_step(x, problem, method, options, H_BFGS, g)
        elif method.name == 'BFGSW':
            x_new, f_new, g_new, H_BFGS = algorithms.BFGSW_step(x, problem, method, options, H_BFGS, g)
        elif method.name == 'DFP':
            x_new, f_new, g_new, H_BFGS = algorithms.DFP_step(x, problem, method, options, H_BFGS, g)
        elif method.name == 'DFPW':
            x_new, f_new, g_new, H_BFGS = algorithms.DFP_step(x, problem, method, options, H_BFGS, g)
        elif method.name == 'L-BFGS':
            x_new, f_new, g_new = algorithms.L_BFGS_step(x, problem, method, options, y_stored, s_stored, g)                                      
        else:
            print('Warning: method is not implemented yet')
        cpu_times.append(time.time()-start)
        # update old and new function values        
        #x_old = x; f_old = f; g_old = g; norm_g_old = norm_g
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g,ord=np.inf)

        # increment iteration counter
        k = k + 1

    return x,f,k,cpu_times