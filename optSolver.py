# IOE 511/MATH 562, University of Michigan
# Code written by: Team NaN

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)
import numpy as np
import matplotlib.pyplot as plt
import algorithms 
from class_definition import *
import timeit

def optSolver(problem: Problem, method: Method, options: Options):
    # check the necessary fields in the problem and method
    if not problem.check():
        print('Error: problem is not properly defined')
        return None
    if not method.check():
        print('Error: method is not properly defined')
        return None

    # records
    cpu_times = []
    f_values = []
    norm_g_values = []
    
    # compute initial function/gradient/Hessian is needed
    x = problem.x0
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    H = problem.compute_H(x) if method.name == 'Newton' else None

    norm_g = np.linalg.norm(g,ord= np.inf)
    # parameter for termination
    term_tol = options.term_tol
    max_iterations  = options.max_iterations 
    
    norm_g_x0 = np.linalg.norm(problem.compute_g(problem.x0), ord=np.inf)

    f_values.append(f)
    norm_g_values.append(norm_g)
    cpu_times.append(0)

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
        n = x.shape[0]
        H_BFGS = np.eye(n)

    # initialization for L-BFGS(or k=0 case)
    if method.name == 'L-BFGS':
        y_stored = []
        s_stored = []
        d = -g
        if method.step_type == 'wolfe':
            alpha = algorithms.wolfe_line_search(x, f, g, d, problem, options)
        elif method.step_type == 'backtracking':
            alpha = algorithms.backtracking(x, f, g, d, problem, options)
        s_stored.append(alpha*d)
        x_new = x+alpha*d
        y_stored.append(problem.compute_g(x_new)-g)

    # set initial iteration counter
    k = 0
    start = timeit.default_timer()
    while (k < max_iterations) and (norm_g > term_tol*max(norm_g_x0, 1)) :
        if method.name == 'TRNewtonCG':
            x_new, f_new, g_new, delta = algorithms.TRNewtonCGStep(x, problem, method, options, delta, f, g)
        elif method.name == 'Newton':
            x_new, f_new, g_new, H = algorithms.newton_step(x, f, g, H, problem, method, options)
        elif method.name == 'Modified Newton':
            x_new,f_new,g_new, l_computeEta = algorithms.Modified_NewtonStep(x, f, g, problem, method, options)
        elif method.name == 'GradientDescent':
            x_new, f_new, g_new = algorithms.gradient_descent_step(x, f, g, problem, method, options)
        elif method.name == 'TRSR1CG':
            x_new, f_new, g_new, delta, B_sr1 = algorithms.TRSR1CGStep(x, problem, method, options, delta, f, g, B_sr1)
        elif method.name == 'BFGS':
            x_new, f_new, g_new, H_BFGS = algorithms.BFGS_step(x, f, g, H_BFGS, problem, method, options)
        elif method.name == 'DFP':
            x_new, f_new, g_new, H_BFGS = algorithms.DFP_step(x, f, g, H_BFGS, problem, method, options)
        elif method.name == 'L-BFGS':
            x_new, f_new, g_new = algorithms.L_BFGS_step(x, f, g, problem, method, options, y_stored, s_stored)                                      
        else:
            print('Warning: method is not implemented yet')
        
        # update new function values        
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g,ord=np.inf)
        cpu_times.append(timeit.default_timer() - start)
        f_values.append(f)
        norm_g_values.append(norm_g)

        # increment iteration counter
        k = k + 1

    if k == max_iterations:
        print('Warning: Maximum number of iterations reached. Consider increasing max_iterations.')
    return x, f, k, cpu_times, f_values, norm_g_values

def optSolver_NaN(problem: Problem, method: Method, options: Options):
    x, f, _, _, _, _ = optSolver(problem, method, options)
    return x, f