# IOE 511/MATH 562, University of Michigan
# Code written by: Team NaN

# Compute the next step for all iterative optimization algorithms given current solution x:
# 1. GradientDescent, with backtracking line search
# 2. GradientDescentW, with Wolfe line search
# 3. Newton, (modified Newton) with backtracking line search
# 4. NewtonW, (modified Newton) with Wolfe line search
# 5. TRNewtonCG, trust region Newton with CG subproblem solver
# 6. TRSR1CG, SR1 quasi-Newton with CG subproblem solver
# 7. BFGS, BFGS quasi-Newton with backtracking line search
# 8. BFGSW, BFGS quasi-Newton with Wolfe line search
# 9. DFP, DFP quasi-Newton with backtracking line search
# 10. DFPW, DFP quasi-Newton with Wolfe line search

import numpy as np


# 5.0 CG_steighhaug, CG subproblem solver
def CG_steighhaug(x, g, B, delta, options):
    z = np.zeros(len(x)).reshape(-1,1)
    r = g
    p = -r
    if np.linalg.norm(r) < options.term_tol_CG:
        return z
    j = 0
    while True:
        if not (p.T@B@p)[0,0] > 0:
            a = (np.linalg.norm(p))**2; b = 2*p.T@z
            c = (np.linalg.norm(z))**2 - delta**2
            tau = (-b + np.sqrt(b**2-4*a*c))/(2*a)
            return z + tau*p
        alpha = (r.T@r/(p.T@B@p))[0,0]
        z = z + alpha*p
        if  not np.linalg.norm(z) < delta:
            a = (np.linalg.norm(p))**2; b = 2*p.T@z
            c = (np.linalg.norm(z))**2 - delta**2
            tau = (-b + np.sqrt(b**2-4*a*c))/(2*a)
            return z + tau*p
        r_new = r + alpha*B@p
        if not np.linalg.norm(r) > options.term_tol_CG:
            return z
        beta = (r_new.T@r_new)[0,0]/(r.T@r)[0,0]
        p = -r_new + beta*p
        r = r_new
        j += 1

# 5.1 TRNewtonCG, trust region Newton with CG subproblem solver
def TRNewtonCGStep(x,problem,method,options,delta):
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    B = problem.compute_H(x)

    # CG Steighhaug for suboptimization problem
    d = CG_steighhaug(x, g, B, delta, options)
    rho = (f - problem.compute_f(x + d)) / (-g.T@d - 0.5*d.T@B@d)
    if rho > options.c_1_tr:
        x_new = x + d
        if rho > options.c_2_tr:
            delta = 3*delta
    else:
        x_new = x
        delta = 0.5*delta

    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    
    return x_new, f_new, g_new, delta



