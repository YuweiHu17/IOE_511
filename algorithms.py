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
from class_definition import *
import scipy.sparse as sparse

# Subroutine for backtracking line search and Wolfe line search
def backtracking(x: np.ndarray, f: np.ndarray, g: np.ndarray, d: np.ndarray, problem: Problem, options: Options):
    alpha = options.alpha_bar
    while problem.compute_f(x+alpha*d) > f+(options.c1)*alpha*(g.T)@d:
        alpha = alpha*options.tau
    return alpha

def wolfe_line_search(x: np.ndarray, f: np.ndarray, g: np.ndarray, d: np.ndarray, problem: Problem, options: Options):
    alpha = options.alpha_bar
    i = 0
    alpha_low = options.alpha_low
    alpha_high = options.alpha_high
    c = options.c_wolfe
    while i < options.max_iterations:
        if problem.compute_f(x + alpha * d) < f + options.c1 * alpha * g.T @ d:
            if problem.compute_g(x + alpha * d).T @ d > options.c2_curve * g.T @ d:
                break
        if problem.compute_f(x + alpha * d) < f + options.c1 * alpha * g.T @ d:
            alpha_low = alpha
        else:
            alpha_high = alpha
        alpha = c*alpha_low + (1-c)*alpha_high
        i += 1
    return alpha

# gradient descent step
def gradient_descent_step(x, f, g, problem, method, options):
    d = -g
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, f, g, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, f, g, d, problem, options)    
    else:
        raise ValueError('Step type is not defined')

    # renewal x
    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    return x_new, f_new, g_new

# 2.0 Newton, (modified Newton) with backtracking line search/wolfe line search
def newton_step(x, f, g, H, problem, method, options):
    # Check if H is a sparse matrix and convert it to a dense array if it is
    if sparse.issparse(H):
        H_dense = H.toarray()
    else:
        H_dense = H  # Use the dense matrix directly

    try:
        d = -np.linalg.solve(H_dense, g)
    except np.linalg.LinAlgError:
        raise ValueError("Failed to solve Hd = -g with np.linalg.solve due to singular Hessian.")

    if method.step_type == 'backtracking':
        alpha = backtracking(x, f, g, d, problem, options)
    elif method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, f, g, d, problem, options)
    else:
        raise ValueError('Step type is not defined for Newton method')

    x_new = x + alpha * d
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    H_new = problem.compute_H(x_new)

    return x_new, f_new, g_new, H_new


# the subroutine to select eta_k for modified Newton Method
def compute_eta(A, beta, max_iter):
    n = A.shape[0]
    # initialize eta
    eta_val = 0 if min(A.diagonal()) > 0 else (-min(A.diagonal()) + beta)
    l = 0
    while l < max_iter:
        try:
            L = np.linalg.cholesky(A + eta_val * np.identity(n))
            break  
        except:
            eta_val = max(2 * eta_val, beta)
            l += 1
    return L, l

def Modified_NewtonStep(x, f, g, problem, method, options):
    #g = problem.compute_g(x)
    H = problem.compute_H(x)

    L, l_computeEta = compute_eta(A=H, beta=options.beta, max_iter=options.max_iterations)
    d = -np.linalg.solve(L@L.T, g) # solve the linear system to get direction  

    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, f, g, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, f, g, d, problem, options)   
    else:
        print('Warning: step type is not defined')
    
    x_new = x + alpha * d  
    f_new = problem.compute_f(x_new)  
    g_new = problem.compute_g(x_new)  

    return x_new, f_new, g_new, l_computeEta

# 5.0 CG_steighhaug, CG subproblem solver
def CG_steighhaug(x, g, B, delta, options):
    # B here is the Hessian or Hessian approximation
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
            tau = ((-b + np.sqrt(b**2-4*a*c))/(2*a))[0,0]
            return z + tau*p
        alpha = (r.T@r/(p.T@B@p))[0,0]
        z = z + alpha*p
        if  not np.linalg.norm(z) < delta:
            a = (np.linalg.norm(p))**2; b = 2*p.T@z
            c = (np.linalg.norm(z))**2 - delta**2
            tau = ((-b + np.sqrt(b**2-4*a*c))/(2*a))[0,0]
            return z + tau*p
        r_new = r + alpha*B@p
        if not np.linalg.norm(r) > options.term_tol_CG:
            return z
        beta = (r_new.T@r_new)[0,0]/(r.T@r)[0,0]
        p = -r_new + beta*p
        r = r_new
        j += 1

# 5.1 TRNewtonCG, trust region Newton with CG subproblem solver
def TRNewtonCGStep(x,problem,method,options,delta,f,g):
    B = problem.compute_H(x)

    # CG Steighhaug for sub-optimization problem
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

# 6. TRSR1CG, SR1 quasi-Newton with CG subproblem solver
def TRSR1CGStep(x,problem,method,options,delta,f,g,B):
    # B is the Hessian approximation from previous step
    # CG Steighhaug for sub-optimization problem
    d = CG_steighhaug(x, g, B, delta, options)
    rho = (f - problem.compute_f(x + d)) / (-g.T@d - 0.5*d.T@B@d)[0,0]
    if rho > options.c_1_tr:
        x_new = x + d
        if rho > options.c_2_tr:
            delta = 2*delta
    else:
        x_new = x
        delta = 0.5*delta

    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    s_k = x_new - x; y_k = g_new - g
    residual = y_k - B@s_k
    if np.abs(residual.T@s_k) > options.c_3_sr1 * np.linalg.norm(residual) * np.linalg.norm(s_k):
        B = B + (residual@residual.T) / (residual.T@s_k)[0,0] # update B_k+1

    return x_new, f_new, g_new, delta, B


def BFGS_step(x: np.ndarray, f:np.ndarray, g:np.ndarray, H: np.ndarray, 
              problem: Problem, method: Method, options: Options):
    n = x.shape[0]
    d = -H@g
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, f, g, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, f, g, d, problem, options) 
    x_new = x + alpha*d
    s = x_new - x
    y = problem.compute_g(x_new) - g

    # Skip if the inner product of s and y is not sufficiently positive
    if s.T@y >= options.epsilon*(np.linalg.norm(s))*(np.linalg.norm(y)):
        rho = 1/((s.T@y)[0][0])
        V = np.eye(n) -rho*y@(s.T)
        H_new = (V.T)@H@V + rho*s@(s.T)
    else:
        H_new = H

    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, H_new


def DFP_step(x: np.ndarray, f: np.ndarray, g: np.ndarray, H: np.ndarray,
             problem: Problem, method: Method, options: Options):
    d = -H@g
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, f, g, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, f, g, d, problem, options)
    x_new = x + alpha*d
    s = x_new - x
    y = problem.compute_g(x_new) - g

    # Skip if the inner product of s and y is not sufficiently positive
    if s.T@y >= options.epsilon*(np.linalg.norm(s))*(np.linalg.norm(y)):
        Hy = H@y
        denominator = (y.T@H@y)[0][0]
        outer = s@s.T
        inner = (s.T@y)[0][0]
        H_new = H - ((Hy)@(Hy.T))/denominator + outer/inner
    else:
        H_new = H

    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, H_new

# subroutine for L-BFGS
def two_loop(grad: np.ndarray, y_stored: list, s_stored: list, options: Options):
    q = grad
    curr_m = len(y_stored)
    alpha_list = np.zeros(curr_m)
    rho_list = np.array([1/((y_stored[j]).T@s_stored[j])[0][0] for j in range(curr_m)])
    
    for i in range(curr_m):
        alpha_list[curr_m-i-1] = rho_list[curr_m-i-1]*(s_stored[curr_m-i-1].T)@q
        q = q-alpha_list[curr_m-i-1]*y_stored[curr_m-i-1]
    
    H_k0 = options.H0
    r = H_k0@q

    for i in range(curr_m):
        beta = rho_list[i]*((y_stored[i].T@r)[0][0])
        r = r + (alpha_list[i]-beta)*s_stored[i]
    return r

def L_BFGS_step(x: np.ndarray, f:np.ndarray, g:np.ndarray, 
                problem: Problem, method: Method, options: Options, 
                y_stored: list, s_stored: list):
    d = -two_loop(g, y_stored, s_stored, options)
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, f, g, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, f, g, d, problem, options)
    x_new = x + alpha*d
    g_new = problem.compute_g(x_new)
    s = x_new - x
    y = g_new - g
    s_stored.append(s)
    y_stored.append(y)

    # remove the oldest pair
    if len(s_stored) > options.m:
        s_stored.pop(0)
        y_stored.pop(0)

    f_new = problem.compute_f(x_new)

    return x_new, f_new, g_new    