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
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import scipy.sparse as sparse

# Subroutine for backtracking line search and Wolfe line search
def backtracking(x: np.ndarray, d: np.ndarray, problem: Problem, options: Options):
    alpha = options.alpha_bar
    fx = problem.compute_f(x)
    gx = problem.compute_g(x)
    while problem.compute_f(x+alpha*d) > fx+(options.c1)*alpha*(gx.T)@d:
        alpha = alpha*options.tau
    return alpha

def wolfe_line_search(x: np.ndarray, d: np.ndarray, problem: Problem, options: Options):
    alpha = options.alpha_bar
    i = 0
    fx = problem.compute_f(x)
    gx = problem.compute_g(x)
    alpha_low = options.alpha_low
    alpha_high = options.alpha_high
    c = options.c_wolfe
    while i < options.max_iterations:
        if problem.compute_f(x + alpha * d) < fx + options.c1 * alpha * gx.T @ d:
            if problem.compute_g(x + alpha * d).T @ d > options.c2_curve * gx.T @ d:
                break
        if problem.compute_f(x + alpha * d) < fx + options.c1 * alpha * gx.T @ d:
            alpha_low = alpha
        else:
            alpha_high = alpha
        alpha = c*alpha_low + (1-c)*alpha_high
        i += 1
    return alpha

# def backtracking(x, d, problem, options):
#     alpha = options.alpha_bar
#     fx = problem.compute_f(x)
#     gx = problem.compute_g(x)

#     # 确保d的形状正确
#     print("Shape of d in backtracking:", d.shape)

#     dot_product = np.dot(gx.T, d)
#     if isinstance(dot_product, np.ndarray):
#         dot_product = dot_product.item()

#     # 添加调试输出以查看比较的值
#     while True:
#         x_new = x + alpha * d
#         fx_new = problem.compute_f(x_new)
#         if fx_new > fx + options.c1 * alpha * dot_product:
#             alpha *= options.tau
#             print("Updated alpha:", alpha)  # 打印更新后的alpha值
#         else:
#             break

#     return alpha

# def wolfe_line_search(x: np.ndarray, d: np.ndarray, problem: Problem, options: Options):
#     alpha = options.alpha_bar
#     fx = problem.compute_f(x)
#     gx = problem.compute_g(x)

#     # Debug prints
#     print("Initial x:", x)
#     print("Initial alpha:", alpha)
#     print("Initial fx:", fx)
#     print("Initial gx:", gx)

#     if gx.ndim == 1:
#         gx = gx[:, np.newaxis]

#     dot_product = np.dot(gx.T, d)
#     if isinstance(dot_product, np.ndarray):
#         dot_product = dot_product.item()

#     # Debug prints
#     print("Dot product (gx.T @ d):", dot_product)

#     while True:
#         x_new = x + alpha * d
#         fx_new = problem.compute_f(x_new)
#         gx_new = problem.compute_g(x_new)

#         if gx_new.ndim == 1:
#             gx_new = gx_new[:, np.newaxis]

#         grad_new_dot_d = np.dot(gx_new.T, d)
#         if isinstance(grad_new_dot_d, np.ndarray):
#             grad_new_dot_d = grad_new_dot_d.item()

#         # Debug prints
#         print("New x:", x_new)
#         print("New fx:", fx_new)
#         print("Grad_new_dot_d:", grad_new_dot_d)

#         if fx_new > fx + options.c1 * alpha * dot_product or grad_new_dot_d < options.c2_curve * dot_product:
#             alpha *= options.tau
#             # Debug prints
#             print("Updated alpha:", alpha)
#         else:
#             break

#     return alpha


# #1,0 GradientDescent, with backtracking line search
# def backtracking_line_search(x, f, g, d, problem, c1, tau):
#     alpha = 1
#     while problem.compute_f(x + alpha * d) > f + c1 * alpha * np.dot(g, d):
#         alpha *= tau
#     return alpha

# def wolfe_search(x, f, g, d, problem, c1, c2, tau):
#     alpha = 1  
#     while ((problem.compute_f(x + alpha * d) > f + c1 * alpha * np.dot(g, d)) or
#            (np.dot(problem.compute_g(x + alpha * d), d) < c2 * np.dot(g, d))):
#         alpha *= tau
#     return alpha


# gradient descent step
def gradient_descent_step(x, f, g, problem, method, options):
    d = -g
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, d, problem, options)    
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
    # def newton_step(x, f, g, H, problem, method, options):
    # # Check if H is a sparse matrix and convert it to a dense array if it is
    if sparse.issparse(H):
        H_dense = H.toarray()
    else:
        H_dense = H  # Use the dense matrix directly


    d = -np.linalg.solve(H_dense, g)

    # # Convert H from sparse to dense if it's in sparse format
    # if isinstance(H, csr_matrix):
    #     H = H.toarray()
    #     print("Converted Hessian from sparse to dense.")
    
    # print("H matrix (dense) content:\n", H)

    # print(g)
    # d = -np.linalg.solve(H.toarray(), g) zhaohaoyu
    # try:
    #     d = -np.linalg.solve(H, g)
    # except np.linalg.LinAlgError:
    #     # Handle cases where the Hessian is singular or nearly singular
    #     d = -np.linalg.pinv(H) @ g
    #     print("Used pseudo-inverse due to singular Hessian.")
    # # try:

    #     print("Solved for direction d successfully.")
    # except np.linalg.LinAlgError as e:
    #     print("Failed to solve Hd = -g with np.linalg.solve due to:", str(e))
    #     d = np.linalg.pinv(H) @ -g
    #     print("Using pseudo-inverse for computing direction.")

    # print("Shape of d after solving or pinv:", d.shape)
    # print("Content of d:\n", d)

    # # Assuming backtracking line search for simplicity
    if method.step_type == 'backtracking':
        alpha = backtracking(x, d, problem, options)
    elif method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, d, problem, options)
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

def Modified_NewtonStep(x, problem, method, options, g):
    #g = problem.compute_g(x)
    H = problem.compute_H(x)

    L, l_computeEta = compute_eta(A=H, beta=options.beta, max_iter=options.max_iterations)
    d = -np.linalg.solve(L@L.T, g) # solve the linear system to get direction  

    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, d, problem, options)   
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
    #f = problem.compute_f(x)
    #g = problem.compute_g(x)
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


def BFGS_step(x: np.ndarray, problem: Problem, method: Method, options: Options, H: np.ndarray, gx: np.ndarray):
    #gx = problem.compute_g(x)
    n = x.shape[0]

    d = -H@gx
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, d, problem, options) 
    x_new = x + alpha*d
    s = x_new - x
    y = problem.compute_g(x_new) - gx

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


# def BFGSW_step(x: np.ndarray, problem: Problem, method: Method, options: Options, H: np.ndarray, gx: np.ndarray):
#     #gx = problem.compute_g(x)
#     n = x.shape[0]

#     d = -H@gx
#     if method.step_type == 'wolfe':
#         alpha = wolfe_line_search(x, d, problem, options)
#     elif method.step_type == 'backtracking':
#         alpha = backtracking(x, d, problem, options) 
#     x_new = x + alpha*d
#     s = x_new - x
#     y = problem.compute_g(x_new) - gx

#     # Skip if the inner product of s and y is not sufficiently positive
#     if s.T@y >= options.epsilon*(np.linalg.norm(s))*(np.linalg.norm(y)):
#         rho = 1/((s.T@y)[0][0])
#         V = np.eye(n) -rho*y@(s.T)
#         H_new = (V.T)@H@V + rho*s@(s.T)
#     else:
#         H_new = H

#     f_new = problem.compute_f(x_new)
#     g_new = problem.compute_g(x_new)

#     return x_new, f_new, g_new, H_new

def DFP_step(x: np.ndarray, problem: Problem, method: Method, options: Options, H: np.ndarray, gx: np.ndarray):
    #gx = problem.compute_g(x)
    #n = x.shape[0]

    d = -H@gx
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, d, problem, options)
    x_new = x + alpha*d
    s = x_new - x
    y = problem.compute_g(x_new) - gx

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

# def DFPW_step(x: np.ndarray, problem: Problem, options: Options, H: np.ndarray, gx: np.ndarray):
#     #gx = problem.compute_g(x)
#     #n = x.shape[0]

#     d = -H@gx
#     alpha = wolfe_line_search(x, d, problem, options)
#     x_new = x + alpha*d
#     s = x_new - x
#     y = problem.compute_g(x_new) - gx

#     # Skip if the inner product of s and y is not sufficiently positive
#     if s.T@y >= options.epsilon*(np.linalg.norm(s))*(np.linalg.norm(y)):
#         Hy = H@y
#         denominator = (y.T@H@y)[0][0]
#         outer = s@s.T
#         inner = (s.T@y)[0][0]
#         H_new = H - ((Hy)@(Hy.T))/denominator + outer/inner
#     else:
#         H_new = H

#     f_new = problem.compute_f(x_new)
#     g_new = problem.compute_g(x_new)

#     return x_new, f_new, g_new, H_new

def two_loop(grad: np.ndarray, y_stored: list, s_stored: list, options: Options):
    q = grad
    curr_m = len(y_stored)
    alpha_list = np.zeros(curr_m)
    rho_list = np.array([1/((y_stored[j]).T@s_stored[j])[0][0] for j in range(curr_m)])
    
    for i in range(curr_m):
        alpha_list[curr_m-i-1] = rho_list[curr_m-i-1]*(s_stored[curr_m-i-1].T)@q
        q = q-alpha_list[curr_m-i-1]*y_stored[curr_m-i-1]
    
    # H_k0 = (s_stored[-1])@(y_stored[-1].T)/((y_stored[-1].T@y_stored[-1])[0][0])
    H_k0 = options.H0
    r = H_k0@q

    for i in range(curr_m):
        beta = rho_list[i]*((y_stored[i].T@r)[0][0])
        r = r + (alpha_list[i]-beta)*s_stored[i]
    return r

def L_BFGS_step(x: np.ndarray, problem: Problem, method: Method, options: Options, y_stored: list, s_stored: list, gx: np.ndarray):
    #gx = problem.compute_g(x)
    d = -two_loop(gx, y_stored, s_stored, options)
    if method.step_type == 'wolfe':
        alpha = wolfe_line_search(x, d, problem, options)
    elif method.step_type == 'backtracking':
        alpha = backtracking(x, d, problem, options)
    x_new = x + alpha*d
    g_new = problem.compute_g(x_new)
    s = x_new - x
    y = g_new - gx
    s_stored.append(s)
    y_stored.append(y)

    if len(s_stored) > options.m:
        s_stored.pop(0)
        y_stored.pop(0)

    f_new = problem.compute_f(x_new)

    return x_new, f_new, g_new    