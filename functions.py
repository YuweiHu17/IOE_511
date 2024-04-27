# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi
# Team NaN: Ke Hu & Yuwei Hu & Haoyu Zhao

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.io
# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1)(2)(3)(4) Quadractic function
# (5)(6) Quartic function
# (7)(8) Rosenbrock function 
# (9) Data fit
# (10)(11) Exponential

 
# Problem Number: 1
# Problem Name: quad_10_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 10


# all inputes' shape is (n,1) column vector

# function that computes the function value of the quad_10_10 function

def quad_10_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0,0]

def quad_10_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']
    
    return Q@x + q   
    

def quad_10_10_Hess(x):
    # set raondom seed
    #np.random.seed(0)
    # Generate random data
    #q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_10_Q.mat')
    Q = mat['Q']
    
    return Q

# Problem Number: 2
# Problem Name: quad_10_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 10; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_1000 function

def quad_10_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))

    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0,0]

def quad_10_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(10,1))
    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q']

    return Q@x + q

def quad_10_1000_Hess(x):
    mat = scipy.io.loadmat('quad_10_1000_Q.mat')
    Q = mat['Q']
    
    return Q

# Problem Number: 3
# Problem Name: quad_1000_10
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 10

# function that computes the function value of the quad_1000_10 function

def quad_1000_10_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0,0]

def quad_1000_10_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = mat['Q']

    return Q@x + q

def quad_1000_10_Hess(x):
    mat = scipy.io.loadmat('quad_1000_10_Q.mat')
    Q = mat['Q']
    
    return Q

# Problem Number: 4
# Problem Name: quad_1000_1000
# Problem Description: A randomly generated convex quadratic function; the 
#                      random seed is set so that the results are 
#                      reproducable. Dimension n = 1000; Condition number
#                      kappa = 1000

# function that computes the function value of the quad_10_10 function

def quad_1000_1000_func(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = mat['Q']
    
    # compute function value
    return (1/2*x.T@Q@x + q.T@x)[0, 0]

def quad_1000_1000_grad(x):
    # set raondom seed
    np.random.seed(0)
    # Generate random data
    q = np.random.normal(size=(1000,1))
    
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = mat['Q']

    return Q@x + q

def quad_1000_1000_Hess(x):
    mat = scipy.io.loadmat('quad_1000_1000_Q.mat')
    Q = mat['Q']
    return Q


# Problem Number: 5
# Problem Name: quartic_1
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_1 function

def quartic_1_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4
    
    return (1/2*(x.T @x) + sigma/4*(x.T@Q@x)**2)[0,0]

def quartic_1_grad(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4

    return x + sigma*(x.T@Q@x)*Q@x

def quartic_1_Hess(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e-4

    return np.eye(4) + 2*sigma*(Q@x@x.T@Q) + sigma*(x.T@Q@x)*Q

# Problem Number: 6
# Problem Name: quartic_2
# Problem Description: A quartic function. Dimension n = 4

# function that computes the function value of the quartic_2 function

def quartic_2_func(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4
    
    return (1/2*(x.T@x) + sigma/4*(x.T@Q@x)**2)[0,0]

def quartic_2_grad(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4

    return x + sigma*(x.T@Q@x)*Q@x

def quartic_2_Hess(x):
    Q = np.array([[5,1,0,0.5],
     [1,4,0.5,0],
     [0,0.5,3,0],
     [0.5,0,0,2]])
    sigma = 1e4

    return np.eye(4) + 2*sigma*(Q@x@x.T@Q) + sigma*(x.T@Q@x)*Q


# Problem Number: 7
# Problem Name: Rosenbrock_2
# Problem Description: A Rosenbrock fucntion, with d = 2

def rosen_2_func(x):
    return ((1-x[0])**2 + 100*(x[1] - x[0]**2)**2)[0]

def rosen_2_grad(x):
    grad_1 = (400*x[0]**3 + (2-400*x[1])*x[0] - 2)[0]
    grad_2 = (200*(x[1] - x[0]**2))[0]
    return np.array([[grad_1], [grad_2]])

def rosen_2_Hess(x):
    hess_11 = (1200*x[0]**2 + 2 - 400*x[1])[0]
    hess_12 = (-400*x[0])[0]
    hess_22 = 200
    return np.array([[hess_11, hess_12], [hess_12, hess_22]])

#TO-DO
# Problem Number: 8
# Problem Name: Rosenbrock_100
# Problem Description: A Rosenbrock fucntion, with d = 100

def Rosenbrock_100_func(x: np.ndarray):
    return np.sum((1 - x[:-1])**2 + 100 * (x[1:] - x[:-1]**2)**2)

# def Rosenbrock_100_grad(x: np.ndarray):
#     assert x.shape[1] == 1
#     n = x.shape[0]
#     grad = np.zeros((n, 1))

#     grad[0] = 2*(x[0]-1) + 400*x[0]*(x[0]**2-x[1])
#     grad[1:n-1] = 200*(x[1:n-1]-x[:n-2]**2) + 2*(x[1:n-1]-1) + 400*x[1:n-1]*(x[1:n-1]**2-x[2:n])
#     grad[n-1] = 200*(x[n-1]-x[n-2]**2) + 2*(x[n-1]-1)

#     return grad

def Rosenbrock_100_grad(x):
    assert x.shape[1] == 1
    n = x.shape[0]
    grad = np.zeros((n, 1))

    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1:-1] = 200 * (x[1:-1] - x[:-2]**2) - 2 * (1 - x[1:-1]) - 400 * x[1:-1] * (x[2:] - x[1:-1]**2)
    grad[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]**2)
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    return grad

def Rosenbrock_100_Hess(x):
    x = x.flatten()
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros(len(x), dtype=x.dtype)
    diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H 

# Problem Number: 9
# Problem Name: DataFit_2
# Problem Description: Dimension is 2

def DataFit_2_func(x):
    y = np.array([1.5,2.25,2.625])
    i = np.array([1,2,3])
    return np.sum((y - x[0]*(1-x[1]**i))**2)

def DataFit_2_grad(x):
    y = np.array([1.5,2.25,2.625])
    i = np.array([1,2,3])
    t = 1-x[1]**i
    g1 = np.sum(2 * t**2 * x[0] - 2 * t * y)
    g2 = np.sum(2 * (y - x[0]*t) * x[0] * i * x[1]**(i-1))
    return np.array([[g1],[g2]])

def DataFit_2_Hess(x):
    y = np.array([1.5,2.25,2.625])
    i = np.array([1,2,3])
    t = 1-x[1]**i
    h11 = np.sum(2 * t**2)
    h12 = np.sum(-4 * x[0] * i * x[1]**(i-1) + 4 * x[0] * i * x[1]**(2*i-1) + 2 * y * i * x[1]**(i-1))
    h22 = np.sum(2*y*x[0]*i*(i-1)*x[1]**(i-2) - 2*x[0]**2*i*((i-1)*x[1]**(i-2)-(2*i-1)*x[1]**(2*i-2)))
    return np.array([[h11,h12], [h12,h22]])




# Functions that computes the value/Gradient/Hessian of Function3 in HW3
# Problem Number: 10 & 11
# Problem Name: Exponential_10 & Exponential_100
# Problem Description: Dimension is 10 or 100, 
# which depends on the dimension of starting point

def Exponential_func(x):
    z1 = x[0][0]
    return (np.exp(z1)-1)/(np.exp(z1)+1) + 0.1*np.exp(-z1) + np.sum((x[1:]-1)**4)

def Exponential_grad(x):
    n = len(x)
    z1 = x[0][0]
    gi = np.zeros((n,1))
    gi[0] = 2*np.exp(z1)/(np.exp(z1)+1)**2 - 0.1*np.exp(-z1)
    gi[1:] = 4*(np.array(x[1:]) - 1)**3
    return gi

def Exponential_Hess(x):
    #n = len(x)
    t = np.exp(x[0][0])
    hii = 12 * (x.flatten() - 1)**2  # diagonal entris
    H = np.diag(hii)  
    H[0,0] = 2*t*(1-t)/(t+1)**3 + 0.1/t # update H11
    return H

# Problem Number: 12
# Problem Name: genhumps_5
# Problem Description: A multi-dimensional problem with a lot of humps.
#                      This problem is from the well-known CUTEr test set.

# function that computes the function value of the genhumps_5 function


def genhumps_5_func(x):
    f = 0
    for i in range(4):
        f = f + np.sin(2*x[i])**2*np.sin(2*x[i+1])**2 + 0.05*(x[i]**2 + x[i+1]**2)
    return f[0]

# function that computes the gradient of the genhumps_5 function

def genhumps_5_grad(x):
    g = [4*np.sin(2*x[0])*np.cos(2*x[0])* np.sin(2*x[1])**2                  + 0.1*x[0],
         4*np.sin(2*x[1])*np.cos(2*x[1])*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2) + 0.2*x[1],
         4*np.sin(2*x[2])*np.cos(2*x[2])*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2) + 0.2*x[2],
         4*np.sin(2*x[3])*np.cos(2*x[3])*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2) + 0.2*x[3],
         4*np.sin(2*x[4])*np.cos(2*x[4])* np.sin(2*x[3])**2                  + 0.1*x[4]]
    
    return np.array(g).reshape(-1,1)

# function that computes the Hessian of the genhumps_5 function
def genhumps_5_Hess(x) :
    H = np.zeros((5,5))
    x = x.flatten()
    H[0,0] =  8* np.sin(2*x[1])**2*(np.cos(2*x[0])**2 - np.sin(2*x[0])**2) + 0.1
    H[0,1] = 16* np.sin(2*x[0])*np.cos(2*x[0])*np.sin(2*x[1])*np.cos(2*x[1])
    H[1,1] =  8*(np.sin(2*x[0])**2 + np.sin(2*x[2])**2)*(np.cos(2*x[1])**2 - np.sin(2*x[1])**2) + 0.2
    H[1,2] = 16* np.sin(2*x[1])*np.cos(2*x[1])*np.sin(2*x[2])*np.cos(2*x[2])
    H[2,2] =  8*(np.sin(2*x[1])**2 + np.sin(2*x[3])**2)*(np.cos(2*x[2])**2 - np.sin(2*x[2])**2) + 0.2
    H[2,3] = 16* np.sin(2*x[2])*np.cos(2*x[2])*np.sin(2*x[3])*np.cos(2*x[3])
    H[3,3] =  8*(np.sin(2*x[2])**2 + np.sin(2*x[4])**2)*(np.cos(2*x[3])**2 - np.sin(2*x[3])**2) + 0.2
    H[3,4] = 16* np.sin(2*x[3])*np.cos(2*x[3])*np.sin(2*x[4])*np.cos(2*x[4])
    H[4,4] =  8* np.sin(2*x[3])**2*(np.cos(2*x[4])**2 - np.sin(2*x[4])**2) + 0.1
    H[1,0] = H[0,1]
    H[2,1] = H[1,2]
    H[3,2] = H[2,3]
    H[4,3] = H[3,4]
    return H
