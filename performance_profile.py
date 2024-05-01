from class_definition import *
import functions 
import optSolver
import numpy as np
import math

def performance_profile(method: Method, options: Options):
    # input: method, options
    # get the performance profile of the method on the given metric
    # metric: 'iteration', 'time', 'f_eval', 'g_eval'
    # output: 1 txt files
    np.random.seed(0)
    x0_1 = 20*np.random.rand(10,1)-10
    problem1 = Problem('P1_quad_10_10', x0=x0_1, 
                   compute_f=functions.quad_10_10_func, compute_g=functions.quad_10_10_grad, compute_H=functions.quad_10_10_Hess)
    problem2 = Problem('P2_quad_10_1000', x0=x0_1, 
                   compute_f=functions.quad_10_1000_func, compute_g=functions.quad_10_1000_grad, compute_H=functions.quad_10_1000_Hess)
    np.random.seed(0)
    x0_2 = 20*np.random.rand(1000,1)-10
    problem3 = Problem('P3_quad_1000_10', x0=x0_2, 
                   compute_f=functions.quad_1000_10_func, compute_g=functions.quad_1000_10_grad, compute_H=functions.quad_1000_10_Hess)
    problem4 = Problem('P4_quad_1000_1000', x0=x0_2, 
                   compute_f=functions.quad_1000_1000_func, compute_g=functions.quad_1000_1000_grad, compute_H=functions.quad_1000_1000_Hess)
    problem5 = Problem('P5_quartic_1', x0=np.array([math.cos(70),math.sin(70),math.cos(70),math.sin(70)]).reshape(-1,1), 
                   compute_f=functions.quartic_1_func, compute_g=functions.quartic_1_grad, compute_H=functions.quartic_1_Hess)
    problem6 = Problem('P6_quartic_2', x0=np.array([math.cos(70),math.sin(70),math.cos(70),math.sin(70)]).reshape(-1,1), 
                   compute_f=functions.quartic_2_func, compute_g=functions.quartic_2_grad, compute_H=functions.quartic_2_Hess)
    problem7 = Problem('P7_rosenbrock_2', x0=np.array([-1.2, 1.]).reshape(-1,1), 
                   compute_f=functions.rosen_2_func, compute_g=functions.rosen_2_grad, compute_H=functions.rosen_2_Hess)
    problem8 = Problem('P8_rosenbrock_100', x0=np.array([-1.2] + [1.]*99).reshape(-1,1),
                     compute_f=functions.Rosenbrock_100_func, compute_g=functions.Rosenbrock_100_grad, compute_H=functions.Rosenbrock_100_Hess)
    problem9 = Problem('P9_DataFit_2', x0=np.array([1.,1.]).reshape(-1,1),
                        compute_f=functions.DataFit_2_func, compute_g=functions.DataFit_2_grad, compute_H=functions.DataFit_2_Hess)
    problem10 = Problem('P10_Exponential_10', x0=np.array([1.] + [.0]*9).reshape(-1,1),
                        compute_f=functions.Exponential_func, compute_g=functions.Exponential_grad, compute_H=functions.Exponential_Hess)
    problem11 = Problem('P11_Exponential_100', x0=np.array([1.] + [.0]*99).reshape(-1,1),
                        compute_f=functions.Exponential_func, compute_g=functions.Exponential_grad, compute_H=functions.Exponential_Hess)
    problem12 = Problem('P12_Genhumps_5', x0=np.array([-506.2]+[506.2]*4).reshape(-1,1),
                        compute_f=functions.genhumps_5_func, compute_g=functions.genhumps_5_grad, compute_H=functions.genhumps_5_Hess)
        
    problems = [problem1, problem2, problem3, problem4, problem5, problem6, problem7, problem8, problem9, problem10, problem11, problem12]
    
    # restore the performance profile
    f_list = []
    iteration_list = []
    f_eval_list = []
    g_eval_list = []
    time_list = []
    for problem in problems:
        _, f, K, cpu_times = optSolver.optSolver(problem, method, options)
        iteration_list.append(K)
        f_list.append(f)
        time_list.append(cpu_times[-1])
        f_eval_list.append(problem.f_eval)
        g_eval_list.append(problem.g_eval)

    profile = open(f'./profiles/profile_{method.name}_{method.step_type}.txt', 'w')
    #print('Writing profile')
    for i in range(len(problems)):
        profile.write(problems[i].name+'\t')
        profile.write(str(iteration_list[i])+'\t')
        profile.write(str(f_eval_list[i])+'\t')
        profile.write(str(g_eval_list[i])+'\t')
        profile.write(str(time_list[i])+'\t')
        profile.write(str(f_list[i])+'\n')
    profile.close()

    return