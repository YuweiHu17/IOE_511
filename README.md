# Unconstrained optimization methods by Team NaN: Haoyu Zhao, Ke Hu, Yuwei Hu

## input:

problem: a Problem class object

method: a Method class object

options: a Options class object

The user can define a problem, a method, and options for the optimization solver.

starting point x0, objective function, and gradient function are required for the problem.

x0 must be a column vector.

step_type is required for the method.

##output for optSolver_NaN:

x, f

## output for optSolver:

x, f, k, cpu_times, f_values, norm_g_values

Users can choose optSolver_NaN or optSoler as their needs.

Please see the run_rosen.ipynb file for examples.
