# IOE_project

Person 1: 
1. GradientDescent, with backtracking line search （def gdStep:）
2. GradientDescentW, with Wolfe line search （def gdWstep:）
3. Newton, (modified Newton) with backtracking line search （def newtonStep:   / （def modifiedNewtonStep:））
4. NewtonW, (modified Newton) with Wolfe line search （def newtonWStep:  / def modifiedNewtonWStep )

Person 2:
5. TRNewtonCG, trust region Newton with CG subproblem solver (def )
6. TRSR1CG, SR1 quasi-Newton with CG subproblem solver

person 3:
7. BFGS, BFGS quasi-Newton with backtracking line search
8. BFGSW, BFGS quasi-Newton with Wolfe line search
9. DFP, DFP quasi-Newton with backtracking line search
10. DFPW, DFP quasi-Newton with Wolfe line search

4.8-4.14
1. all functions in function.py 
(2,3), (4,5), (6,8) （周三11：59前)
x0 = np.array([0,0,0,1]).reshape(-1, 1)
如果需要输入response，也确保是column vector: y = np.array([1,1,-1,-1]).reshape(-1, 1)
3. Implement all methods on each 12 testing problem
algotithm.py 格式：见第一部分，不要互相调用
optSolver.py 格式：确保后缀明确alpha_lists_1 alpha_lists_3
