#= This is an example of a quadratic program where the Hessian is not neg. semidefinite
for a maximization problem.
It shows the difference between the solvers.
The example is taken from https://web.mit.edu/15.053/www/AMP-Chapter-13.pdf, p.46

HiGHS fails, even though the is has linear constraints and quadratic objective,
i.e. all shapes are supported by the solver and JuMP/MOI does not complain.
```
ERROR:   Hessian has 2 diagonal entries in (0, 0.36] so is not negative semidefinite for maximization
ERROR:   Cannot solve non-convex QP problems with HiGHS
```

Ipopt solves it, but does not find the global optimum:
```
Total seconds in IPOPT = 0.004
EXIT: Optimal Solution Found.
```
Values for x = [0.1, 0.8333, 1.066, 0, 0]
Objective is 0.2006

SCS solves it with global optimum:
```
status:  solved
timings: total: 3.89e-03s = setup: 5.20e-05s + solve: 3.84e-03s
         lin-sys: 3.03e-05s, cones: 1.30e-05s, accel: 9.58e-04s
------------------------------------------------------------------
objective = -0.210843
```
Values for x = [1, 0.333, 0.666, 0, 0]
Objective value is 0.2108
=#

import HiGHS, Ipopt, SCS, SCIP
using JuMP
using LinearAlgebra: dot, Diagonal, isposdef, eigvals

mdl = Model(SCIP.Optimizer)

maturities = [6, 10, 1, 4, 3]
lin_returns = [0.03, 0.02, exp(0.05) - 1, 0.06, 0.055]

@variable(mdl, x[1:5])
@constraint(mdl, non_neg[i=3:5], x[i] >= 0)
@constraint(mdl, x1_lower, x[1] >= 0.1)
@constraint(mdl, x1_upper, x[1] <= 1)
@constraint(mdl, x2_lower, x[2] >= 0.2)
@constraint(mdl, x2_upper, x[2] <= 1)

@constraint(mdl, budget, sum(x) == 2)
@constraint(mdl, maturity, dot(maturities, x) <= 10)

@objective(mdl, Max, dot(lin_returns, x) + 0.12 * x[1]^2 + 0.18 * x[2]^2)

optimize!(mdl)

Q = 2 .* Diagonal([0.12, 0.18, 0.0, 0.0, 0.0])
eigvals(Q)