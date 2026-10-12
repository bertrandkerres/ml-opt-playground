# This is an example of using the package DiffEqParamEstim
# to estimate the parameters of an ODE system
# The data and ODE form are adapted from https://ctsm.info/building2.pdf

using ComponentArrays
using DataFrames
using Plots
using DataInterpolations
using UnPack
using Statistics

using DifferentialEquations
using DiffEqParamEstim
import SciMLBase

using Optimization
using Optim
using LineSearches
using OptimizationOptimJL

# Loads the data and dynamics functions
include("ctsm_setup_prob2.jl")

function noise_r2c2!(du, u, p, t)
    @unpack p11, p22 = p
    du[1] = exp(p11)
    du[2] = exp(p22)
end

tspan_train = (0.0, 100.0)

dynamics = dynamics_r2c2!
params = rc2_params

prob = ODEProblem(dynamics, params.u0, tspan_train, params.p0)
sol = solve(prob, Tsit5())
plot(sol)
plot!(df.t, df.yTi)
ylims!(0, 25)


obj = build_loss_objective(
    prob, Tsit5(), loss_fun, Optimization.AutoForwardDiff(),
    maxiters = 100000, verbose = false
)

# opt = Optim.Adam()
opt = Optim.BFGS(linesearch = BackTracking())

optprob = Optimization.OptimizationProblem(obj, params.p0)
popt_sol = solve(optprob, opt)

p_opt = popt_sol.u

tspan_opt = (0.0, maximum(df.t))
prob2 = ODEProblem(dynamics, params.u0, tspan_opt, p_opt)
sol2 = solve(prob2, Tsit5())
plot(sol2)
plot!(df.t, df.yTi)
vline!([tspan_train[2]], color = :black, label = "Training cut-off")



#=
ERROR: InexactError: Int64(NaN)
Probably if SciML DE solver returns NaN somewhere

import NLopt
using OptimizationNLopt

nlopt = NLopt.Opt(:LN_COBYLA, length(p0))
NLopt.lower_bounds!(nlopt, fill(0.1, length(p0)))
NLopt.upper_bounds!(nlopt, [20.0, 20.0, 50.0, 50.0, 40.0])
NLopt.maxtime!(nlopt, 60.0)
nlobj = build_loss_objective(
    prob, Tsit5(), loss_fun,
    maxiters = 10000, verbose = false
)

nloptprob = Optimization.OptimizationProblem(nlobj, p0)
res = solve(nloptprob, nlopt)
=#

