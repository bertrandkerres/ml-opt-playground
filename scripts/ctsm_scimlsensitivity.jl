import OrdinaryDiffEq as ODE
import Optimization as OPT
import OptimizationPolyalgorithms as OPA
import SciMLSensitivity as SMS
import SciMLBase
import Zygote
import Plots

import Optim
import LineSearches

# Loads the data and dynamics functions
include("ctsm_setup_prob2.jl")

tspan = (0.0, 100.0)
tsteps = df.t[df.t .<= 100.0]
y_true = df.yTi[df.t .<= 100.0]

dynamics = dynamics_r2c2!
params = rc2_params

# Setup the ODE problem, then solve
prob = ODE.ODEProblem(dynamics, params.u0, tspan, params.p0)
sol = ODE.solve(prob, ODE.Tsit5())

Plots.plot(sol)
Plots.plot!(df.t, df.yTi, labels="true y")

function _loss(prob, y_true, p; tsteps = tsteps)
    tmp_prob = ODE.remake(prob, p = p)
    tmp_sol = ODE.solve(tmp_prob, ODE.Tsit5(), p = p, saveat = tsteps)
    
    # https://docs.sciml.ai/SciMLSensitivity/dev/tutorials/training_tips/divergence/
    # or: if tmp_sol.retcode == SciMLBase.ReturnCode.Success works for e.g. ForwardDiff
    if size(tmp_sol, 2) == length(y_true)
        y_hat = reduce(hcat, tmp_sol.u)[1, :]
        loss = sum(abs2, y_hat .- y_true)
    else
        loss = Inf
    end
    return loss
end

loss(p) = _loss(prob, y_true, p)

adtype = OPT.AutoForwardDiff()
optf = OPT.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = OPT.OptimizationProblem(optf, params.p0)

result_ode = OPT.solve(
    optprob,
    Optim.BFGS(linesearch = LineSearches.BackTracking()), # or OPA.PolyOpt()
    callback = (s, l) -> false,
    maxiters = 100
)

remade_solution = ODE.solve(
    ODE.remake(prob, p = result_ode.u), ODE.Tsit5(),
    saveat = tsteps
)

Plots.plot(remade_solution)
Plots.plot!(y_true, labels="y_true")