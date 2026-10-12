using Turing
using DifferentialEquations
# Load StatsPlots for visualizations and diagnostics.
using StatsPlots
using LinearAlgebra
using Distributions
using ComponentArrays

import Plots
import StatsBase

using ControlSystemIdentification: iddata, newpem

include("boptest_data.jl")

cols = colnames[TESTCASE]

dt_h = dt / 3600

uw = Matrix(df[!, ["reaQHea_y"; cols["w"]]])'
uw[1, :] ./= 1000
uw[2, :] .-= 273.15
uw[3, :] ./= 1000
sysid_data = detrend(iddata(
    Matrix(df[!, cols["y"]])',
    uw,
    dt_h
))

y_true = Matrix(df[!, cols["y"]])' .- 273.15

@model function fit_r3c3(data, prob)
    σ ~ InverseGamma(2, 3)
    Ci ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=100.0)
    Ce ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=100.0)
    Ch ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=100.0)
    Rie ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=500.0)
    Rea ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=500.0)
    Rih ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=500.0)
    Aw ~ truncated(Normal(20.0, 5.0); lower=5.0, upper=500.0)

    p = ComponentVector((;Ci, Ce, Ch, Rie, Rea, Rih, Aw))
    predicted = solve(prob, Tsit5(); p=p, saveat=tsteps, save_idxs=[1], abstol=1e-6, reltol=1e-6)

    for i in eachindex(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

t_h = df.time ./ 3600 .- 0.25
tspan = (t_h[1], t_h[2000])

x0 = [y_true[1], y_true[1] - 10, y_true[1]]
interp = (
    Ps = CubicSpline(uw[3, :], t_h),
    Ta = CubicSpline(uw[2, :], t_h),
    Ph = CubicSpline(df.reaQHea_y ./ 1000, t_h),
)
dynamics!(dx, x, p, t) = _dynamics_r3c3!(dx, x, p, t, interp)

prob_rc3 = ODEProblem(dynamics!, x0, tspan, rc3_params.p0)
sol_rc3 = solve(prob_rc3, Tsit5(); saveat = t_h[t_h .< tspan[2]])
mdl = fit_r3c3(y_true, prob_rc3)

p_mle_est = maximum_likelihood(mdl)
p_mle = ComponentVector(;( Symbol(k) => v for (k, v) in p_mle_est.params)...)

prob_eval_mle = ODEProblem(dynamics!, x0, tspan, p_mle)
sol_eval_mle = solve(prob_eval_mle, Tsit5())
plot(sol_eval_mle)
plot!(t_h, y_true')
ylims!(0, 25)

p_map_est = maximum_a_posteriori(mdl)
p_map = ComponentVector(;( Symbol(k) => v for (k, v) in p_map_est.params)...)
prob_eval_map = ODEProblem(dynamics!, x0, tspan, p_map)
sol_eval_map = solve(prob_eval_map, Tsit5())
plot(sol_eval_map)
plot!(t_h, y_true')
ylims!(0, 25)
