using Turing
using DifferentialEquations
# Load StatsPlots for visualizations and diagnostics.
using StatsPlots
using LinearAlgebra
using Distributions
using ComponentArrays

import Plots
import StatsBase

include("ctsm_setup_prob2.jl")

df_resampled = df[1:6:end, :]

y_true = df_resampled.yTi'
tspan = (df_resampled.t[1], df_resampled.t[end])
tsteps = df_resampled.t

prob = ODEProblem(dynamics_r2c2!, rc2_params.u0, tspan, rc2_params.p0)
sol = solve(prob, Tsit5(), saveat=tsteps, save_idxs=[1])
Plots.plot(sol)
Plots.plot!(y_true')
y_sim = Array(sol)

@model function fit_r2c2(data, prob)
    Ci ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=20.0)
    Ce ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=20.0)
    Rie ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=50.0)
    Rea ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=50.0)
    Aw ~ truncated(Normal(20.0, 5.0); lower=5.0, upper=50.0)

    p = ComponentVector((;Ci, Ce, Rie, Rea, Aw))
    predicted = solve(prob, Tsit5(); p=p, saveat=tsteps, save_idxs=[1], abstol=1e-6, reltol=1e-6)

    for i in eachindex(predicted)
        data[:, i] ~ MvNormal(predicted[i], 0.27 * I)
    end

    return nothing
end

model_r2c2 = fit_r2c2(y_true, prob)

#=
This takes forever, need to check priors
chain = sample(model_r2c2, NUTS(), MCMCSerial(), 1000, 3; progress=true)

plot(chain)
=#

p_ml_est = maximum_likelihood(model_r2c2)
StatsBase.coeftable(p_ml_est)

p_mle = ComponentVector(;( Symbol(k) => v for (k, v) in p_ml_est.params)...)

dynamics = dynamics_r2c2!
params = rc2_params

tspan_train = (0.0, 100.0)
tspan_eval = (0.0, maximum(df.t))
prob = ODEProblem(dynamics, params.u0, tspan_eval, p_mle)
sol = solve(prob, Tsit5())
plot(sol)
plot!(df.t, df.yTi)
ylims!(0, 25)


prob_rc3 = ODEProblem(dynamics_r3c3!, rc3_params.u0, tspan, rc3_params.p0)
sol_rc3 = solve(prob_rc3, Tsit5(), saveat=tsteps, save_idxs=[1])
Plots.plot(sol_rc3)
Plots.plot!(y_true')
y_sim = Array(sol_rc3)

@model function fit_r3c3(data, prob)
    Ci ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=20.0)
    Ce ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=20.0)
    Ch ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=20.0)
    Rie ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=50.0)
    Rea ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=50.0)
    Rih ~ truncated(Normal(5.0, 2.0); lower=0.1, upper=50.0)
    Aw ~ truncated(Normal(20.0, 5.0); lower=5.0, upper=50.0)

    p = ComponentVector((;Ci, Ce, Ch, Rie, Rea, Rih, Aw))
    predicted = solve(prob, Tsit5(); p=p, saveat=tsteps, save_idxs=[1], abstol=1e-6, reltol=1e-6)

    for i in eachindex(predicted)
        data[:, i] ~ MvNormal(predicted[i], 0.27 * I)
    end

    return nothing
end

model_r3c3 = fit_r3c3(y_true, prob_rc3)

p_mle_est = maximum_likelihood(model_r3c3)
p_mle = ComponentVector(;( Symbol(k) => v for (k, v) in p_mle_est.params)...)

prob_eval = ODEProblem(dynamics_r3c3!, rc3_params.u0, tspan_eval, p_mle)
sol_eval = solve(prob_eval, Tsit5())
plot(sol_eval)
plot!(df.t, df.yTi)
ylims!(0, 25)


u_eval = Array(sol_eval(df.t))
y_err = df.yTi - u_eval[1, :]

lags = 1:min(length(y_err) - 1, round(Int, 10*log10(length(y_err))))
err_acf = StatsBase.autocor(y_err, lags)
err_pacf = StatsBase.pacf(y_err, lags)
plot(lags, [err_acf err_pacf], labels=["acf" "pacf"], xlabel="Lags [n]")


using ControlSystems: lsim, ss
ufun(_x, t) = [
    interpolations.Ps(t),
    interpolations.Ta(t),
    interpolations.Ph(t)
]

sys = ss(rc3_matrices(p_mle)...)

res = lsim(sys, ufun, df.t[1:end-1]; x0 = rc3_params.u0)

