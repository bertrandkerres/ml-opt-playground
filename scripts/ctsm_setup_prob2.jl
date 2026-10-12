# This reads example data and sets up ODE functions for parameter estimation
# The data and ODE form are adapted from https://ctsm.info/building2.pdf

using Statistics

import CSV: read as read_csv
using DataFrames
using Plots

using ComponentArrays
using DataInterpolations
using UnPack
using StaticArrays
import SciMLBase

df = read_csv(joinpath(pwd(), "data/example_building/inputPRBS1.csv"), DataFrame, ntasks=1)

plot(df.t, df.Ta, xlabel="t [h]", ylabel="T_amb [degC]")
plot(df.t, df.Ph, xlabel="t [h]", ylabel="P [kW]")
plot!(SmoothedConstantInterpolation(df.Ph, df.t))

plot(df.t, df.Ps, xlabel="t [h]", ylabel="Ps [kW/m2]")
plot!(CubicSpline(df.Ps, df.t))

interpolations = (
    Ta = CubicSpline(df.Ta, df.t),
    Ps = CubicSpline(df.Ps, df.t),
    Ph = SmoothedConstantInterpolation(df.Ph, df.t),
    yTi = CubicSpline(df.yTi, df.t),
)

function _dynamics_r2c2!(du, u, p, t, interpolations)
    @unpack Ci, Ce, Rie, Rea, Aw = p
    Ti, Te = u[1], u[2]
    Ps = interpolations.Ps(t)
    Ph = interpolations.Ph(t)
    Ta = interpolations.Ta(t)
    du[1] = 1/(Ci*Rie) * (Te-Ti) + Aw/Ci * Ps + 1/Ci * Ph
    du[2] = 1/(Ce*Rie) * (Ti-Te) + 1/(Ce*Rea) * (Ta-Te)
    return nothing
end


function _dynamics_r3c3!(du, u, p, t, interpolations)
    @unpack Ci, Ce, Ch, Rie, Rea, Rih, Aw = p
    Ti, Te, Th = u[1], u[2], u[3]
    Ps = interpolations.Ps(t)
    Ph = interpolations.Ph(t)
    Ta = interpolations.Ta(t)
    du[1] = 1/(Ci*Rih) * (Th-Ti) + 1/(Ci*Rie) * (Te-Ti) + Aw/Ci * Ps
    du[2] = 1/(Ce*Rie) * (Ti-Te) + 1/(Ce*Rea) * (Ta-Te)
    du[3] = 1/(Ch*Rih) * (Ti-Th) + 1/Ch * Ph
    return nothing
end

dynamics_r2c2!(du, u, p, t) = _dynamics_r2c2!(du, u, p, t, interpolations)
dynamics_r3c3!(du, u, p, t) = _dynamics_r3c3!(du, u, p, t, interpolations)

function _loss_fun(sol, interpolations)
    tot_loss = 0.0
    if any((!SciMLBase.successful_retcode(s.retcode) for s in sol))
        tot_loss = Inf
    else
        yTi = interpolations.yTi(sol.t)
        Ti_hat = reduce(hcat, sol.u)[1, :]
        tot_loss = mean(abs2, yTi .- Ti_hat)
    end
    return tot_loss
end

loss_fun(sol) = _loss_fun(sol, interpolations)


rc2_params = (
    p_corr = ComponentVector(Ci = 5.72, Ce = 1.44, Rie = 2.43, Rea = 1.19, Aw = 23.8),
    p0 = ComponentVector(Ci = 5.0, Ce = 0.5, Rie = 5.0, Rea = 0.5, Aw = 10),
    u0 = [df.yTi[1], 10.0],
)

rc3_params = (
    p_corr = ComponentVector(
        Ce = 2.92, Ch = 1e-3, Ci = 1.07, Rea = 4.54, Rih = 1.19e2, Rie = 8.63e-1, Aw = 5.63,
    ),
    p0 = ComponentVector{Float64}(
        Ce = 10.0, Ch = 1.0, Ci = 0.1, Rea = 10.0, Rih = 1.0, Rie = 10.0, Aw = 10.0,
    ),
    u0 = [df.yTi[1], 5.0, 15.0]
)

function rc3_matrices(p)
    @unpack Ci, Ce, Ch, Rie, Rea, Rih, Aw = p
    A = @SMatrix [-(1/(Ci*Rih) + 1/(Ci*Rie)) 1/(Ci*Rie)                 1/(Ci*Rih);
                  1/(Ce*Rie)                 -(1/(Ce*Rie) + 1/(Ce*Rea)) 0;
                  1/(Ch*Rih)                 0                          -1/(Ch*Rih)]
    B = @SMatrix [Aw/Ci     0           0;
                  0         1/(Ce*Rea)  0;
                  0         0           1/Ch]
    C = @SMatrix [1.0 0 0;
                  0 1.0 0;
                  0 0 1.0]
    D = @SMatrix zeros(3, 3)
    return A, B, C, D
end
