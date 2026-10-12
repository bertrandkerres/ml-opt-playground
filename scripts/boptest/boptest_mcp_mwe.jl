using LinearAlgebra
using Statistics

using BOPTestAPI
using DataFrames
using Plots
using StaticArrays

using ControlSystemsBase
using ControlSystemIdentification
using ModelPredictiveControl

include("boptest_data.jl")
include("graybox_helpers.jl")


function scaled_iddata(df::AbstractDataFrame, colnames)
    u = Matrix(df[!, colnames["u"]])'
    um, us = mean(u; dims=2), std(u; dims=2)
    u = (u .- um) ./ us

    w = Matrix(df[!, colnames["w"]])'
    wm, ws = mean(w; dims=2), std(w; dims=2)
    w = (w .- wm) ./ ws

    y = Matrix(df[!, colnames["y"]])'
    ym, ys = mean(y; dims=2), std(y; dims=2)
    y = (y .- ym) ./ ys

    scales = (
        u = (mu = um, sigma = us),
        w = (mu = wm, sigma = ws),
        y = (mu = ym, sigma = ys),
    )

    dt = unique(diff(df.time))
    length(dt) == 1 || error("Sampling interval not constant")
    sysid_data = iddata(y, [u; w], dt[])
    return sysid_data, scales
end


function control_transform(u_scaled, scales, signal_names)
    u = u_scaled .* scales.u.sigma .+ scales.u.mu
    act_sigs = [s * "_activate" for s in signal_names]
    u_sigs = [s * "_u" for s in signal_names]
    return Dict(
        Pair.(act_sigs, 1)..., 
        Pair.(u_sigs, u)...
    )
end

function observation_inv_transform(y_dict::AbstractDict, scales, signal_names)
    ym, ys = scales.y.mu, scales.y.sigma
    y_vals = (y_dict[k] for k in signal_names)
    return vec((y_vals .- ym) ./ ys)
end


sysid_data, scales = scaled_iddata(df[:, :], colnames[TESTCASE])
sysh, x0h, opt_res = newpem(sysid_data, 1; focus=:prediction, h=24, zeroD=true, time_limit = 300)
predplot(sysh, sysid_data, h=8)

model = LinModel(sysh.sys, i_u=[1], i_d=[2])
mpc = LinMPC(model, Hp=4*12)

limits = Dict(
    "testcase1" => Dict(
        "umin" => [0.0],
        "umax" => [1e4],
    ),
    "bestest_hydronic" => Dict(
        "umin" => [293.15],
        "umax" => [353.15],
    ),
)
setconstraint!(
    mpc,
    umin=vec((limits[TESTCASE]["umin"] .- scales.u.mu) ./ scales.u.sigma),
    umax=vec((limits[TESTCASE]["umax"] .- scales.u.mu) ./ scales.u.sigma),
)

u0 = [0.0]
y0 = model(sysid_data.u[2, 1:1])

"""
    run_mpc!(mpc::PredictiveController, Ry::AbstractMatrix, D::AbstractMatrix, scales)

Executes the Model Predictive Control (MPC) algorithm for the given predictive controller.

# Arguments
- `mpc::PredictiveController`: The predictive controller instance that encapsulates the MPC logic and parameters.
- `Ry::AbstractMatrix`: The reference trajectory matrix that the controller aims to track.
  Size (number-of-signals)-by-(number of steps), i.e. time is in the second dimension.
- `D::AbstractMatrix`: The disturbance matrix representing external influences on the system.
  Size (number-of-signals)-by-(number of steps), i.e. time is in the second dimension.
- `scales`: A scaling factor or structure used to normalize or scale the inputs and outputs.

# Description
This function updates the state of the predictive controller by solving the optimization problem 
defined by the MPC formulation. It uses the provided reference trajectory, disturbance matrix, 
and scaling factors to compute the optimal control actions.

# Notes
- Ensure that the dimensions of `Ry` and `D` are compatible with the predictive controller's model.

"""
function run_mpc!(
    mpc::PredictiveController,
    Ry::AbstractMatrix,
    D::AbstractMatrix,
    scales,
    colnames;
    u0 = fill(0.0, mpc.estim.model.nu)
)
    N = mpc.Hp
    Ts = mpc.estim.model.Ts
    nu, ny = mpc.estim.model.nu, mpc.estim.model.ny
    u_name, y_names = colnames["u_input"], colnames["y"]

    size(Ry, 2) == size(D, 2) || throw(DimensionMismatch(
        "Reference trajectory `Ry` and measured disturbances `D` must have same number of elements in the second (time) dimension"
    ))
    N_steps = size(D, 2) - N - 1
    u_data = zeros(nu, N_steps)
    y_data = zeros(ny, N_steps)
    ry_data = zeros(ny, N_steps)

    u = u0
    t_bt, t_opt = 0.0, 0.0
    
    plant = BOPTestPlant("http://localhost", TESTCASE, dt = Ts)
    fc = getforecasts(plant, Ts * N_steps, Ts)
    res, kpi = try
        for i = 1:N_steps
            d = D[:, i]
            Dhat = D[:, i+1:i+N][:]
            udict = control_transform(u, scales, u_name)
            _t_bt = @elapsed y_dict = advance!(plant, udict)
            t_bt += _t_bt
            y = observation_inv_transform(y_dict, scales, y_names)
            ry = Ry[:, i]
            Rhaty = Ry[:, i+1:i+N][:]
            _t_opt = @elapsed u = begin
                preparestate!(mpc, y, d)
                u = moveinput!(mpc, ry, d, Dhat=Dhat, Rhaty=Rhaty)
                u_data[:,i], y_data[:,i], ry_data[:,i] = u, y, ry
                updatestate!(mpc, u, y, d)
                u
            end
            t_opt += _t_opt
        end
        getmeasurements(plant, 0.0, N_steps*Ts), getkpi(plant)
    finally
        stop!(plant)
        @info "Mean step time: BOPTEST: $(t_bt/N_steps) s, Opt: $(t_opt/N_steps) s"
    end
    res = leftjoin(fc, res, on = :time => :time)
    mpc_data = (u = u_data, y = y_data, ry = ry_data)
    timing = (bt = t_bt, opt = t_opt)
    return res, kpi, mpc_data, timing
end

D = copy(sysid_data.u[2,:]')
Ry = (Matrix(df[!, colnames[TESTCASE]["yr"]])' .- scales.y.mu) ./ scales.y.sigma

res, kpi, mpc_data, timing = run_mpc!(mpc, Ry, D, scales, colnames[TESTCASE])
plot_results(res, colnames[TESTCASE])