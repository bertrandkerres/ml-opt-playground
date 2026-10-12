using LinearAlgebra
using Statistics

using BOPTestAPI
using DataFrames
using Plots
using StaticArrays

function plot_results(df::AbstractDataFrame, colnames)
    pl1 = plot(
        df.time ./ 3600,
        Matrix(df[!, [colnames["y"]; colnames["yr"]]]);
        xlabel = "t [h]",
        ylabel = "y",
        labels = reduce(hcat, [colnames["y"]; colnames["yr"]]),
    )
    pl2 = plot(
        df.time ./ 3600,
        Matrix(df[!, colnames["u"]]);
        xlabel = "t [h]",
        ylabel = "u",
        labels = reduce(hcat, colnames["u"])
    )
    pl3 = plot(
        df.time ./ 3600,
        Matrix(df[!, colnames["w"]]);
        xlabel = "t [h]",
        ylabel = "w",
        labels = reduce(hcat, colnames["w"])
    )
    return plot(pl1, pl2, pl3; layout = (3, 1), size=(600, 900))
end

dt = 900.0 # time step in seconds
TESTCASE = "bestest_hydronic"

plant = BOPTestPlant("http://localhost", TESTCASE, dt = dt)

# Get available measurement points
mpts = measurement_points(plant)

N = round(Int, 3600 / dt * 24 * 7 * 12)

# Get forecast data as well (for plotting later)
fc = getforecasts(plant, N * dt, dt)

# Run N time steps of baseline control
res = []
for t = 1:N
    u = Dict() # here you would put your own controller
    y = advance!(plant, u)
    push!(res, y)
end
baseline_kpi = getkpi(plant)
stop!(plant)

dfres = DataFrame(res)

# Create single df with all data
df = leftjoin(dfres, fc, on = :time => :time)
disallowmissing!(df)

colnames = Dict(
    "testcase1" => Dict(
        "y" => ["TRooAir_y"],
        "u" => ["PHea_y"],
        "w" => ["TDryBul"],
        "yr" => ["LowerSetp[1]"],
        "u_input" => ["oveAct"]
    ),
    "bestest_hydronic" => Dict(
        "y" => ["reaTRoo_y"],
        "u" => ["oveTSetSup_u"],
        "w" => ["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHGloHor_y"],
        "yr" => ["LowerSetp[1]"],
        "u_input" => ["oveTSetSup"]
    ),
)
