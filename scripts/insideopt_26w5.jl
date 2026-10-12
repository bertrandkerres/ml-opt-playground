#=
In the face of scarce resources, we are sometimes forced to make very hard choices.
Analytics can help us make most of what we have.
Can you help this municipality make vaccine allocations to its different communities?

The municipality consists of three towns, one hamlet, and one village. Each community has
different population sizes. In total, the municipality received 50,000 doses of an oral
vaccine from the state for a total population of 120,000. 

To get close to herd immunity, each community needs to vaccinate about 60% of its population.
The likelihood of reaching herd immunity is calculated as

P = 1 / (1 + exp((T-v) / 10))

where T is 0.6 * population size and v is the number of people vaccinated.
Population sizes:

City, population
Phlegmus, 40000
Bronchtisburgh, 30000
Snottington-on-Sea, 25000
Wheezewick, 15000
Hack Hollow, 10000

How would you allocate the available doses of oral vaccine to the different communities? 
=#

import CSV
import StaticArrays
using DataFrames

data = """
City,population
Phlegmus, 40000
Bronchtisburgh, 30000
Snottington-on-Sea, 25000
Wheezewick, 15000
Hack Hollow, 10000
"""

df = CSV.read(IOBuffer(data), DataFrame)
df.population ./= 1000
df.T = df.population .* 0.6

function P(v, T)
    z = (T - v) * 100
    return one(z) / (one(z) + exp(z))
end

function expected_immune(v, n)
    return v + P(v, 0.6 * n) * (n - v)
end

function objective(V, N)
    min_obj = -sum(expected_immune.(V, N))
    ineq_con = [-1.0]
    eq_cons = [sum(V) - 50.0]
    return min_obj, ineq_con, eq_cons
end

import QuasiMonteCarlo
lb = fill(0.0, 5)
ub = df.population
n = 10000

V = QuasiMonteCarlo.sample(n, lb, ub, QuasiMonteCarlo.GridSample())
V = V[:, vec(sum(V, dims=1) .<= 50)]

import Metaheuristics

bounds = Metaheuristics.boxconstraints(lb = lb, ub = ub)
options = Metaheuristics.Options(
    time_limit = 60.0,
    iterations = 100_000,
    f_calls_limit = 1_000_000,
#    parallel_evaluation = true,
    store_convergence = true,
)
algorithm = Metaheuristics.ECA(options = options)

res = Metaheuristics.optimize(v -> objective(v, df.population), bounds, algorithm)

function immune_population(T, v; N_mc::Int = 10_000)
    r = rand(N_mc)
    p = P.(T, v)
end


# Deterministic optimization model
# Knapsack where weights := T, rewards := 0.4 * pop (free gain through herd immunity)
using JuMP
using LinearAlgebra
import HiGHS

function create_knapsack(populations, eta::Real)
    mdl = JuMP.Model(HiGHS.Optimizer)
    @variable(mdl, x[1:5], Bin)
    @constraint(mdl, dot(eta .* populations, x) <= 50)
    @objective(mdl, Max, dot((1 .- eta) .* populations, x))
    return mdl
end

function knapsack_objective(xi, populations, eta, n_vacc)
    n_immune = dot(xi, populations)
    vaccines_used = n_immune * eta
    return n_immune + n_vacc - vaccines_used
end

function setup_and_solve(populations, eta)
    mdl = create_knapsack(populations, eta)
    JuMP.optimize!(mdl)
    if ~is_solved_and_feasible(mdl)
        return fill(NaN, 5), NaN
    end
    xi = value.(mdl[:x])
    obj = knapsack_objective(xi, populations, eta, 50.0)
    return xi, obj 
end

eta = 0.55:0.0025:0.65
reward = setup_and_solve.(Ref(df.population), eta)
X = reduce(hcat, (first(r) for r in reward))
R = [last(r) for r in reward]

df_res = DataFrame(round.(Int, X)', :auto)
insertcols!(df_res, 1, :eta => eta)
insertcols!(df_res, :n_imm => R)

import Plots
Plots.plot(
    df_res.eta,
    df_res.n_imm,
    xlabel = "Actual T_rel",
    ylabel = "Imm. population ('000)"
)