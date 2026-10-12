#=
One way to make money with money is to sell insurance to people who want to protect their existing
investments. On average, you win, but there is a chance you may lose.

1. Invest $1. Win $1.00: 55%, lose $1.00: 45%
2. Invest $1. Win $0.05: 50%, lose $0.04: 50%
3. Invest $1. Win $0.56: 51%, lose $0.53: 49%

Consider the three financial instruments above. Each instrument protects the same asset,
so if you buy, e.g., $100 of Instrument 2, then there are only two possible outcomes after one week:
You either walk away with $105 with 50% probability or you are left with only $96 with the same
probability. Of course, if we invest different amounts of money in all three instruments, then
there are 8 potential outcomes. The three different bets are independent from each other; there
is no correlation.

Our initial budget is $10,000. We offer these instruments every week for 2 years, so for 104
weeks in total. We need to decide what percentage of our current budget we invest in each
instrument. This percentage will remain constant for the entire 2 years.

Our objective is twofold: 

* Maximize our expected budget at the end of two years. It should be at least $15,000, but it would
  be great if we can triple our money. 
* Maximize the average budget in the 10% worst scenarios. The CVaR-10% should be no less than $8,000
  and ideally be close to $12,000.

Can you advise how best to invest our money?
=#

import StatsBase
using StaticArrays
using Statistics

N = 104
const initial_budget = 10.0

const P_Ω = [
    (0.55 => 2.0, 0.45 => 0.0),
    (0.5 => 1.05, 0.5 => 0.96),
    (0.51 => 1.56, 0.49 => 0.47),
]

P = (
    first(P_Ω[1][i]) * first(P_Ω[2][j]) * first(P_Ω[3][k])
    for i in 1:2 for j in 1:2 for k in 1:2
)
Ω = (
    (a=last(P_Ω[1][i]), b=last(P_Ω[2][j]), c=last(P_Ω[3][k]))
    for i in 1:2 for j in 1:2 for k in 1:2
)

function simulate_inner(x, P, Ω, N::Int; budget = 10.0)
    R = StatsBase.wsample(Ω, P, N)
    for r in R
        rel_payout = x[1] * r[1] + x[2] * r[2] + x[3]* r[3] + x[4]
        budget *= rel_payout
    end
    return budget
end

function simulate(x, P, Ω; N_periods = 104, N_mc = 10_000, budget = 10.0)
    Pvec = collect(P)
    Ωvec = collect(Ω)
    res = [simulate_inner(x, Pvec, Ωvec, N_periods) for _ in 1:N_mc]
    return res
end


function inner_objective_function(x, P, Ω)
    x4 = max(1 - sum(x), 0.0)
    _x = StaticArrays.SA[x[1], x[2], x[3], x4]
    R = simulate(_x, P, Ω)
    return R
end

function objective(x, P, Ω)
    R = inner_objective_function(x, P, Ω)
    fx1 = -mean(log1p.(R))          # max log(E(R))
    q10 = quantile(R, 0.1)          # 10th %-tile relative to 10 USD
    fx2 = -mean(log1p.(R[R .< q10]))

    gx = [sum(x) - 1.0, -x[1], -x[2], -x[3]]
    hx = [0.0]
    return [fx1, fx2], gx, hx
end


function objective_parallel(X, P, Ω)
    m = size(X, 1)
    fx, gx, hx = zeros(m, 2), zeros(m, 4), zeros(m, 1)
    Threads.@threads for i in 1:m
        x = X[i, :]
        fx[i, :], gx[i, :], hx[i, :] = objective(x, P, Ω)
    end
    return fx, gx, hx
end


using QuasiMonteCarlo
lb = fill(0.0, 4)
ub = fill(1.0, 4)
n = 100

S = QuasiMonteCarlo.sample(n, lb, ub, SobolSample())
S ./= sum(S, dims=1)

_R = [inner_objective_function(s[1:3], P, Ω) for s in eachcol(S)]
R = reduce(hcat, _R)

import Metaheuristics

bounds = Metaheuristics.boxconstraints(lb = [0.0, 0.0, 0.0], ub = [1.0, 1.0, 1.0])
options = Metaheuristics.Options(
    x_tol = 0.001,
    f_tol = 0.001,
    time_limit = 60.0,
    parallel_evaluation = true,
    store_convergence = true,
)
algorithm = Metaheuristics.NSGA2(options = options)

res = Metaheuristics.optimize(x -> objective_parallel(x, P, Ω), bounds, algorithm)

pf = Metaheuristics.pareto_front(res)
pf_norm = expm1.(-pf)

import Plots
plt = Plots.scatter(
    pf_norm[:, 1],
    pf_norm[:, 2],
    xlabel = "exp(mean(logR))",
    ylabel = "exp(CVar10(logR))",
)

idx = findall((pf_norm[:, 1] .> 20) .& (pf_norm[:, 2] .> 8))
res_opt = res.population[[idx...]]

for r in res_opt
    x = [r.x; 1 - sum(r.x)]
    _R = simulate(x, P, Ω, N_mc = 100_000)
    q = quantile(_R, 0.1)
    m = mean(_R)
    cvar = mean(_R[_R .< q])
    println(
        "x = ", round.(x, digits=3),
        ", E(x) = ", round(m, digits=2),
        ", q10(x) = ", round(q, digits=2),
        ", CVaR(x) = ", round(cvar, digits=2),
    )
end

res_best = res_opt[1]
x_best = [res_best.x; 1 - sum(res_best.x)]
R_best = simulate(x_best, P, Ω, N_mc = 100_000)
Plots.histogram(R_best)
Plots.xlims!(0, 100)
println("Mean: ", mean(R_best))
println("q10: ", quantile(R_best, 0.1))
println("CVar10: ", mean(R_best[R_best .< quantile(R_best, 0.1)]))