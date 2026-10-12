#=
Your client has funded 1,000 start-ups. If they can achieve a successful exit on a startup,
the return is anywhere between $5 and $10 million. Particularly, if startup number i in {0,..,999}
succeeds, the expected return is $10-0.005*i million. Alas, most startups fail and return nothing.

To improve the likelihood that a startup succeeds, we can support it with additional funds.
A budget of $30 million is available to support the development of the startups. If we invest
$S thousand in a startup, its expected likelihood of failure is

P_failure(S) = 0.3 + 0.65 / (1 + exp((S-40) / 8))

Without additional support, a startup therefore has a probability of about 95% that it will fail.
The more money we invest to support it, the lower the chances of failure and the higher the
probability that we will have an exit for anywhere between $5 to $10 million, depending on the
startup. But no matter how much we invest, we cannot lower the failure likelihood below 30%.

The simple question that the client has for you is how to invest the $30 million in the startups.
Since this is a simple deterministic optimization problem, surely you can help. Or can you?
=#

import Plots

P(s) = 0.3 + 0.65 / (1 + exp(s/8 - 5))
dP_ds(s) = -0.65 / 8 * exp(s/8 - 5) / (exp(s/8 - 5) + 1)^2
function d2P_ds2(s)
    z = s/8 - 5
    dz1 = exp(2z) / (exp(z) + 1)^3
    dz2 = -exp(z) / (2 * (exp(z) + 1)^2)
    return 0.65 / 32 * (dz1 + dz2)
end

s = 0:100
plt = Plots.plot(s, P.(s), xlabel="Investment [kUSD]", ylabel="P_fail", label="P_fail")
plt2 = Plots.twinx(plt)
Plots.plot!(plt2, s, dP_ds.(s), label="dP/ds", color=2)
Plots.plot!(plt2, s, d2P_ds2.(s), label="d2P/ds2", color=3)
plt


## As direct NLP model
using JuMP
using DataFrames
import Ipopt

function setup_solve_nlp(primal_start = 0.0)
    mdl = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 3))

    @variable(mdl, x[i=1:1000] >= 0, start = primal_start)
    @constraint(mdl, budget_constraint, sum(x) <= 30_000)
    @expression(mdl, prob_failure[i=1:1000], 0.3 + 0.65 / (1 + exp(x[i]/8 - 5)))
    @expression(mdl, expected_return[i=1:1000], (1 - prob_failure[i]) * (10_000 - 5*(i-1)))
    @objective(mdl, Max, sum(expected_return) - sum(x))

    optimize!(mdl)
    return mdl
end

function solve_search(S)
    sols = []
    for s in S
        mdl = setup_solve_nlp(s)
        if ~is_solved_and_feasible(mdl)
            continue
        end
        x = value.(mdl[:x])
        i_max = findfirst(<=(1), x)
        obj = objective_value(mdl)
        push!(sols, (; s, x, obj, i_max))
    end
    return sols
end

S0 = 0:5:100
sols = solve_search(S0)
df = DataFrame(
    :primal_start => S0,
    :obj => [s.obj for s in sols],
    :i_max => [s.i_max for s in sols]
)

plt1 = Plots.plot(
    df.primal_start,
    df.obj ./ 1e3,
    label="Objective value",
    xlabel="primal start for all x_i [k USD]",
    ylabel="Objective [M USD]"
)
plt2 = Plots.twinx(plt1)
Plots.plot!(plt2, df.primal_start, df.i_max, label="Last startup funded", ylabel="i_max", color=2)
plt1

## Using ODE
# From optimality condition:
# ∂E/∂(s_i) = lambda = const for all i
# F(s, i) := ∂E/∂(s_i) = (10000 - 5i) * dP/d(s_i)
# Optimal:
# dF/di = 0
# dF/di = ∂F/∂i + ∂F/∂s * ∂s/∂i
# ∂F/∂s = (10000 - 5i) * ∂²P/∂s²
# => ∂s/∂i = - ∂F/∂i / (∂F/∂s)
import DifferentialEquations as DE
import SciMLBase: successful_retcode

function dynamics(s, p, t)
    if s >= 39.5 && s <= 40.5
        # singularity
        return 0.0
    end
    ∂F_∂t = -5 * dP_ds(s)
    ∂F_∂s = (10_000 - 5*t) * d2P_ds2(s)
    ds = - ∂F_∂t / ∂F_∂s
    return ds
end

s0 = 50.0
t = 1:1000
prob = DE.ODEProblem(dynamics, s0, (t[begin], t[end]))
sol = DE.solve(prob, DE.Tsit5())
Plots.plot(sol)

function expected_profit(S)
    cum_profit = 0.0
    for (i, s) in enumerate(S)
        pf_i = P(s)
        cum_profit += (1 - pf_i) * (10_000 - 5*(i-1)) - s
    end
    return cum_profit
end

function loss(s0)
    t = 1:1000
    prob = DE.ODEProblem(dynamics, s0, (t[begin], t[end]))
    sol = DE.solve(prob, DE.Tsit5(); saveat = t)
    if ~successful_retcode(sol)
        return Inf
    end
    invested = cumsum(sol.u)
    S = copy(sol.u)
    S[invested .> 30_000] .= 0.0
    return -expected_profit(S)
end

S0 = 45:0.01:70
L = [-loss(s0) for s0 in S0]
Plots.plot(S0, L ./ 1000, xlabel="s0 [kUSD]", ylabel="E(profit) [kUSD]")