# See: https://www.linkedin.com/pulse/pipelining-meinolf-sellmann-spoxe/

import Ipopt
using JuMP


products = [:petrol, :diesel, :kerosine]
seq_items = [2, 4, 8, 16, 3, 9, 27, 5, 25]
# seq_items = [fill(1, 4); fill(2, 3); fill(3, 2)]

function perm2seq(p)
    res = ones(eltype(p), length(p))
    for i in eachindex(p)
        p[i] % 2 == 0 && continue
        res[i] = p[i] % 3 == 0 ? 2 : 3
    end
    return res
end

function isvalid(seq)
    for i in 1:length(seq)-1
        seq[i] == seq[i+1] && return false
    end
    return true
end

function nlp_model(sequence)
    products = [:petrol, :diesel, :kerosine]

    waste_costs = [
        0 2000 8000;
        2000 0 3500;
        8000 3500 0
    ]

    volumes = (
        petrol=200, diesel=150, kerosine=100
    )
    mdl = Model(optimizer_with_attributes(
        Ipopt.Optimizer, "print_level" => 0
    ))

    @variable(mdl, 1 <= u[1:9] <= 3)
    @variable(mdl, 10 <= V[1:9] <= 100)

    energy = @expression(mdl, energy[i=1:9], 500 * V[i] * u[i]^2)
    batch_time = @expression(mdl, batch_time[i=1:9], V[i] / (1.8 * u[i]))

    @constraint(mdl, time_limit, sum(batch_time) <= 168)

    for f in 1:3
        f_pos = findall(==(f), sequence)
        c = @constraint(mdl, sum(V[i] for i in f_pos) >= volumes[f])
        name = Symbol("volume_$(products[f])")
        mdl[name] = c
    end

    c_ij = [waste_costs[sequence[i], sequence[i+1]] for i in 1:8]
    switch_cost = @expression(mdl, switch_cost[i=1:8], c_ij[i] * u[i]^1.5)

    @objective(mdl, Min, sum(energy) + sum(switch_cost))
    return mdl
end

import Metaheuristics
using Metaheuristics: GA, OrderCrossover, SlightMutation, PermutationSpace, cardinality

function objective(p)
    M = 1e7
    seq = perm2seq(p)
    if ~isvalid(seq)
        return M
    end

    mdl = nlp_model(seq)
    optimize!(mdl)
    if is_solved_and_feasible(mdl)
        return objective_value(mdl)
    else
        return M
    end
end

search_space = PermutationSpace(seq_items)
algo = GA(crossover = OrderCrossover(), mutation = SlightMutation())

res = Metaheuristics.optimize(objective, search_space, algo)

opt_seq = perm2seq(res.best_sol.x)
@assert isvalid(opt_seq)

opt_mdl = nlp_model(opt_seq)
optimize!(opt_mdl)
@assert is_solved_and_feasible(opt_mdl)

using DataFrames

V_opt = round.(value.(opt_mdl[:V]), digits=2)
u_opt = round.(value.(opt_mdl[:u]), digits=3)
T = [products[i] for i in opt_seq]
df = DataFrame(:fuel => T, :volume => V_opt, :velocity => u_opt)
println("Objective value: ", objective_value(opt_mdl))
println("Results:")
println(df)
