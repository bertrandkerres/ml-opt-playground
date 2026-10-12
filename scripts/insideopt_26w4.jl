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

using SDDP
import HiGHS
import SCS

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

function generate_fixed_graph(x, P, Ω, N; upper_bound = 300.0)
    if ~isapprox(sum(x), 1.0)
        throw(ArgumentError("Shares must sum to 1"))
    end
    x1, x2, x3, x4 = x[1], x[2], x[3], x[4]
    model_static = SDDP.LinearPolicyGraph(;
        stages = N, sense = :Max, upper_bound = upper_bound, optimizer = SCS.Optimizer,
    ) do subproblem, node
        @variable(subproblem, budget >= 0, SDDP.State, initial_value = initial_budget)
        @variable(subproblem, rel_payout)
        SDDP.parameterize(subproblem, Ω, P) do ω
            JuMP.fix(rel_payout, ω[1] * x1 + ω[2] * x2 + ω[3] * x3)
        end
        @variable(subproblem, payout)
        @constraint(subproblem, payout == rel_payout * budget.in)
        @constraint(subproblem, budget.out == x4 * budget.in + payout)
        @stageobjective(subproblem, 0)
    end
    return model_static
end

x0 = [0.05, 0.8, 0.05, 0.1]
static_model = generate_fixed_graph([0.05, 0.8, 0.05, 0.1], collect(P), collect(Ω), N)
SDDP.train(static_model, iteration_limit = 10)

static_simulations = SDDP.simulate(static_model, 1000, [:budget, :payout], parallel_scheme = SDDP.Threaded())
import Plots
SDDP.publication_plot(
    static_simulations;
    quantile = [0.1, 0.25, 0.5, 0.75, 0.9],
    ylabel = "total_cash") do sim
        return sim[:budget].out
end


function generate_dynamic_graph(P, Ω, N; upper_bound = 300.0)
    mdl_dyn = SDDP.LinearPolicyGraph(;
        stages = N, sense = :Max, upper_bound = upper_bound, optimizer = SCS.Optimizer,
    ) do subproblem, node
        @variable(subproblem, budget >= 0, SDDP.State, initial_value = initial_budget)
        @variable(subproblem, share[1:4] >= 0, SDDP.State, initial_value = 0.0)
        
        if node == 1
            @constraint(subproblem, share[1].out + share[2].out + share[3].out + share[4].out == 1)
            @constraint(subproblem, budget.out == budget.in)
            @stageobjective(subproblem, budget.out)
        else
            @variable(subproblem, R[1:4])
            @constraint(subproblem, ret1, share[1].in - R[1] == 0)
            @constraint(subproblem, ret2, share[2].in - R[2] == 0)
            @constraint(subproblem, ret3, share[3].in - R[3] == 0)
            @constraint(subproblem, ret4, share[4].in - R[4] == 0)
            SDDP.parameterize(subproblem, Ω, P) do ω
                JuMP.set_normalized_coefficient(ret1, share[1].in, ω[1])
                JuMP.set_normalized_coefficient(ret2, share[2].in, ω[2])
                JuMP.set_normalized_coefficient(ret3, share[3].in, ω[3])
            end
            @variable(subproblem, total_roi)
            @constraint(subproblem, [i=1:4], share[i].out == share[i].in)
            @constraint(subproblem, total_roi == R[1] + R[2] + R[3] + R[4])
            @constraint(subproblem, budget.out == budget.in * total_roi)
            @stageobjective(subproblem, budget.out)
        end
    end
    return mdl_dyn
end

dyn_model = generate_dynamic_graph(collect(P), collect(Ω), 40; upper_bound = 50)
SDDP.train(dyn_model, iteration_limit = 10, parallel_scheme = SDDP.Threaded())
dyn_simulations = SDDP.simulate(dyn_model, 20, [:budget, :share], parallel_scheme = SDDP.Threaded())
import Plots
SDDP.publication_plot(
    dyn_simulations;
    quantile = [0.1, 0.25, 0.5, 0.75, 0.9],
    ylabel = "total_cash") do sim
        return sim[:budget].out
end


model_invest_decision = SDDP.LinearPolicyGraph(;
    stages = N, sense = :Max, upper_bound = 300.0, optimizer = SCS.Optimizer,
) do subproblem, node
    @variable(subproblem, saved >= 0, SDDP.State, initial_value = initial_budget)
    @variable(subproblem, invested >= 0, SDDP.State, initial_value = 0.0)
    @variable(subproblem, 0 <= share[1:3] <= 1, SDDP.State, initial_value = 0.0)
    
    if node == 1
        @constraint(subproblem, share[1].out + share[2].out + share[3].out == 1)
        @constraint(subproblem, invested.out <= saved.in)
        @constraint(subproblem, saved.out == saved.in - invested.out)
        @stageobjective(subproblem, saved.out)
    else
        # Cannot change relative investment amounts
        @constraint(subproblem, [i=1:3], share[i].out == share[i].in)
        @variable(subproblem, payout)
        @variable(subproblem, R[1:3])
        SDDP.parameterize(subproblem, Ω, P) do ω
            JuMP.fix(R[1], ω[1])
            JuMP.fix(R[2], ω[2])
            JuMP.fix(R[3], ω[3])
        end
        @constraint(subproblem, payout == R[1] * share[1].in + R[2] * share[2].in + R[3] * share[3].in)
        @constraint(subproblem, invested.out <= saved.in + payout * invested.in)
        @constraint(subproblem, saved.out == saved.in + payout * invested.in - invested.out)
        @stageobjective(subproblem, payout * invested.in - invested.out)
    end
end

SDDP.train(model_invest_decision, iteration_limit = 100, parallel_scheme = SDDP.Threaded())

simulations = SDDP.simulate(model_invest_decision, 20, [:saved, :invested, :share], parallel_scheme = SDDP.Threaded())

import Plots
Plots.plot(
    SDDP.publication_plot(
        simulations;
        quantile = [0.1, 0.25, 0.5, 0.75, 0.9],
        ylabel = "total_cash") do sim
            return sim[:invested].out + sim[:saved].out
        end,
    SDDP.publication_plot(simulations; ylabel = "x1") do sim
        return sim[:share][1].out
    end;
    layout = (2, 1),
)

#=
import MathOptInterface as MOI
mdl = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_MOF)
MOI.read_from_file(mdl, "subproblem_10.mof.json")
MOI.Utilities.latex_formulation(mdl)
=#