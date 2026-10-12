#=
The simplest sports bet is on the DS favorite passtime, Wobbleknock. Each team has four members,
but the goalie in Fantasy Wobbleknock is always given. That leaves just three players that we can
choose. There are 18 players in total, 6 for each position.

Name,Position,Salary,MeanScore,StdDev
Johannes,Alpha,38.5,28.8,1.4
Julia,Alpha,36.2,28.7,1.8
Sturla,Alpha,34.0,28.5,1.9
Ingrid,Alpha,32.5,28.3,2.0
Lou,Alpha,30.8,28.2,2.1
Tarjei,Alpha,29.0,28.0,2.8
Sebastian,Beta,35.5,28.8,5.7
Elvira,Beta,33.8,28.6,5.2
Tommaso,Beta,32.1,28.5,4.5
Eric,Beta,28.4,28.5,3.5
Justine,Beta,28.7,28.4,3.2
Lisa,Beta,27.0,28.2,2.9
Vetle,Gamma,35.5,28.9,4.2
Quentin,Gamma,34.2,28.8,4.7
Dorothea,Gamma,33.1,28.6,3.8
Hanna,Gamma,31.8,28.5,5.0
Martin,Gamma,30.5,28.5,5.7
Andrejs,Gamma,29.2,28.3,2.8

One can only purchase one ticket in Fantasy Wobbleknock. On this ticket, we get to select two
teams, i.e., two times three players that form two teams. If one or both of those two teams scores
at least as many fantasy points than anyone else's bets, the ticket wins the jackpot. If multiple
tickets win, the jackpot is split equally among all winning teams.

The same player may appear in both teams. The only constraint is that the total salary paid to all
members on a team cannot exceed $100. The stats table above gives, for every player, the mean
fantasy points score as well as the standard deviation for a normal distribution around that mean.

The simple task for you is to form two teams from the players above so that the chances are
maximized that at least one of the two teams gets the high score. 
=#

import CSV: read as read_csv
using DataFrames

using JuMP
using LinearAlgebra
import HiGHS

import Plots

data = read_csv(joinpath(pwd(), "data/insideopt_26w2_data.csv"), DataFrame)
data.Variance = data.StdDev .^ 2

disp_names = let data = data
    n -> data.Name[[n...]]
end
# Knapsack deterministic formulation
# i.e. assuming every player will score according to MeanScore
function milp_formulation(salaries::T1, scores::T2) where {
    T1 <: AbstractVector{<:AbstractVector},
    T2 <: AbstractVector{<:AbstractVector}
}
    kp_mdl = Model(HiGHS.Optimizer)

    score_vars = []
    cost_vars = []
    for (p, (pos_salary, pos_score)) in enumerate(zip(salaries, scores))
        n = length(pos_salary)
        x = @variable(kp_mdl, [1:n], base_name = "pos_$p", binary = true)
        registered_name = Symbol("pos_$p")
        kp_mdl[registered_name] = x
        c = @variable(kp_mdl, base_name = "cost_$p")
        es = @variable(kp_mdl, base_name = "expected_score_$p")
        push!(cost_vars, c)
        push!(score_vars, es)
        @constraint(kp_mdl, sum(x) == 1)
        @constraint(kp_mdl, dot(x, pos_salary) == c)
        @constraint(kp_mdl, dot(x, pos_score) == es)
    end

    @constraint(kp_mdl, sum(cost_vars) <= 100)
    @objective(kp_mdl, Max, sum(score_vars))

    return kp_mdl
end

function winning_team(solved_model)
    vals = [
        value.(solved_model[:pos_1]);
        value.(solved_model[:pos_2]);
        value.(solved_model[:pos_3])
    ]
    return round.(Bool, vals)
end


data_grp = groupby(data, :Position)
salaries = [df.Salary for df in data_grp]
expected_scores = [df.MeanScore for df in data_grp]
kp_mdl = milp_formulation(salaries, expected_scores)
optimize!(kp_mdl)

println("Deterministic solution:")
println(solution_summary(kp_mdl))
println("Selected team:")
println(data[winning_team(kp_mdl), :Name])
println("Expected score: " * string(objective_value(kp_mdl)))


## MC stochastic solutions
# MC of knapsack problems (i.e. find best team MC)
using Distributions

N = 1_000

dist = MvNormal(data.MeanScore, Diagonal(data.Variance))
raw_realized_scores = rand(dist, N)
realized_scores = ([r[1:6], r[7:12], r[13:18]] for r in eachcol(raw_realized_scores))

sol = []
objs = Float64[]
for scores in realized_scores
    mdl = milp_formulation(salaries, scores)
    optimize!(mdl)
    if is_solved_and_feasible(mdl)
        cur_sol = winning_team(mdl)
        obj = objective_value(mdl)
    else
        cur_sol = fill(false, 18)
        obj = NaN
    end
    push!(sol, cur_sol)
    push!(objs, obj)
end


Plots.histogram(objs)

best_team = reduce(hcat, sol)

function count_team_compositions(best_teams)
    unique_teams = unique(eachcol(best_teams))
    counts = Int[]
    for t in unique_teams
        tc = 0
        for c in eachcol(best_teams)
            if t == c
                tc += 1
            end
        end
        push!(counts, tc)
    end
    p = sortperm(counts, rev = true)
    return unique_teams[p], counts[p]
end

teams, counts = count_team_compositions(best_team)

##
# A pure MC formulation to align with the additional info that
# the aim is to beat the deterministic reference team by 10%

function all_possible_teams(data)
    cost = data.Salary
    indices = (
        team for team in Base.product(1:6, 7:12, 13:18)
        if sum(cost[[team...]]) <= 100
    )
    return indices
end

ref_team = findall(winning_team(kp_mdl))

function mc_data(data, ref_team; N = 10_000)
    dist = MvNormal(data.MeanScore, Diagonal(data.Variance))
    realized = rand(dist, N)
    
    ref_score = sum(realized[ref_team, :], dims=1)
    teams_vec = collect(all_possible_teams(data))
    team_scores = sum_scores(realized, teams_vec)
    return team_scores, ref_score, teams_vec
end

function sum_scores(X, teams)
    team_scores = zeros(length(teams), size(X, 2))
    for (j, s) in enumerate(eachcol(X))
        for (i, t) in enumerate(teams)
            team_scores[i, j] = s[t[1]] + s[t[2]] + s[t[3]]
        end
    end
    return team_scores
end


N = 1_000_000
@time team_scores, ref_score, all_teams = mc_data(data, ref_team; N)
is_winning = team_scores .> 1.1 .* ref_score

function count_either_wins(is_winning)
    col_major_X = copy(transpose(is_winning))
    n_teams = size(is_winning, 1)
    n_wins_comb = zeros(Int, n_teams, n_teams)
    for i in 1:n_teams
        for j in (i+1):n_teams
            w = sum(@views col_major_X[:, i] .| col_major_X[:, j])
            n_wins_comb[i, j] = w
            n_wins_comb[j, i] = w
        end
    end
    return n_wins_comb
end

@time n_wins_comb = count_either_wins(is_winning)

team1, team2 = Tuple(argmax(n_wins_comb))
win_pct = n_wins_comb[team1, team2] / N
println("Pure MC based solution")
println("Team 1:")
println(data.Name[[all_teams[team1]...]])
println("Team 2:")
println(data.Name[[all_teams[team2]...]])
println("Expected win percentage: " * string(round(win_pct * 100; digits = 2)) * "%")

## Using somewhat analytical method
# 1. Calculate the distribution of Z_t: the difference to the reference team for team t
#    Z_t is normally distributed with mu = sum(mu_t) - sum(mu_ref) 
#    and s2 = sum(var_tr) where tr is the symmetric difference between t and ref
# 2. We want to max P(Z_1 > D or Z_2 > D), where D is an absolute score distance to
#    the reference team. This is equivalent to min P(Z_1 < D and Z_2 < D).
# 3. For independent Z_i, Z_j, this means to find 
#       min_{i,j} P(Z_i < D) * Pr(Z_j < D)
#     = min_{i,j} cdf(Z_i, D) * cdf(Z_j, D)
#    but Z_i and Z_j are only independent if they don't share a team member.
import Copulas

function marginal_distributions(data, ref_team)
    D_vec = Normal{Float64}[]
    Sr = BitSet(ref_team)
    for t in all_possible_teams(data)
        St = BitSet(t)
        m, s2 = univariate_norm_params(St, Sr, data.MeanScore, data.Variance)
        push!(D_vec, Normal(m, s2))
    end
    return D_vec
end

function univariate_norm_params(St, Sr, mu_vec, var_vec)
    S_pos = setdiff(St, Sr)
    S_neg = setdiff(Sr, St)

    m = sum((mu_vec[i] for i in S_pos); init = 0.0) - sum((mu_vec[i] for i in S_neg); init = 0.0)
    s2 = sqrt(sum((var_vec[i] for i in symdiff(St, Sr)); init = 0.0))
    return m, s2
end


function first_nonintersecting_team(t, all_teams)
    St = BitSet(t)
    for (i, t2) in enumerate(all_teams)
        if isempty(intersect(St, t2))
            return i, t2
        end
    end
    return -1, nothing
end

target_diff = 8.6
md = marginal_distributions(data, ref_team)
P_marg = cdf.(md, target_diff)
si = sortperm(P_marg)

t1_md = all_teams[si[1]]
i_t2, t2_md = first_nonintersecting_team(t1_md, all_teams[si])

win_pct_md = 1 - P_marg[si[1]] * P_marg[i_t2]
println("Analytic using marginal distributions")
println("Team 1:")
println(disp_names(t1_md))
println("Team 2:")
println(disp_names(t2_md))
println("Expected win rate: " * string(round(win_pct_md * 100, digits=2)) * "%")

function covariance_matrix(data; ref_team = [])
    all_teams = collect(all_possible_teams(data))
    N = length(all_teams)
    C = zeros(N, N)
    Sr = BitSet(ref_team)
    for i in 1:N
        Si = symdiff(Sr, all_teams[i])
        C[i, i] = sum((data.Variance[k] for k in Si); init = 0.0)
        for j in (i+1):N
            Sj = symdiff(Sr, all_teams[j])
            c_ij = sum((data.Variance[k] for k in intersect(Si, Sj)); init = 0.0)
            C[i, j] = c_ij
            C[j, i] = c_ij
        end
    end
    return C
end

function bivariate_sklar(d1, d2, C)
    size(C) == (2, 2) || error("C must be a 2-by-2 matrix")
    gc = Copulas.GaussianCopula(C)
    S = Copulas.SklarDist(gc, (d1, d2))
    return S
end

function all_copulas(marginal_distributions, covariances)
    N = length(marginal_distributions)
    @assert N == size(covariances, 1) == size(covariances, 2)

    d = Dict{Tuple{Int, Int}, Copulas.SklarDist}()
    for i in 1:N
        for j in (i+1):N
            c_ij = covariances[i, j]
            C = [covariances[i,i] c_ij; c_ij covariances[j,j]]
            d[(i, j)] = bivariate_sklar(
                marginal_distributions[i], marginal_distributions[j], C
            )
        end
    end
    return d
end


function eval_cdf(marginal_dist, copulas, x)
    N = maximum((max(k...)) for k in keys(copulas))
    length(marginal_dist) == N || throw(DimensionMismatch(
        "Marginal distributions and keys in `copulas` do not match"
    ))
    res = zeros(N, N)
    for i in 1:N
        res[i, i] = cdf(marginal_dist[i], x[1])
        for j in (i+1):N
            P = cdf(copulas[(i, j)], x)
            res[i, j] = P
            res[j, i] = P
        end
    end
    return res
end

covariances = covariance_matrix(data; ref_team)

sigma2 = [_m.σ^2 for _m in md]
@assert all(isapprox.(sigma2, diag(covariances), atol=0.0001))
copulas = all_copulas(md, covariances)

target_diff = 8.6
F = eval_cdf(md, copulas, fill(target_diff, 2))
P_copula = 1 .- F

t1, t2 = Tuple(argmax(P_copula))
win_pct_cop = P_copula[t1, t2]
println("Analytic solution with copula")
println("Team 1:")
println(data.Name[[all_teams[t1]...]])
println("Team 2:")
println(data.Name[[all_teams[t2]...]])
println("Expected win rate: " * string(round(win_pct_cop * 100, digits=2)) * "%")