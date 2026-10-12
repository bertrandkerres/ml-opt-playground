#=
Assume we can choose to work on ten different tasks. Each task has a fixed profit but an 
uncertain demand on the work hours involved. We are allowed to have overtime and work more 
than the available 267 regular hours. However, if we have overtime, no matter how little, 
we pay $100, and another $100 for every hour of overtime (or fractions thereof). 

The actual work hours for each task we choose to take on are Gamma distributed as shown 
below (the two parameters (1,1) give shape and scale of the Gamma distribution).

What tasks 0 - 9 should we accept to maximize our expected profit?

Max 20x₀ + 30x₁ + 45x₂ + 53x₃ + 58x₄ + 74x₅ + 78x₆ + 79x₇ + 95x₈ + 98x₉ - 100V - 100O

s.t. W = a₀x₀ + a₁x₁ + a₂x₂ + a₃x₃ + a₄x₄ + a₅x₅ + a₆x₆ + a₇x₇ + a₈x₈ + a₉x₉
    O ≥ W - 267
    V ≥ 10⁻⁶O
    a₀ = 10 * (10 + Γ(1,1))/11 = 10/11 * 10 + 10/11 * Γ(1,1)
    a₁ = 22 * (10 + Γ(1,1))/11
    a₂ = 37 * (10 + Γ(1,1))/11
    a₃ = 45 * (10 + Γ(1,1))/11
    a₄ = 53 * (10 + Γ(1,1))/11
    a₅ = 65 * (10 + Γ(1,1))/11
    a₆ = 68 * (10 + Γ(1,1))/11
    a₇ = 68 * (10 + Γ(1,1))/11
    a₈ = 84 * (10 + Γ(1,1))/11
    a₉ = 88 * (10 + Γ(1,1))/11
    O ≥ 0
    V, xᵢ ∈ {0,1} for all i
=#

using Distributions
using LinearAlgebra
import Plots

A = [10, 22, 37, 45, 53, 65, 68, 68, 84, 88] ./ 11
C = [20, 30, 45, 53, 58, 74, 78, 79, 95, 98]

function moment_matching(x, a)
    E = dot(x, a)
    Var = dot(x, a.^2)
    theta = Var / E
    alpha = E / theta
    return alpha, theta
end

function objective(x, a, c)
    any(x .> 0) || return 0.0
    loc = 10 * dot(x, a)  # distribution shift from the (10 + ...) in a_i
    alpha, theta = moment_matching(x, a)
    D = Gamma(alpha, theta)
    W_realized = rand(D, 10000) .+ loc
    OTV_mean = expected_overtime_cost(W_realized)
    return -dot(x, c) + 100 * OTV_mean
end


function expected_overtime_cost(W)
    OTV_realized = zeros(size(W))
    is_OT = W .> 267.0
    OTV_realized[is_OT] .= W[is_OT] .- 167.0
    return mean(OTV_realized)
end

function mc_objective(x, a, c, G)
    gamma_realizations = rand(G, 10_000, 10) .+ 10
    W_realized = gamma_realizations * (a .* x)
    OTV_mean = expected_overtime_cost(W_realized)
    return -dot(x, c) + 100 * OTV_mean
end


import Metaheuristics

S = Metaheuristics.SearchSpaces.BitArraySpace(10)
Metaheuristics.cardinality(S)

alg = Metaheuristics.GA
res = Metaheuristics.optimize(x -> objective(x, A, C), S, alg)

res_mc = Metaheuristics.optimize(x -> mc_objective(x, A, C, Gamma(1, 1)), S, alg)
