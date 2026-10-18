#=
Description: https://www.linkedin.com/pulse/bathtub-curve-ball-meinolf-sellmann-bc3tc/
Data: https://colab.research.google.com/drive/1pB8__6Jt7Cl_IYi49MoOC7-XR30pjeKS#scrollTo=kifhTWK7PytT
=#

using Plots
using Random
using DataFrames
import CSV: read as read_csv
import Distributions: Uniform, Weibull

const BAYS = 2
const WEEKS = 52
const KLIFE = 6

const C_EMERG = 25.0  # kEUR: breakdown repair, overhaul, lost margin per truck-week out of service
const C_PLAN = 5.0
const C_DOWN = 15.0

const D_EMERG = 4  # weeks of bay time
const D_PLAN = 2

const I_MIN = 8  # allowed overhaul ages (weeks); 60 is practically "never"
const I_MAX = 60


# one row per truck: Weibull shape and scale, and age0 = weeks it has already been running
wbl_params = read_csv("data/puzzles/insideopt_26w42_wbl_params.csv", DataFrame)


# =============================================================================
# Scenarios
# =============================================================================
# Lifetimes are stored as L[a, k, s]: truck a, k-th life, scenario s
# (the Python version uses L[s, a, k]; this order keeps one scenario contiguous)
function make_scenarios(params::AbstractDataFrame, U::AbstractArray{<:Real, 3})
    (; shape, scale, age0) = params
    size(U, 1) == length(age0) || throw(DimensionMismatch("U must have one row per truck"))
    L = @. scale * (-log(U))^(1 / shape)
    # a truck that has survived age0 weeks has a shorter expected remaining life
    @views @. L[:, 1, :] = scale * ((age0 / scale)^shape - log(U[:, 1, :]))^(1 / shape) - age0
    return L
end

# `lives` per truck and scenario; a truck that runs out keeps reusing its last one
make_scenarios(rng::AbstractRNG, params::AbstractDataFrame, n::Integer; lives = KLIFE) =
    make_scenarios(params, rand(rng, nrow(params), lives, n))
make_scenarios(params::AbstractDataFrame, n::Integer, seed::Integer; lives = KLIFE) =
    make_scenarios(Xoshiro(seed), params, n; lives)
make_scenarios(params::AbstractDataFrame, n::Integer; lives = KLIFE) =
    make_scenarios(params, rand(nrow(params), lives, n))


# =============================================================================
# Simulation
# =============================================================================
@enum TruckState RUNNING WAITING REPAIR DUE OVERHAUL
# RUNNING:  running
# WAITING:  broken down, waiting for a bay
# REPAIR:   broken down, in bay
# DUE:      overhaul due, still running
# OVERHAUL: overhaul in bay

in_bay(st::TruckState) = st == REPAIR || st == OVERHAUL
is_down(st::TruckState) = st == WAITING || in_bay(st)
is_running(st::TruckState) = st == RUNNING || st == DUE

struct TruckBuffers
    age::Vector{Float64}
    life::Vector{Float64}
    k::Vector{Int}
    state::Vector{TruckState}
    remain::Vector{Int}
    qtime::Vector{Int}
end
TruckBuffers(na::Integer) = TruckBuffers(
    zeros(na), zeros(na), ones(Int, na), fill(RUNNING, na), zeros(Int, na), zeros(Int, na)
)

# Next truck to enter a free bay: breakdowns first (FIFO), then due overhauls (FIFO);
# ties go to the lowest index. Returns 0 if nobody is queued.
function next_in_queue(state, qtime)
    best = 0
    bestkey = (true, typemax(Int))
    @inbounds for a in eachindex(state, qtime)
        st = state[a]
        (st == WAITING || st == DUE) || continue
        key = (st == DUE, qtime[a])
        if key < bestkey
            bestkey = key
            best = a
        end
    end
    return best
end

# overhaul threshold of truck a in week w: static I[a], or week-dependent I[a, w]
threshold(I::AbstractVector, a, w) = I[a]
threshold(I::AbstractMatrix, a, w) = I[a, w]

function scenario_cost!(buf::TruckBuffers, age0, I, L, s, weeks = WEEKS)
    (; age, life, k, state, remain, qtime) = buf
    na = length(age0)
    @inbounds for a in 1:na
        age[a] = age0[a]
        life[a] = L[a, 1, s]
        k[a] = 1
        state[a] = RUNNING
        remain[a] = 0
        qtime[a] = 0
    end

    cost = 0.0
    busy = 0    # trucks in a bay
    queued = 0  # trucks waiting for a bay (broken down or overhaul due)
    @inbounds for w in 1:weeks
        # ageing, breakdowns, overhaul becomes due
        for a in 1:na
            is_running(state[a]) || continue
            age[a] += 1
            life[a] -= 1
            if life[a] <= 0
                state[a] == RUNNING && (queued += 1)
                state[a] = WAITING
                qtime[a] = w
                cost += C_EMERG
            elseif state[a] == RUNNING && age[a] >= threshold(I, a, w)
                state[a] = DUE
                qtime[a] = w
                queued += 1
            end
        end

        # free bays
        while busy < BAYS && queued > 0
            best = next_in_queue(state, qtime)
            queued -= 1
            if state[best] == WAITING
                state[best] = REPAIR
                remain[best] = D_EMERG
            else
                state[best] = OVERHAUL
                remain[best] = D_PLAN
                cost += C_PLAN
            end
            busy += 1
        end

        # downtime and job completion
        for a in 1:na
            st = state[a]
            is_down(st) && (cost += C_DOWN)
            if in_bay(st)
                remain[a] -= 1
                if remain[a] == 0
                    busy -= 1
                    state[a] = RUNNING
                    age[a] = 0
                    k[a] = min(k[a] + 1, size(L, 2))
                    life[a] = L[a, k[a], s]
                end
            end
        end
    end
    return cost
end

"""
    simulate(params, I, L; weeks = WEEKS)

Expected cost over `weeks` weeks when truck `a` is overhauled once it reaches age `I[a]` (or
`I[a, w]` in week `w`), averaged over the scenarios in `L` (see `make_scenarios`). Scenarios
are run in parallel. Only `params.age0` is used; it can also be passed directly as a vector.
"""
simulate_threaded(params::AbstractDataFrame, I, L; kw...) = simulate_threaded(params.age0, I, L; kw...)

function check_sizes(na, I, L, weeks)
    size(I, 1) == na || throw(DimensionMismatch("I must have $na rows"))
    I isa AbstractMatrix && size(I, 2) < weeks &&
        throw(DimensionMismatch("I must have a column for each of the $weeks weeks"))
    size(L, 1) == na || throw(DimensionMismatch("L must have $na rows"))
end

function simulate_threaded(age0::AbstractVector{<:Real}, I::AbstractVecOrMat{<:Real}, L::AbstractArray{Float64, 3}; weeks = WEEKS)
    na = length(age0)
    check_sizes(na, I, L, weeks)
    n = size(L, 3)
    chunks = Iterators.partition(1:n, cld(n, Threads.nthreads()))
    tasks = map(chunks) do idx
        Threads.@spawn begin
            buf = TruckBuffers(na)
            sum(s -> scenario_cost!(buf, age0, I, L, s, weeks), idx)
        end
    end
    # all costs are multiples of 5, so the sum is exact regardless of order
    return sum(t -> fetch(t)::Float64, tasks) / n
end


function simulate(age0::AbstractVector{<:Real}, I::AbstractVecOrMat{<:Real}, L::AbstractArray{Float64, 3}; weeks = WEEKS)
    na = length(age0)
    check_sizes(na, I, L, weeks)
    n = size(L, 3)
    results = [scenario_cost!(TruckBuffers(na), age0, I, L, s, weeks) for s in 1:n]
    return sum(results)
end

simulate(params::AbstractDataFrame, I, L; kw...) = simulate(params.age0, I, L; kw...)


using Metaheuristics

function objective_threaded(x, wbl_params; N = 10_000)
    L = make_scenarios(wbl_params, N)
    return simulate_threaded(wbl_params, x, L)
end

function objective(x, wbl_params; N = 10_000)
    L = make_scenarios(wbl_params, N)
    return simulate_threaded(wbl_params, x, L)
end


function objective_parallel(X, wbl_params; N = 10_000)
    res = zeros(size(X, 1))
    Threads.@threads for i in axes(X, 1)
        res[i] = objective(X[i, :], wbl_params; N)
    end
    return res
end


options = Options(f_calls_limit = 100_000, parallel_evaluation=true)
algo = ECA(options = options)
# Continuous search space
# (there seems to be a bug for parallel evals of discrete search spaces)
search_space = boxconstraints(
    lb = fill(Float64(I_MIN), size(wbl_params, 1)),
    ub = fill(Float64(I_MAX), size(wbl_params, 1))
)

res_cont = optimize(x -> objective_parallel(x, wbl_params; N = 500), search_space, algo)
obj_cont = minimum(res_cont)
I_best = round.(Int, minimizer(res_cont))

# Now we have a good solution in continuous space, and can do a search among
# the integers in a narrow band +- 3
discrete_search_space = boxconstraints(
    lb = max.(floor.(Int, I_best) .- 3, I_MIN),
    ub = min.(ceil.(Int, I_best) .+ 3, I_MAX)
)
local_algo = ECA(
    options = Options(f_calls_limit = 5_000),
)
objective_function(x) = objective(x, wbl_params; N = 5000)
# set_user_solutions!(local_algo, I_best, objective_function)
res_disc = optimize(objective_function, discrete_search_space, local_algo)

I_best_disc = minimizer(res_disc)
obj_disc = objective(I_best_disc, wbl_params; N = 20_000)