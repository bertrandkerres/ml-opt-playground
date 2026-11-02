#=
https://www.linkedin.com/pulse/trick-treat-meinolf-sellmann-u1hzc/

It’s Halloween in DS City. Your kid leaves the house at 6:00pm with an empty bag and Mom’s one rule:
"Be home by 7:30!" For every minute late, Mom confiscates 5 pieces of candy as a “late fee.” Tough love!!

The neighborhood has 15 houses with Halloween decoartions. Each one hands out a known amount of candy,
but only while its porch light is on, and you know only roughly when each light goes out. If you arrive
to a dark porch, you get nothing, but you also lose no time and simply walk on.

Walking is along the streets. A block takes between 1.6 and 4 minutes. Two minutes is most likely,
but kids in costume dawdle, so the average is about 2.5 minutes (a triangular distribution). Each walk
from one house to the next is independent. At a lit house you spend a few minutes ringing the bell,
shouting “Trick or treat!” and choosing your candy. The Haunted Garage always has a line.

Lights-out times are normally distributed.
=#

import CSV: read as read_csv
using DataFrames
using Distributions: Normal, TriangularDist
using Random
using Statistics: mean, quantile

import POMDPTools: ImplicitDistribution, Deterministic

struct Position
    x::Int
    y::Int
end

struct HouseSpec{LD, DD}
    id::Int
    candy::Int
    pos::Position
    lights_out_dist::LD
    door_time_dist::DD
end

struct House{T <: Real}
    candy::Int
    travel_time::T
    lights_out::T
    door_time::T
end

manhattan(pos::Position, origin::Position) = abs(pos.x - origin.x) + abs(pos.y - origin.y)

expected_travel_time(d::Real) = d * (1.6 + 4.0 + 2.0) / 3
expected_travel_time(pos::Position, origin::Position) = expected_travel_time(manhattan(pos, origin))

const BLOCK_TIME_DIST = TriangularDist(1.6, 4.0, 2.0)

function travel_time(rng::AbstractRNG, d::Integer)
    t = 0.0
    for _ in 1:d
        t += rand(rng, BLOCK_TIME_DIST)
    end
    return t
end
travel_time(rng::AbstractRNG, pos::Position, origin::Position) =
    travel_time(rng, manhattan(pos, origin))


function sample_next_house(rng::AbstractRNG, spec::HouseSpec, current_pos::Position)
    t = travel_time(rng, spec.pos, current_pos)
    return House(spec.candy, t, rand(rng, spec.lights_out_dist), rand(rng, spec.door_time_dist))
end

sample_next_house(spec::HouseSpec, current_pos::Position) =
    sample_next_house(Random.default_rng(), spec, current_pos)

positions = read_csv("data/puzzles/insideopt_26w44_positions.csv", DataFrame)
params = read_csv("data/puzzles/insideopt_26w44_params.csv", DataFrame)

function generate_house_specs(positions, params)
    df = innerjoin(positions, params, on = :Name)

    return [
        HouseSpec(
            i, h.Candy,
            Position(h.X, h.Y),
            Normal(h.LightsOutMean, h.LightsOutStd),
            TriangularDist(h.DoorMin, h.DoorMax, h.DoorMode),
        )
        for (i, h) in enumerate(eachrow(df))
    ]
end

function mean_house_specs(positions, params)
    df = innerjoin(positions, params, on = :Name)

    return [
        HouseSpec(
            i, h.Candy,
            Position(h.X, h.Y),
            Deterministic(h.LightsOutMean),
            Deterministic((h.DoorMin + h.DoorMax + h.DoorMode) / 3),
        )
        for (i, h) in enumerate(eachrow(df))
    ]
end

const HOME = Position(3, 0)

function rollout(rng::AbstractRNG, nodes::AbstractVector{Int}, house_specs; pos = HOME, reward = 0, t = 0.0)
    for n in nodes
        if n == 0
            break  # going home
        end

        next_house_spec = house_specs[n]
        expected_home = t + expected_travel_time(pos, HOME)
        if expected_home > 90.0
            # 5 candy / minute is a lot, so no need to compare
            break
        end

        h = sample_next_house(rng, next_house_spec, pos)
        t += h.travel_time
        pos = next_house_spec.pos
        if t > h.lights_out
            continue
        else
            t += h.door_time
            reward += h.candy
        end
    end
    t += travel_time(rng, pos, HOME)
    penalty = 5 * floor(Int, max(t - 90.0, 0.0))
    reward -= penalty
    return max(reward, 0), t
end

rollout(nodes::AbstractVector{Int}, house_specs; kwargs...) =
    rollout(Random.default_rng(), nodes, house_specs; kwargs...)

const HOUSE_SPECS = generate_house_specs(positions, params)
static_sol = [5, 4, 2, 3, 11, 6, 13, 12, 14]
rollout(static_sol, HOUSE_SPECS)


function simulate(rng::AbstractRNG, nodes::AbstractVector{Int}, house_specs; N = 1_000, kwargs...)
    rewards = [first(rollout(rng, nodes, house_specs; kwargs...)) for _ in 1:N]
    return mean(rewards)
end

simulate(nodes::AbstractVector{Int}, house_specs; kwargs...) =
    simulate(Random.default_rng(), nodes, house_specs; kwargs...)


#=================================================================================#
# Pre-sampled scenarios (sample average approximation / common random numbers)
#=================================================================================#
#=
A scenario fixes every random quantity up front:
- one travel time per directed pair (from, to), incl. HOME
- one lights-out time and one door time per house
=#
struct Scenarios
    travel::Array{Float64, 3}       # (16, 16, S): travel[to + 1, from + 1, s]
    lights_out::Matrix{Float64}     # (15, S)
    door_time::Matrix{Float64}      # (15, S)
    candy::Vector{Int}              # (15,)
    expected_home::Vector{Float64}  # (16,): expected travel time from node k to HOME
end

function Scenarios(rng::AbstractRNG, house_specs; S = 1_000)
    P = [HOME; [h.pos for h in house_specs]]
    travel = [travel_time(rng, manhattan(P[to], P[from])) for to in eachindex(P), from in eachindex(P), _ in 1:S]
    lights_out = [rand(rng, h.lights_out_dist) for h in house_specs, _ in 1:S]
    door_time = [rand(rng, h.door_time_dist) for h in house_specs, _ in 1:S]
    candy = [h.candy for h in house_specs]
    expected_home = [expected_travel_time(p, HOME) for p in P]
    return Scenarios(travel, lights_out, door_time, candy, expected_home)
end

Scenarios(house_specs; kwargs...) = Scenarios(Random.default_rng(), house_specs; kwargs...)

Base.length(sc::Scenarios) = size(sc.travel, 3)

function evaluate(nodes::AbstractVector{Int}, sc::Scenarios; start::Int = 0, reward::Int = 0, t::Float64 = 0.0)
    total = 0
    @inbounds for s in 1:length(sc)
        ts = t
        rs = reward
        cur = start
        for n in nodes
            if n == 0
                break  # going home
            end
            if ts + sc.expected_home[cur + 1] > 90.0
                break
            end
            ts += sc.travel[n + 1, cur + 1, s]
            cur = n
            if ts > sc.lights_out[n, s]
                continue
            end
            ts += sc.door_time[n, s]
            rs += sc.candy[n]
        end
        ts += sc.travel[1, cur + 1, s]
        rs -= 5 * floor(Int, max(ts - 90.0, 0.0))
        total += max(rs, 0)
    end
    return total / length(sc)
end


#=================================================================================#
# Offline solution
#=================================================================================#
using Metaheuristics

# search_space is [1, 16], domain for x is [0, 15]
search_space = PermutationSpace(0:15, 16)
const SCENARIOS = Scenarios(Xoshiro(42), HOUSE_SPECS; S = 1_000)
objective(x) = -evaluate(x, SCENARIOS)

algorithm = Metaheuristics.GA(
    crossover = OrderCrossover(),
    mutation = SlightMutation(),
    options = Metaheuristics.Options(iterations=200)
)
offline_result = optimize(objective, search_space, algorithm)
offline_perm = collect(Iterators.takewhile(!=(0), minimizer(offline_result)))
offline_reward = -minimum(offline_result)  # in-sample, biased upwards
simulated_offline_reward = simulate(offline_perm, HOUSE_SPECS; N = 10_000)  # out-of-sample

#=================================================================================#
# Online solution: MDP
#=================================================================================#
import POMDPs
using QuickPOMDPs: QuickPOMDP
using StaticArrays: MVector

#=
This MDP is completely observable; we don't need a belief state
State: position ∈ {0, ..., 15}, candy ∈ [0, ∞), time ∈ [0, ∞), visited ∈ {0, 1}^N :: Tuple{Int, Float64}
Action: next position ∈ {0, ..., 15} \ visited :: Int
Reward: Final candy in terminal state, else 0
=#

struct MDPState
    pos::Position
    candy::Int
    t::Float64
    visited::MVector{15, Bool}
end

function MDPState()
    MDPState(HOME, 0, 0.0, MVector{15, Bool}(falses(15)))
end

mdp = QuickPOMDP(;
    discount = 1.0,
    obstype = MDPState,

    actions = function (s = nothing)
        isnothing(s) && return 0:15
        return [0; findall(.!s.visited)]
    end,
    
    transition = function (s, a)
        ImplicitDistribution() do rng
            if a == 0
                dt = travel_time(rng, s.pos, HOME)
                return MDPState(HOME, s.candy, s.t + dt, s.visited)
            else
                next_house_spec = HOUSE_SPECS[a]
                h = sample_next_house(rng, next_house_spec, s.pos)
                t = s.t + h.travel_time
                visited = copy(s.visited)
                visited[a] = true
                candy = s.candy
                if t <= h.lights_out
                    t += h.door_time
                    candy += h.candy
                end
                return MDPState(h.pos, candy, t, visited)
            end
        end
    end,

    observation = (a, sp) -> sp,

    reward = function (s, a, sp)
        if sp.pos == HOME && sum(sp.visited) > 0
            penalty = 5 * floor(Int, max(sp.t - 90.0, 0.0))
            return max(sp.candy - penalty, 0)
        else
            return 0
        end
    end,
    
    initialstate = Deterministic(MDPState(HOME, 0, 0.0, MVector{15, Bool}(falses(15)))),
    isterminal = s -> s.pos == HOME && sum(s.visited) > 0 ? true : false,
)

#=================================================================================#
# Online solution: MPC
#=================================================================================#
function update_current_sol!(rng::AbstractRNG, sol::Vector{Int}, not_visited)
    circshift!(sol, -1)
    pick_from = setdiff(not_visited, sol)
    sol[end] = rand(rng, pick_from)
    return nothing
end

update_current_sol!(sol::Vector{Int}, not_visited) =
    update_current_sol!(Random.default_rng(), sol, not_visited)


#=
Planning reuses one scenario set for all steps: lights-out and door times of unvisited
houses are independent of everything observed so far, so their unconditional draws stay
valid, and the travel matrix already covers every starting node. The "real" walk in
`update_state` must use draws independent of the planning scenarios.
=#
function solve_next_step(
    state::MDPState, current_node::Int, scenarios::Scenarios, previous_sol; n_iter = 200
)
    algorithm = Metaheuristics.GA(
        crossover = OrderCrossover(),
        mutation = SlightMutation(),
        options = Metaheuristics.Options(iterations = n_iter)
    )
    f = x -> -evaluate(x, scenarios; start = current_node, reward = state.candy, t = state.t)

    not_visited = findall(.!state.visited)
    set_user_solutions!(algorithm, previous_sol, f)
    search_space = PermutationSpace([0; not_visited])
    res = optimize(f, search_space, algorithm)
    return res
end


function update_state(rng::AbstractRNG, state::MDPState, next_house_spec)
    h = sample_next_house(rng, next_house_spec, state.pos)
    t = state.t + h.travel_time
    candy = state.candy
    if t <= h.lights_out
        t += h.door_time
        candy += h.candy
    end

    visited = copy(state.visited)
    visited[next_house_spec.id] = true
    return MDPState(next_house_spec.pos, candy, t, visited)
end


# `rng` drives the real walk; planning uses `scenarios`, sampled from a separate stream.
function solve_as_mpc(
    rng::AbstractRNG, house_specs;
    scenarios = Scenarios(Xoshiro(rand(rng, UInt)), house_specs),
    initial_solution = randperm(rng, 16) .- 1, n_iter = 100,
)
    state = MDPState()
    res = solve_next_step(state, 0, scenarios, initial_solution; n_iter = 2n_iter)
    current_sol = minimizer(res)

    next_node = popfirst!(current_sol)
    while next_node != 0
        state = update_state(rng, state, house_specs[next_node])
        println("Visiting node $next_node, t = $(state.t), candy = $(state.candy)")

        res = solve_next_step(state, next_node, scenarios, current_sol; n_iter)
        current_sol = minimizer(res)
        next_node = popfirst!(current_sol)
    end
    t = state.t + travel_time(rng, state.pos, HOME)
    candy = state.candy - 5 * floor(Int, max(t - 90.0, 0.0))
    state = MDPState(HOME, candy, t, copy(state.visited))
    return (; state, res)
end

solve_as_mpc(house_specs::AbstractVector{<:HouseSpec}; kwargs ...) =
    solve_as_mpc(Random.default_rng(), house_specs; kwargs...)

# Takes a while
mpc_sol = [first(solve_as_mpc(HOUSE_SPECS)) for _ in 1:1000]
time_home = [s.t for s in mpc_sol]
reward = [s.candy for s in mpc_sol]

using Plots
histogram(reward)