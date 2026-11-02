#=
Rollout policy: adapt the planned overhauls to what has happened so far

Load after the simulator section of insideopt_w6w42.jl (uses its constants, `wbl_params`,
`TruckBuffers`, `next_in_queue`, `threshold`, `make_scenarios` and `scenario_cost!`).
The thresholds `I` can be one age per truck or week-dependent `I[a, w]`, and `weeks` sets the
horizon, e.g. this year's plan followed by a continuation (see insideopt_w6w42_terminal.jl).

Every week, once broken trucks have taken their bays, a still-free bay is a decision: leave it
free, or start an overhaul on one (or two) of the trucks close to their threshold. Each option
is simulated to the horizon under the static-threshold base policy, on the same M scenarios
sampled from the current state (common random numbers), and the cheapest is carried out. The
scenarios only use what is known at that point: current ages and states, remaining lives drawn
from the Weibull conditioned on each truck's current age, fresh lives after a renewal. They come
from one Sobol point set, randomised per decision by a random digital shift (XOR of the 32-bit
digits), which keeps the point set's structure and the estimates unbiased.
=#

using Random
import QuasiMonteCarlo: sample as qmc_sample, SobolSample, NoRand


# =============================================================================
# Simulator from any state
# =============================================================================
# The fleet state is a `TruckBuffers`; k[a] indexes the lives matrix `lives[a, k]`, where k = 1
# is the life running at the start (as in L[:, :, s]). The three phases of a week follow
# `scenario_cost!`, and each returns its cost.

function init!(f::TruckBuffers, age0, lives)
    for a in eachindex(age0)
        f.age[a] = age0[a]
        f.life[a] = lives[a, 1]
        f.k[a] = 1
        f.state[a] = RUNNING
        f.remain[a] = 0
        f.qtime[a] = 0
    end
    return f
end

function copy_fleet!(g::TruckBuffers, f::TruckBuffers)
    for name in fieldnames(TruckBuffers)
        copyto!(getfield(g, name), getfield(f, name))
    end
    return g
end

nbusy(f) = count(in_bay, f.state)

function start_overhaul!(f, a)
    f.state[a] = OVERHAUL
    f.remain[a] = D_PLAN
    return C_PLAN
end

# phase 1: ageing, breakdowns, overhauls becoming due at age I[a] (or I[a, w])
function age!(f, w, I)
    cost = 0.0
    @inbounds for a in eachindex(f.age)
        is_running(f.state[a]) || continue
        f.age[a] += 1
        f.life[a] -= 1
        if f.life[a] <= 0
            f.state[a] = WAITING
            f.qtime[a] = w
            cost += C_EMERG
        elseif f.state[a] == RUNNING && f.age[a] >= threshold(I, a, w)
            f.state[a] = DUE
            f.qtime[a] = w
        end
    end
    return cost
end

# phase 2: free bays go to breakdowns first, then to due overhauls unless `planned = false`
function assign!(f, planned::Bool)
    cost = 0.0
    busy = nbusy(f)
    while busy < BAYS
        a = next_in_queue(f.state, f.qtime)
        (a == 0 || (!planned && f.state[a] == DUE)) && break
        if f.state[a] == WAITING
            f.state[a] = REPAIR
            f.remain[a] = D_EMERG
        else
            cost += start_overhaul!(f, a)
        end
        busy += 1
    end
    return cost
end

# phase 3: downtime and job completion; a renewed truck starts its next life from `lives`
function finish!(f, lives)
    cost = 0.0
    @inbounds for a in eachindex(f.state)
        st = f.state[a]
        is_down(st) && (cost += C_DOWN)
        if in_bay(st)
            f.remain[a] -= 1
            if f.remain[a] == 0
                f.state[a] = RUNNING
                f.age[a] = 0
                f.k[a] = min(f.k[a] + 1, size(lives, 2))
                f.life[a] = lives[a, f.k[a]]
            end
        end
    end
    return cost
end

# base policy (thresholds I) from the start of week w0 to the end of week `weeks`
function run_base!(f, w0, I, lives, weeks = WEEKS)
    cost = 0.0
    for w in w0:weeks
        cost += age!(f, w, I) + assign!(f, true) + finish!(f, lives)
    end
    return cost
end


# =============================================================================
# Rollout
# =============================================================================
# M Sobol points in na * nlives dimensions as 32-bit digits; the first na dimensions (the best
# spread ones) drive the lives running now, which matter most
sobol_digits(M, na, nlives = KLIFE) =
    floor.(UInt32, qmc_sample(M, na * nlives, SobolSample(R = NoRand())) .* 2.0^32)

# M scenarios of the future given the current fleet: lives[a, 1, m] is the remaining life of a
# running truck, conditioned on its current age; lives[a, 2:end, m] are fresh lives.
# Point m of the digitally shifted Sobol set P gives scenario m.
function sample_lives!(lives, f, shape, scale, P, shift, rng)
    rand!(rng, shift)
    na = size(lives, 1)
    for m in axes(lives, 3), k in axes(lives, 2), a in 1:na
        d = (k - 1) * na + a
        u = (xor(P[d, m], shift[d]) + 0.5) / 2.0^32
        lives[a, k, m] = k == 1 && is_running(f.state[a]) ?
            scale[a] * ((f.age[a] / scale[a])^shape[a] - log(u))^(1 / shape[a]) - f.age[a] :
            scale[a] * (-log(u))^(1 / shape[a])
    end
    return lives
end

# options for the free bays in week w: nothing, or overhauls among the K trucks nearest to
# their threshold (due trucks first, in queue order; others at most `window` weeks short of it)
function options(f, I, w, free; window = 6, K = 6)
    trucks = [a for a in eachindex(f.state)
              if is_running(f.state[a]) && f.age[a] >= max(I_MIN, threshold(I, a, w) - window)]
    sort!(trucks; by = a -> f.state[a] == DUE ? (0, f.qtime[a]) : (1, threshold(I, a, w) - f.age[a]))
    trucks = first(trucks, K)
    opts = [Int[]]
    append!(opts, [[a] for a in trucks])
    free >= 2 && append!(opts, [[a, b] for (i, a) in enumerate(trucks) for b in trucks[i+1:end]])
    return opts
end

# mean cost from the decision in week w to the end of week `weeks` when overhauling `opt` now
function rollout_cost!(g, f, w, opt, I, lives, weeks = WEEKS)
    total = 0.0
    for m in axes(lives, 3)
        lv = view(lives, :, :, m)
        copy_fleet!(g, f)
        for a in eachindex(g.k)
            g.k[a] = 1
            is_running(g.state[a]) && (g.life[a] = lv[a, 1])
        end
        total += sum(a -> start_overhaul!(g, a), opt; init = 0.0)
        total += finish!(g, lv) + run_base!(g, w + 1, I, lv, weeks)
    end
    return total / size(lives, 3)
end

"""
    policy_costs(decide, params, I, L; context = s -> nothing, weeks = WEEKS, last_week = WEEKS)

Cost over `weeks` weeks in each scenario of `L` (which plays the true lives) of the policy
that, whenever a bay is free after the breakdowns, picks one of `options(...)` by
`decide(ctx, s, w, f, opts)`; `ctx = context(s)` holds per-scenario buffers. From week
`last_week + 1` on, and for the overhaul ages that become due, the thresholds `I` apply.
"""
function policy_costs(decide, params::AbstractDataFrame, I, L; context = s -> nothing, weeks = WEEKS,
                      last_week = WEEKS)
    na, n = nrow(params), size(L, 3)
    costs = zeros(n)
    Threads.@threads :dynamic for s in 1:n
        ctx = context(s)
        f = TruckBuffers(na)
        truth = view(L, :, :, s)
        init!(f, params.age0, truth)
        cost = 0.0
        for w in 1:weeks
            cost += age!(f, w, I)
            if w > last_week
                cost += assign!(f, true)
            else
                cost += assign!(f, false)
                free = BAYS - nbusy(f)
                opts = free > 0 ? options(f, I, w, free) : [Int[]]
                if length(opts) > 1
                    best = decide(ctx, s, w, f, opts)
                    cost += sum(a -> start_overhaul!(f, a), best; init = 0.0)
                end
            end
            cost += finish!(f, truth)
        end
        costs[s] = cost
    end
    return costs
end

"""
    rollout_costs(params, I, L; M = 1024, weeks = WEEKS, last_week = WEEKS, seed = 1,
                  shortlist = (f, w, opts) -> opts, on_decision = nothing)

Cost of the rollout policy (base policy: thresholds `I`) over `weeks` weeks in each scenario of
`L`. Each decision simulates the options in `shortlist(f, w, opts)` to the end of week `weeks`
on `M` scenarios of its own (a power of 2, for the Sobol points), with as many lives per truck
as `L` has. Small M makes noisy choices among the options: with M = 100 iid
scenarios the rollout was worse than the base policy before week 40. `on_decision(s, w, f,
opts, values)`, if given, is called with each decision's simulated costs-to-go (from any
thread, so it must be thread-safe).
"""
function rollout_costs(params::AbstractDataFrame, I, L; M = 1024, weeks = WEEKS, last_week = WEEKS,
                       seed = 1, shortlist = (f, w, opts) -> opts, on_decision = nothing)
    ispow2(M) || throw(ArgumentError("M must be a power of 2"))
    (; shape, scale) = params
    na = nrow(params)
    P = sobol_digits(M, na, size(L, 2))
    context(s) = (; rng = Xoshiro(hash((seed, s))), g = TruckBuffers(na),
                    lives = zeros(na, size(L, 2), M), shift = zeros(UInt32, size(P, 1)))
    function decide(ctx, s, w, f, opts)
        opts = shortlist(f, w, opts)
        length(opts) == 1 && return only(opts)
        sample_lives!(ctx.lives, f, shape, scale, P, ctx.shift, ctx.rng)
        values = [rollout_cost!(ctx.g, f, w, opt, I, ctx.lives, weeks) for opt in opts]
        isnothing(on_decision) || on_decision(s, w, f, opts, values)
        return opts[argmin(values)]
    end
    return policy_costs(decide, params, I, L; context, weeks, last_week)
end

base_costs(params::AbstractDataFrame, I, L; weeks = WEEKS) =
    [scenario_cost!(TruckBuffers(nrow(params)), params.age0, I, L, s, weeks) for s in axes(L, 3)]


# static thresholds from coordinate descent, ≈ 2096 with the puzzle's simulator
I_base = [31, 25, 23, 39, 34, 26, 22, 45, 41, 35, 27, 47, 37, 32, 39, 28, 34, 29, 30, 43, 25, 34, 38, 33]

L = make_scenarios(wbl_params, 1000, 2)
c_base = base_costs(wbl_params, I_base, L)
c_roll = rollout_costs(wbl_params, I_base, L)
# paired difference: same true lives for both policies
Δ = c_roll .- c_base
