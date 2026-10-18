#=
Learned advantage model for the free-bay decisions (a minimal approximate policy iteration)

Load after the definitions of insideopt_w6w42_rollout.jl (and the simulator section of
insideopt_w6w42.jl before it). Keyword arguments such as `weeks` pass on to the rollout.

1. Run the rollout policy on training scenarios and log each decision: for every option, its
   simulated cost-to-go minus that of leaving the bays free, i.e. its advantage. With common
   random numbers this difference is much less noisy than either cost.
2. Fit a linear model of the advantage on hand-made features of the state and the option.
   Leaving the bays free has all features 0, so its advantage is 0 by construction.
3. Decide with the model: directly (no simulation at all), or by simulating only the options it
   ranks best (a shortlist for the rollout).
=#

using LinearAlgebra: Diagonal
using Statistics: mean

"""
    features(shape, scale, I, f, w, opt)

Features of starting overhauls in week `w` on the trucks in `opt` (one or two truck indices),
given the fleet `f`, the Weibull parameters and thresholds `I`. A named tuple, all 0 for
`opt = []` (leaving the bays free):

- `overhauls`: number of overhauls started
- `end_2w`, `end_6w`, `end_12w`, `end_20w`: `overhauls` if fewer than 2, 6, 12, 20 weeks of
  the plan (to week `WEEKS`) are left, else 0
- `pfail_2w`, `pfail_4w`, `pfail_8w`, `pfail_16w`, `pfail_end`: probability that the trucks in
  `opt` fail within 2, 4, 8, 16 weeks or before the end of the plan if left running, summed
- `due`: how many of them are already due
- `past_threshold`: their weeks past the threshold (negative: before it), summed
- `risk_2w`, `risk_4w`, `risk_8w`: `overhauls` × expected number of failures of the other
  running trucks within 2, 4, 8 weeks, which would compete for the bays
- `bays_full`: whether the option leaves no bay free
- `bays_full_risk_4w`: `bays_full` × expected failures of the others within 4 weeks
- `queue`: `overhauls` × number of other trucks already due
"""
function features(shape, scale, I, f, w, opt)
    # P(truck a fails within h weeks | survived to its current age)
    pfail(a, h) = -expm1((f.age[a] / scale[a])^shape[a] - ((f.age[a] + h) / scale[a])^shape[a])
    others = [a for a in eachindex(f.state) if is_running(f.state[a]) && a ∉ opt]
    pfail_own(h) = sum((pfail(a, h) for a in opt); init = 0.0)
    fails_others(h) = sum((pfail(a, h) for a in others); init = 0.0)

    overhauls = length(opt)
    left = WEEKS - w
    bays_full = nbusy(f) + overhauls == BAYS
    return (;
        overhauls,
        end_2w = overhauls * (left < 2),
        end_6w = overhauls * (left < 6),
        end_12w = overhauls * (left < 12),
        end_20w = overhauls * (left < 20),
        pfail_2w = pfail_own(2),
        pfail_4w = pfail_own(4),
        pfail_8w = pfail_own(8),
        pfail_16w = pfail_own(16),
        pfail_end = pfail_own(left),
        due = count(a -> f.state[a] == DUE, opt),
        past_threshold = sum((f.age[a] - threshold(I, a, w) for a in opt); init = 0.0),
        risk_2w = overhauls * fails_others(2),
        risk_4w = overhauls * fails_others(4),
        risk_8w = overhauls * fails_others(8),
        bays_full,
        bays_full_risk_4w = bays_full * fails_others(4),
        queue = overhauls * count(a -> f.state[a] == DUE, others),
    )
end

# predicted advantage of an option with features φ, for coefficients θ with the same names
predict(θ, φ) = sum(values(θ) .* values(φ))


# =============================================================================
# Data and fit
# =============================================================================
"""
    rollout_data(params, I, L; kw...)

Run the rollout policy on the scenarios in `L` and return the features `φs` (see `features`)
and simulated advantages `y` of every non-empty option at every decision, and the scenario and
week of each (together they identify the decision).
"""
function rollout_data(params::AbstractDataFrame, I, L; kw...)
    shape, scale = params.shape, params.scale
    rows = [[] for _ in axes(L, 3)]  # per scenario: thread-safe
    function on_decision(s, w, f, opts, values)
        @assert isempty(first(opts))
        for (opt, v) in zip(opts[2:end], values[2:end])
            push!(rows[s], (features(shape, scale, I, f, w, opt), v - values[1], w))
        end
    end
    rollout_costs(params, I, L; on_decision, kw...)
    φs = [φ for r in rows for (φ, _, _) in r]
    y = [v for r in rows for (_, v, _) in r]
    group = [s for (s, r) in enumerate(rows) for _ in r]
    week = [w for r in rows for (_, _, w) in r]
    return φs, y, group, week
end

"""
    fit_advantage(φs, y; ridge = 1e-6)

Least-squares fit of the advantages `y` on the features `φs`, with a small ridge since the
failure probabilities over different windows are nearly collinear (so the coefficients are not
effect sizes). Returns the coefficients as a named tuple with the features' names.
"""
function fit_advantage(φs, y; ridge = 1e-6)
    X = stack(Float64[values(φ)...] for φ in φs; dims = 1)
    θ = (X'X + ridge * Diagonal(ones(size(X, 2)))) \ (X'y)
    return NamedTuple{keys(first(φs))}(Tuple(θ))
end

r2(θ, φs, y) = 1 - sum(abs2, y .- predict.(Ref(θ), φs)) / sum(abs2, y .- mean(y))

# mean simulated regret, per decision, of picking the option the model ranks best (leaving the
# bays free has advantage 0) instead of the simulated best
function regret(θ, φs, y, group, week)
    pred = predict.(Ref(θ), φs)
    # rows of each decision, identified by its (scenario, week)
    decisions = Dict{Tuple{Int, Int}, Vector{Int}}()
    for (row, decision) in enumerate(zip(group, week))
        push!(get!(decisions, decision, Int[]), row)
    end
    regrets = map(collect(values(decisions))) do rows
        best_pred, pick = findmin(pred[rows])
        chosen = best_pred < 0 ? y[rows[pick]] : 0.0  # the model leaves the bays free otherwise
        best = min(0.0, minimum(y[rows]))
        chosen - best
    end
    return mean(regrets)
end


# =============================================================================
# Policies
# =============================================================================
advantage(θ, params, I, f, w, opt) =
    isempty(opt) ? 0.0 : predict(θ, features(params.shape, params.scale, I, f, w, opt))

# decide by the model alone
q_costs(params::AbstractDataFrame, I, L, θ; kw...) = policy_costs(
    (ctx, s, w, f, opts) -> argmin(opt -> advantage(θ, params, I, f, w, opt), opts),
    params, I, L; kw...)

# rollout shortlist: leaving the bays free, and the k options the model ranks best
q_shortlist(params::AbstractDataFrame, I, θ; k = 3) = (f, w, opts) ->
    pushfirst!(first(sort(opts[2:end]; by = opt -> advantage(θ, params, I, f, w, opt)), k), opts[1])


φs, y, group, week = rollout_data(wbl_params, I_base, make_scenarios(wbl_params, 500, 3))
train = group .<= 400
θ = fit_advantage(φs[train], y[train])
test = .!train
r2_test = r2(θ, φs[test], y[test])
regret_test = regret(θ, φs[test], y[test], group[test], week[test])

L = make_scenarios(wbl_params, 2000, 2)
c_base = base_costs(wbl_params, I_base, L)
c_q = q_costs(wbl_params, I_base, L, θ)
c_qroll = rollout_costs(wbl_params, I_base, L; shortlist = q_shortlist(wbl_params, I_base, θ))
