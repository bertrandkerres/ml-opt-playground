#=
Realistic objective: charge the fleet handed over at the end of the year

Load after the simulator section of insideopt_w6w42.jl. The last section (adaptive policies)
also needs the definitions of insideopt_w6w42_rollout.jl and insideopt_w6w42_qmodel.jl.

Counting only weeks 1-52 makes everything after the horizon free, so tuned policies skip late
overhauls and hand over old trucks. Instead, keep simulating for H more weeks under a fixed
standard policy I_ss, the same for every plan, and count those weeks too:

    cost(weeks 1-52 under this year's plan) + cost(weeks 53-52+H under I_ss)

The continuation's long-run cost per week times H is the same for every plan, so comparing
plans compares the value of the state at week 52 (queueing included). H must be long enough
for the fleet to forget that state.
=#

const H = 156      # continuation weeks; plan differences settle to ~2% (H = 52, 104, 156, 260: 221, 339, 371, 379)
const LIVES = 30   # lives per truck in the scenarios, enough for 4 years


# =============================================================================
# Objective
# =============================================================================
# this year's plan I (one age per truck, or I[a, w] for weeks 1-52), then I_ss from week 53 on
function with_continuation(I, I_ss; H = H)
    J = repeat(I_ss, 1, WEEKS + H)
    J[:, 1:WEEKS] .= I
    return J
end

realistic_cost(params::AbstractDataFrame, I, I_ss, L; H = H) =
    simulate_threaded(params, with_continuation(I, I_ss; H), L; weeks = WEEKS + H)

# plan with no planned overhauls becoming due after week wcut
cutoff(I, wcut) = [w <= wcut ? I[a] : 10^6 for a in eachindex(I), w in 1:WEEKS]


# =============================================================================
# Tuning
# =============================================================================
"""
    coordinate_descent(f, I0; lb = I_MIN, ub = I_MAX, steps = (-3, -2, -1, 1, 2, 3), rounds = 10)

Integer coordinate descent on the thresholds: move one truck's threshold by each of `steps`,
keep any improvement of `f`, until a full round brings none. `f` should use fixed scenarios
(common random numbers), or the noise decides.
"""
function coordinate_descent(f, I0; lb = I_MIN, ub = I_MAX, steps = (-3, -2, -1, 1, 2, 3), rounds = 10)
    I, best = copy(I0), f(I0)
    for _ in 1:rounds
        improved = false
        for a in eachindex(I), δ in steps
            J = copy(I)
            J[a] = clamp(I[a] + δ, lb, ub)
            J == I && continue
            c = f(J)
            c < best && ((I, best, improved) = (J, c, true))
        end
        improved || break
    end
    return I, best
end


# best thresholds for the puzzle as stated (weeks 1-52 only), ≈ 2096
I_cd = [31, 25, 23, 39, 34, 26, 22, 45, 41, 35, 27, 47, 37, 32, 39, 28, 34, 29, 30, 43, 25, 34, 38, 33]

# standard policy for the continuation: best static thresholds over 4 years
L_ss = make_scenarios(wbl_params, 2000, 21; lives = LIVES)
I_ss, _ = coordinate_descent(I -> simulate_threaded(wbl_params, I, L_ss; weeks = 4 * WEEKS), I_cd)

# this year's plan, re-tuned on the realistic objective
L_tune = make_scenarios(wbl_params, 5000, 11; lives = LIVES)
I_real, _ = coordinate_descent(I -> realistic_cost(wbl_params, I, I_ss, L_tune), I_cd)

# fresh scenarios: both objectives for both plans
L_test = make_scenarios(wbl_params, 50_000, 12; lives = LIVES)
cost_52 = Dict(name => simulate_threaded(wbl_params, I, L_test) for (name, I) in ("I_cd" => I_cd, "I_real" => I_real))
cost_real = Dict(name => realistic_cost(wbl_params, I, I_ss, L_test) for (name, I) in ("I_cd" => I_cd, "I_real" => I_real))


# =============================================================================
# Adaptive policies on the realistic objective
# =============================================================================
# Decisions in weeks 1-52 only; from week 53 on every policy hands over to the same I_ss.
# Runtime: about 2 min for the Q-model data, 7 + 15 min for the two rollouts.
plan_real = with_continuation(I_real, I_ss)
weeks_real = WEEKS + H

# Q-model on rollout data from 500 scenarios; fewer Sobol points than for deciding, since the
# fit averages the noise out across decisions
φ_real, y_real, group_real, week_real = rollout_data(wbl_params, plan_real,
    make_scenarios(wbl_params, 500, 3; lives = LIVES); weeks = weeks_real, last_week = WEEKS, M = 512)
θ_real = fit_advantage(φ_real[group_real .<= 400], y_real[group_real .<= 400])
r2_real = r2(θ_real, φ_real[group_real .> 400], y_real[group_real .> 400])  # 0.72

# paired differences to the fixed plan. The Q policy is cheap enough for 50k scenarios; over
# 208 weeks the paired noise is ~3x larger than over 52, so 2000 scenarios only bracket the rollout
L_q = make_scenarios(wbl_params, 50_000, 99; lives = LIVES)
Δ_q = q_costs(wbl_params, plan_real, L_q, θ_real; weeks = weeks_real, last_week = WEEKS) .-
      base_costs(wbl_params, plan_real, L_q; weeks = weeks_real)  # -20.5 ± 2.6

L_roll = make_scenarios(wbl_params, 2000, 2; lives = LIVES)
c_fixed = base_costs(wbl_params, plan_real, L_roll; weeks = weeks_real)
Δ_shortlist = rollout_costs(wbl_params, plan_real, L_roll; weeks = weeks_real, last_week = WEEKS,
                            shortlist = q_shortlist(wbl_params, plan_real, θ_real)) .- c_fixed  # -25.7 ± 13.1
Δ_roll = rollout_costs(wbl_params, plan_real, L_roll; weeks = weeks_real, last_week = WEEKS) .- c_fixed  # +3.3 ± 14.0
