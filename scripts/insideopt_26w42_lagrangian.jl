#=
Lagrangian relaxation of the bay capacity (weakly coupled MDP: Hawkins 2003, Adelman & Mersereau 2008)

Load after the simulator section of insideopt_w6w42.jl (uses its constants, `wbl_params`,
`make_scenarios` and `simulate_threaded`).

Charging λ[w] per bay-week instead of enforcing the 2 bays decouples the trucks:
- backward pass: each truck solves its own finite-horizon DP over (week, age)
- forward pass: propagating its state distribution under that policy gives its expected bay use
The dual D(λ) = Σ_trucks V_1(age0) - BAYS * Σ_w λ[w] is a lower bound on the expected cost of
any policy (static thresholds or fully dynamic), and its subgradient w.r.t. λ[w] is
(expected bay use in week w) - BAYS.
=#

const AMAX = 150  # age cap of the DP tables; a truck this old has failed with probability ~1

# P(failure while ageing from a to a+1 | survived to a); breakdowns happen in week ceil(life)
hazard(shape, scale, a) = -expm1((a / scale)^shape - ((a + 1) / scale)^shape)

# cost of a bay-week in week t; nothing is counted after the horizon
baycost(λ, t) = t <= WEEKS ? C_DOWN + λ[t] : 0.0


# =============================================================================
# Backward pass: per-truck DP
# =============================================================================
struct TruckPolicy
    overhaul::BitMatrix  # [w, a]: overhaul when reaching age a in week w without failing
    repair::BitVector    # [w]: a broken truck enters the bay in week w (else waits a week)
end

function solve_truck(shape, scale, λ)
    # V[w, a+1]: running at age a at the start of week w; R[w]: broken, bay decision in week w
    V = zeros(WEEKS + D_EMERG, AMAX + 1)
    R = zeros(WEEKS + 1)
    overhaul = falses(WEEKS, AMAX)
    repair = falses(WEEKS)
    for w in WEEKS:-1:1
        rep = sum(k -> baycost(λ, w + k), 0:D_EMERG-1) + V[w+D_EMERG, 1]
        repair[w] = rep <= C_DOWN + R[w+1]
        R[w] = min(rep, C_DOWN + R[w+1])
        ovh = C_PLAN + sum(k -> baycost(λ, w + k), 0:D_PLAN-1) + V[w+D_PLAN, 1]
        V[w, AMAX+1] = C_EMERG + R[w]
        for a in 0:AMAX-1
            q = hazard(shape, scale, a)
            cont = V[w+1, a+2]
            overhaul[w, a+1] = a + 1 >= I_MIN && ovh < cont
            V[w, a+1] = q * (C_EMERG + R[w]) + (1 - q) * min(cont, overhaul[w, a+1] ? ovh : cont)
        end
    end
    return TruckPolicy(overhaul, repair)
end

solve_trucks(params::AbstractDataFrame, λ) =
    [solve_truck(r.shape, r.scale, λ) for r in eachrow(params)]

# the puzzle's policy class: overhaul at age I in every week, repair immediately
threshold_policy(I::Integer) = TruckPolicy(repeat((1:AMAX)' .>= I, WEEKS), trues(WEEKS))


# =============================================================================
# Forward pass: state distribution of one truck
# =============================================================================
"""
    bay_use(shape, scale, age0, pol, λ)

Expected bay use per week of a truck following `pol` (with unlimited bays), and its expected
cost with bay-weeks charged at `λ`. With `λ = 0` this is the truck's cost without queueing.
"""
function bay_use(shape, scale, age0::Integer, pol::TruckPolicy, λ)
    use = zeros(WEEKS + D_EMERG)
    restart = zeros(WEEKS + D_EMERG + 1)  # probability of running again at age 0 from week t
    p = zeros(AMAX + 1)                   # p[a+1]: running at age a at the start of the week
    p[age0+1] = 1.0
    broken = 0.0                          # probability of being broken, waiting for the bay
    cost = 0.0
    for w in 1:WEEKS
        pn = zeros(AMAX + 1)
        for a in 0:AMAX
            m = p[a+1]
            m == 0 && continue
            q = a < AMAX ? hazard(shape, scale, a) : 1.0
            broken += q * m
            cost += q * m * C_EMERG
            s = (1 - q) * m
            (s == 0 || a == AMAX) && continue
            if pol.overhaul[w, a+1]
                cost += s * (C_PLAN + sum(k -> baycost(λ, w + k), 0:D_PLAN-1))
                use[w:w+D_PLAN-1] .+= s
                restart[w+D_PLAN] += s
            else
                pn[a+2] += s
            end
        end
        if pol.repair[w]
            cost += broken * sum(k -> baycost(λ, w + k), 0:D_EMERG-1)
            use[w:w+D_EMERG-1] .+= broken
            restart[w+D_EMERG] += broken
            broken = 0.0
        else
            cost += broken * C_DOWN
        end
        pn[1] += restart[w+1]
        p = pn
    end
    return use[1:WEEKS], cost
end

# summed over all trucks
function bay_use(params::AbstractDataFrame, pols, λ)
    use = zeros(WEEKS)
    cost = 0.0
    for (r, pol) in zip(eachrow(params), pols)
        u, c = bay_use(r.shape, r.scale, round(Int, r.age0), pol, λ)
        use .+= u
        cost += c
    end
    return use, cost
end


# =============================================================================
# Dual ascent on λ
# =============================================================================
"""
    lagrangian(params; policies = solve_trucks, iters = 1000, step = 2.0)

Projected subgradient ascent on the bay prices λ, where `policies(params, λ)` returns each
truck's best policy at prices λ. Returns the best dual value found, the λ that attains it, and
the λ averaged over the second half of the iterations. The dual value is a lower bound on the
expected cost of any policy only for `policies = solve_trucks` (all policies allowed).
"""
function lagrangian(params::AbstractDataFrame; policies = solve_trucks, iters = 1000, step = 2.0)
    λ = zeros(WEEKS)
    λsum = zeros(WEEKS)
    bound, λbest = -Inf, copy(λ)
    for t in 1:iters
        use, cost = bay_use(params, policies(params, λ), λ)
        D = cost - BAYS * sum(λ)
        D > bound && ((bound, λbest) = (D, copy(λ)))
        t > iters ÷ 2 && (λsum .+= λ)
        λ .= max.(0.0, λ .+ step / sqrt(t) .* (use .- BAYS))
    end
    return (; bound, λ = λbest, λavg = λsum ./ (iters - iters ÷ 2))
end


# =============================================================================
# Policies for the simulator
# =============================================================================
# best static threshold per truck when bay-weeks are charged at λ (still ignores queueing)
static_thresholds(params::AbstractDataFrame, λ) = [
    argmin(I -> last(bay_use(r.shape, r.scale, round(Int, r.age0), threshold_policy(I), λ)), I_MIN:I_MAX)
    for r in eachrow(params)
]
static_policies(params::AbstractDataFrame, λ) = threshold_policy.(static_thresholds(params, λ))

# week-dependent thresholds I[a, w] for `simulate_threaded`: lowest age the DP overhauls at
function dynamic_thresholds(pols)
    I = fill(10^6, length(pols), WEEKS)
    for (a, pol) in enumerate(pols), w in 1:WEEKS
        age = findfirst(view(pol.overhaul, w, :))
        isnothing(age) || (I[a, w] = age)
    end
    return I
end


# all policies: res_dyn.bound is a lower bound on the expected cost
res_dyn = lagrangian(wbl_params)
I_dyn = dynamic_thresholds(solve_trucks(wbl_params, res_dyn.λavg))

# restricted to the puzzle's static thresholds (no longer a bound: queueing is ignored)
res_static = lagrangian(wbl_params; policies = static_policies, iters = 200)
I_static = static_thresholds(wbl_params, res_static.λavg)

L = make_scenarios(wbl_params, 20_000, 1)
cost_static = simulate_threaded(wbl_params, I_static, L)
cost_dyn = simulate_threaded(wbl_params, I_dyn, L)
