#=
Task: https://www.linkedin.com/pulse/battery-management-meinolf-sellmann-m5zqc/
Data: https://colab.research.google.com/drive/1WWEgG3tyQ5Ybo_3kjfQHYgMwJg6po9Qz?usp=sharing

=#

using Plots
using Random
import DataFrames: DataFrame
import CSV: read as read_csv


# =============================================================================
# Random walk generators
# =============================================================================
function rwbias(x, k, a1, a2; tau = zero(x))
    dx = x .- tau
    return exp.(k) .* (exp.(-a1 .* dx) .- exp.(a2 .* dx))
end

# Clamp every step: the exponential restoring term overshoots for large |x|,
# so an unclamped walk can oscillate with growing amplitude until Inf - Inf = NaN
function brw(rng::AbstractRNG, sigma::T, N::Int;
    k = -5 * one(sigma), a1 = one(sigma), a2 = one(sigma), tau = zero(sigma),
    lo = typemin(sigma), hi = typemax(sigma)
) where T <: AbstractFloat
    eps = randn(rng, T, N-1)
    x = zeros(T, N)
    for i = 2:N
        x[i] = clamp(x[i-1] + rwbias(x[i-1], k, a1, a2; tau = tau) + sigma * eps[i-1], lo, hi)
    end
    return x
end

function brw(sigma::T, N::Int; kwargs...) where T <: AbstractFloat
    brw(Random.default_rng(), sigma, N; kwargs...)
end

# =============================================================================
# Battery data
# =============================================================================
const P_MAX = 50.0        # MW    power limit, charging and discharging (grid side)
const CAP = 100.0         # MWh   usable energy window: 0 <= SOC <= 100
const ETA_C = 0.95        # charging efficiency    (grid -> cell)
const ETA_D = 0.95        # discharging efficiency (cell -> grid)
const DT = 0.5            # h     length of a slot
const M = 48              # slots per day
const SOC0 = 50.0         # MWh   SOC at 00:00 and again at 24:00
const BETA = 2.0          # wear exponent: cycles to failure at depth DoD = 6000 / DoD**BETA
const C_FULL = 3000.0     # $     wear cost of one full-depth cycle ($18M pack / 6000 cycles)

const MAX_UP = P_MAX * DT * ETA_C      # largest SOC increase in one slot  (23.75 MWh)
const MAX_DOWN = P_MAX * DT / ETA_D    # largest SOC decrease in one slot  (26.32 MWh)

prices = read_csv("data/puzzles/insideopt_26w41_prices.csv", DataFrame)

# =============================================================================
# Rainflow counting
# =============================================================================
function reversals(x; tol = 1e-9)
    dx = abs.(diff(x))
    idx_dx_ge_tol = findall(>=(tol), dx) .+ 1
    x_nonrepeated = x[[1; idx_dx_ge_tol]]

    if length(x_nonrepeated) < 3
        return x_nonrepeated
    end

    dx_nr = sign.(diff(x_nonrepeated))
    idx_rev = [i+1 for i in 1:length(dx_nr)-1 if dx_nr[i] * dx_nr[i+1] < 0]
    return x_nonrepeated[[1; idx_rev; length(x_nonrepeated)]]
end

function rainflow_cycles(x::AbstractVector{T}) where T <: Number
    pts = T[]
    sizehint!(pts, length(x))

    res = Tuple{T, Float64}[]
    sizehint!(res, length(x) ÷ 2)

    count_half = 0.5
    count_full = 1.0
    n = 0  # Number of items in pts vector
    for p in reversals(x)
        push!(pts, p)
        n += 1
        while n >= 3
            x = abs(pts[end-1] - pts[end])
            y = abs(pts[end-2] - pts[end-1])
            if x < y
                break
            end
            if n == 3
                push!(res, (y, count_half))
                deleteat!(pts, 1)
                n -= 1
            else
                push!(res, (y, count_full))
                deleteat!(pts, (n-2, n-1))
                n -= 2
            end
        end
    end
    for (a, b) in zip(pts[1:end-1], pts[2:end])
        push!(res, (abs(a - b), count_half))
    end
    return res
end

function wear_cost(soc; beta = BETA)
    return C_FULL * sum(c * (r / CAP)^beta for (r, c) in rainflow_cycles(soc); init = 0.0)
end


function revenue(soc, prices; eta_c = ETA_C, eta_d = ETA_D)
    dsoc = diff(soc)
    discharge = max.(-dsoc, zero(eltype(soc)))
    charge = max.(dsoc, zero(eltype(soc)))
    return sum(prices .* (eta_d .* discharge .- charge ./ eta_c))
end

net_profit(soc, prices; beta = BETA) = revenue(soc, prices) - wear_cost(soc; beta)

@assert wear_cost([50, 90, 70, 100, 50]) ≈ 870.0


# =============================================================================
# Linear model, const wear cost
# =============================================================================
using JuMP
import HiGHS
import LinearAlgebra: dot
import Printf: @printf

function linear_model(price_vec)
    T = length(price_vec)
    linmdl = Model(HiGHS.Optimizer)

    @variable(linmdl, 0 <= soc[1:T+1] <= CAP)
    @variable(linmdl, 0 <= discharge[1:T] <= MAX_DOWN)
    @variable(linmdl, 0 <= charge[1:T] <= MAX_UP)

    @variable(linmdl, is_discharging[1:T], Bin)
    @variable(linmdl, is_charging[1:T], Bin)
    @constraint(linmdl, one_way[t = 1:T], is_charging[t] + is_discharging[t] <= 1)

    M = 100.0
    @constraint(linmdl, charge_link[t = 1:T], charge[t] <= is_charging[t] * M)
    @constraint(linmdl, discharge_link[t = 1:T], discharge[t] <= is_discharging[t] * M)

    @constraint(linmdl, initial, soc[1] == SOC0)
    @constraint(linmdl, intertemporal[t = 1:T], soc[t+1] == soc[t] + charge[t] - discharge[t])
    @constraint(linmdl, final, soc[T+1] == SOC0)

    @objective(linmdl, Max,
        dot(discharge, price_vec .* ETA_D)
      - dot(charge, price_vec ./ ETA_C)
      - C_FULL / (CAP * ETA_D) * sum(discharge)
    )

    return linmdl
end

linmdl = linear_model(prices[!, 1])
optimize!(linmdl)

SOC = value.(linmdl[:soc])
net_profit(SOC, prices[!, 1])
charge = value.(linmdl[:charge])
discharge = value.(linmdl[:discharge])
t = (0:M) / 2
plt1 = plot(t, SOC, label = "SOC [MWh]")
plot!(plt1, t[2:end], prices[!, 1], label = "Price [USD/MWh]")



# =============================================================================
# Exact rainflow wear as LP: play-operator (hysteresis) formulation
# =============================================================================
#=
The function in the wear cost
    phi(c) = const * (c / CAP)^beta
is convex. We can use a weighted sum of total variations (`tv`) instead.

Play path `z`:
    z_t = clamp(z_{t-1},  soc_t - r,  soc_t + r)
where `r` is the slack, i.e. stay within +- r of the actual SOC.

In the LP, play path k is any z_k with
    |z_k - soc| ≤ r_k.
i.e. is the `tube` constraint below.

The variable `tv` captures total variation, defined as
    tv_k_t ≥ | Δz_k_t | for all piecewise linear steps k, time steps t
Meaning:
* If the play path doesn't move between two time steps (Δz = 0), there's no
  total variation to add
* Else the change in play path will be the total variation. The inequality 
  is reduced to equality by the LP solver, because it tries to minimize costs.
* At each time step, the total wear is the cumulative sum of the total variation,
  weighted by second derivative of the phi function
=#


function play_weights(K; beta = BETA)
    phi(d) = C_FULL * (d / CAP)^beta
    d = range(0, CAP, length = K + 1)          # kinks of the PWL interpolant of phi
    slopes = diff(phi.(d)) ./ step(d)
    a = [slopes[1]; diff(slopes)]              # phi_hat(d) = sum_k a_k * (d - d_k)_+
    return d[1:K] ./ 2, a ./ 2                 # half-widths r_k, TV weights w_k
end

function play_model(price_vec; K = 100, beta = BETA)
    T = length(price_vec)
    r, w = play_weights(K; beta)

    playmdl = linear_model(price_vec)
    soc, charge, discharge = playmdl[:soc], playmdl[:charge], playmdl[:discharge]

    @variable(playmdl, z[1:K, 1:T+1])
    @variable(playmdl, tv[1:K, 1:T] >= 0)
    @constraint(playmdl, tube[k = 1:K, t = 1:T+1], -r[k] <= z[k, t] - soc[t] <= r[k])
    @constraint(playmdl, tv_up[k = 1:K, t = 1:T], tv[k, t] >= z[k, t+1] - z[k, t])
    @constraint(playmdl, tv_down[k = 1:K, t = 1:T], tv[k, t] >= z[k, t] - z[k, t+1])
    @expression(playmdl, wear, sum(w[k] * tv[k, t] for k in 1:K, t in 1:T))

    @objective(playmdl, Max,
        dot(discharge, price_vec .* ETA_D)
      - dot(charge, price_vec ./ ETA_C)
      - wear
    )

    return playmdl
end

playmdl = play_model(prices[!, 1])
optimize!(playmdl)

SOC_play = value.(playmdl[:soc])
@printf "Play LP:  objective %.2f, net_profit %.2f (wear LP %.2f vs rainflow %.2f)\n" objective_value(playmdl) net_profit(SOC_play, prices[!, 1]) value(playmdl[:wear]) wear_cost(SOC_play)
plt_play = plot(t, SOC_play, label = "SOC [MWh]")
plot!(plt_play, t[2:end], prices[!, 1], label = "Price [USD/MWh]")



# =============================================================================
# Some simple metaheuristics
# =============================================================================
import Metaheuristics

function constrained_objective(x, price_vec; eta_c = ETA_C, eta_d = ETA_D)
    soc = [SOC0; SOC0 .+ cumsum(x)]
    discharge = min.(x, zero(eltype(x)))
    charge = max.(x, zero(eltype(x)))

    # Discharge has negative sign but gives revenue, charge has pos sign but it costs
    rev = -sum(price_vec .* (eta_d .* discharge .+ charge ./ eta_c))
    c_wear = wear_cost(soc)
    obj = c_wear - rev # minimize

    # soc[1] is fixed at SOC0, so only the T free slots need bound constraints
    gx = [-soc[2:end]; soc[2:end] .- CAP]
    hx = [soc[end] - SOC0]
    return obj, gx, hx
end

function constrained_objective_parallel(X, price_vec; kwargs...)
    obj = zeros(size(X, 1))
    gx = zeros(size(X, 1), 2 * M)
    hx = zeros(size(X, 1), 1)
    Threads.@threads for i in axes(X, 1)
        obj[i], gx[i, :], hx[i, :] = constrained_objective(X[i, :], price_vec; kwargs...)
    end
    return obj, gx, hx
end


mh_objective(x) = constrained_objective(x, prices[!, 1])
mh_objective_parallel(x) = constrained_objective_parallel(x, prices[!, 1])

bounds = Metaheuristics.boxconstraints(
    lb = fill(-MAX_DOWN, M), ub = fill(MAX_UP, M)
)
options = Metaheuristics.Options(f_calls_limit = 10_000_000)
algo = Metaheuristics.ECA(options = options)
Metaheuristics.set_user_solutions!(algo, diff(SOC), mh_objective)
res = Metaheuristics.optimize(mh_objective, bounds, algo)

dSOC_mh = Metaheuristics.minimizer(res)
SOC_mh = 50.0 .+ [0.0; cumsum(dSOC_mh)]

plt_mh = plot(SOC_mh, label = "SOC [MWh]")
plot!(plt_mh, prices[!, 1], label = "Price [USD/MWh]")



# =============================================================================
# Train an NN on the wear cost
# =============================================================================
# Generate training data (synthetic SOC time series) using bounded random walk
using Lux, Optimisers, Enzyme, Reactant
using MLUtils
using Printf

rng = Xoshiro(1)
N = 1_000
lo, hi = -4.0, 4.0

unscale(soc_scaled, lo, hi) = CAP / 2 .+ CAP / (hi - lo) .* soc_scaled

function generate_data(rng::AbstractRNG, T::Type, N::Int; lo = -4.0, hi = 4.0, kwargs...)
    X = [brw(rng, one(T), M; lo = T(lo), hi = T(hi), kwargs...) for _ in 1:N]
    y = [T(wear_cost(unscale(soc, lo, hi))) for soc in X] ./ 1000
    return (X, y)
end

generate_data(T::Type, N::Int; kwargs...) = generate_data(Random.default_rng(), T, N; kwargs...)
generate_data(N::Int; kwargs...) = generate_data(Random.default_rng(), Float64, N; kwargs...)

X, y = generate_data(N)

X_plt = reduce(hcat, X[rand(1:N, min(10, N))])
plot(unscale(X_plt, lo, hi))



# =============================================================================
# Cycle attention
# =============================================================================
#=
Rainflow pairs every reversal i with one partner j; the pair is a cycle of depth
|x_i - x_j| if the path between them stays inside [min(x_i, x_j), max(x_i, x_j)].
Each time step attends over all others ("pick my cycle partner"), with scores and
values computed from pair features (endpoints, range, how far the path between them
over-/undershoots). Wear is additive over cycles, so the readout is a sum over tokens.
=#
const N_PAIR_FEATURES = 8

"""
    pair_features(x::AbstractMatrix{T}) -> Array{T, 4}

Describe every pair of time steps `(i, j)` of each SOC curve, as input for
`CycleAttention`, which uses them to decide which pairs are rainflow cycles.

Takes `x` of size `(M, B)` (one curve per column) and returns `P` of size
`(N_PAIR_FEATURES, M, M, B)`, where `P[f, j, i, b]` is feature `f` for token `i` and
candidate cycle partner `j` in sample `b`. With this layout, `reshape(P, N_PAIR_FEATURES, :)`
feeds a `Dense` layer directly, and a softmax over `dims = 2` distributes each token's
attention over its partners.

| f | Feature                   | Meaning                                              |
|---|---------------------------|------------------------------------------------------|
| 1 | `x_i`                     | value at the token                                   |
| 2 | `x_j`                     | value at the candidate partner                       |
| 3 | `abs(x_j - x_i)`          | cycle depth, if `(i, j)` is a cycle                  |
| 4 | `x_max - max(x_i, x_j)`   | how far the path between i and j rises above both    |
| 5 | `min(x_i, x_j) - x_min`   | how far the path between i and j drops below both    |
| 6 | `x_i - x_{i-1}`           | slope into i (0 at the start)                        |
| 7 | `x_{i+1} - x_i`           | slope out of i (0 at the end)                        |
| 8 | `(j - i) / M`             | signed time offset                                   |

`x_max` and `x_min` are the extrema of the path from i to j, both inclusive.
Features 4 and 5 carry the rainflow logic: `(i, j)` can only close a cycle if the path
between them stays inside `[min(x_i, x_j), max(x_i, x_j)]`, i.e. both are 0. Features 6
and 7 tell whether i is a peak, a valley or neither (only reversals start cycles).

The extrema come from walking away from i, forward and then backward, keeping a running
max/min, so the cost is O(M²) per curve, and `P` is the only allocation.

The features don't depend on model parameters, so they are computed on the CPU while
loading a batch: traced by Reactant, the scalar loops would be unrolled and compile slowly.

# Example
`x = [0, 3, 1, 2, -1]` has a full cycle `1 ↔ 2` (steps 3, 4) and half cycles `0 → 3`
(steps 1, 2) and `3 → -1` (steps 2, 5). All three have features 4 and 5 equal to 0, while
e.g. `(2, 4)` (values 3 and 2) has feature 5 equal to 1, as the path dips to 1 in between.

Clean paths are necessary but not sufficient: adjacent steps such as `(2, 3)` always
pass, and token 2 has the clean partners 1, 3 and 5. Picking the actual partner is left
to the attention scores.
"""
function pair_features(x::AbstractMatrix{T}) where T
    M, B = size(x)
    P = zeros(T, N_PAIR_FEATURES, M, M, B)
    for b in 1:B, i in 1:M
        xi = x[i, b]
        dl = i > 1 ? xi - x[i-1, b] : zero(T)   # slope into i
        dr = i < M ? x[i+1, b] - xi : zero(T)   # slope out of i

        # Walk away from i (forward, then backward), keeping the extrema of the path i..j
        for js in (i:M, i:-1:1)
            x_max = x_min = xi
            for j in js
                xj = x[j, b]
                x_max = max(x_max, xj)
                x_min = min(x_min, xj)
                P[:, j, i, b] .= (
                    xi, xj, abs(xj - xi),
                    x_max - max(xi, xj),   # path rises above both endpoints
                    min(xi, xj) - x_min,   # path drops below both endpoints
                    dl, dr, T((j - i) / M),  # Float64 here would make the tuple heterogeneous and allocate
                )
            end
        end
    end
    return P
end

"""
    CycleAttention(M; h = 24, nheads = 4)

Build a Lux model that predicts the rainflow wear cost of SOC curves with `M` time
steps. It takes `P = pair_features(x)` of size `(N_PAIR_FEATURES, M, M, B)` and returns
the predicted wear of each curve, of size `(1, B)` where `B` is the batch size.

# Idea
Rainflow counting pairs each turning point of a curve with one partner; each pair is a
cycle of depth `|x_i - x_j|`, and the wear is a sum over all cycles. Attention fits this:
the `M` time steps are the tokens, and each token `i` spreads a softmax over all other
steps `j`, i.e. it "picks its cycle partner".

Whether `(i, j)` is a cycle depends on what the path does *between* i and j, which a
standard dot-product score `q_i ⋅ k_j` can't easily express. So scores and values are
computed directly from per-pair features (see `pair_features`), similar to the pair
representations in AlphaFold.

# Architecture
For every pair `(i, j)`, per sample:
1. `pair_mlp` (`N_PAIR_FEATURES => h => h`, gelu) embeds the pair features.
2. `score` (`h => nheads`) gives the attention logit "j is the cycle partner of i".
   The diagonal is masked, as a step can't pair with itself.
3. `value` (`h => nheads`, softplus) gives the damage if it is (non-negative).
4. Softmax of the scores over `j`, then the attention-weighted sum of the values gives
   the damage attributed to token `i`, per head.
5. `head` (`nheads => 1`) combines the heads, and the result is **summed** over tokens:
   wear is additive over cycles, so a sum (not a mean) matches the structure.

Several heads give the model room to keep alternatives apart, e.g. full vs. half
cycles, or the partner before vs. after `i` (what each head learns isn't constrained).

# Arguments
- `M`: number of time steps per curve (tokens).
- `h`: width of the pair embedding.
- `nheads`: number of attention heads.

With the defaults the model has 1,021 parameters. The pair tensors scale as
`M² × B × h`, so batches are much larger in memory than for an MLP on the raw curve.

# Example
```julia
model = CycleAttention(M)
ps, st = Lux.setup(rng, model)
y, _ = model(pair_features(X), ps, st)   # X: (M, B) SOC curves (scaled), y: (1, B)
```
"""
function CycleAttention(M; h = 24, nheads = 4)
    self_mask = reshape(-1.0f4 .* (1:M .== (1:M)'), 1, M, M, 1)  # a token can't pair with itself
    return @compact(
        pair_mlp = Chain(Dense(N_PAIR_FEATURES => h, gelu), Dense(h => h, gelu)),
        score = Dense(h => nheads),            # logit: is j the cycle partner of i?
        value = Dense(h => nheads, softplus),  # damage if it is
        head = Dense(nheads => 1),
    ) do P                                     # P = pair_features(x): (N_PAIR_FEATURES, M, M, B)
        B = size(P, 4)
        H = pair_mlp(reshape(P, N_PAIR_FEATURES, :))
        S = reshape(score(H), nheads, M, M, B) .+ self_mask
        V = reshape(value(H), nheads, M, M, B)
        A = softmax(S; dims = 2)                              # soft partner selection over j
        C = dropdims(sum(A .* V; dims = 2); dims = 2)         # (nheads, M, B) damage per token
        y = sum(reshape(head(reshape(C, nheads, :)), M, B); dims = 1)
        @return y
    end
end

model = CycleAttention(M)


function dataloaders(rng::AbstractRNG; N = 10_000)
    Xv, y = generate_data(rng, Float32, N)
    X = reduce(hcat, Xv)
    y = y'
    (X_train, y_train), (X_val, y_val) = splitobs((X, y); at = 0.8, shuffle = true)
    with_features((x, y)) = (pair_features(x), y)   # applied lazily, per batch

    return (
        train = DataLoader(mapobs(with_features, collect.((X_train, y_train))); batchsize=128, shuffle=true, partial=false),
        val = DataLoader(mapobs(with_features, collect.((X_val, y_val))); batchsize=128, shuffle=false, partial=false),
    )
end
dataloaders(; kwargs...) = dataloaders(Random.default_rng(); kwargs...)


# Cosine decay from lr to lr_min over the epochs
cosine_lr(epoch, epochs, lr, lr_min) = oftype(lr, lr_min + (lr - lr_min) * (1 + cospi((epoch - 1) / (epochs - 1))) / 2)

# Returns the parameters/states of the epoch with the lowest validation loss
function train(model; rng = Random.default_rng(), N = 10_000, epochs = 300, lr = 3f-3, lr_min = 1f-4)
    cpu_dev = cpu_device()
    dev = reactant_device()
    train_dataloader, val_dataloader = dataloaders(rng; N) |> dev
    ps, st = Lux.setup(rng, model) |> dev

    vjp = AutoEnzyme()
    opt = Adam(lr)
    loss_function = MSELoss()

    train_state = Training.TrainState(model, ps, st, opt)
    x_ra = first(val_dataloader)[1]
    model_compiled = @compile model(x_ra, ps, Lux.testmode(st))

    best_loss, best_epoch = Inf32, 0
    best = (train_state.parameters, train_state.states) |> cpu_dev

    for epoch in 1:epochs
        eta = cosine_lr(epoch, epochs, lr, lr_min)
        train_state = Optimisers.adjust!(train_state, eta)   # returns a new TrainState

        total_loss = 0.0f0
        total_samples = 0
        start_time = time()
        for (x, y) in train_dataloader
            _, loss, _, train_state = Training.single_train_step!(vjp, loss_function, (x, y), train_state)
            total_loss += loss * length(y)
            total_samples += length(y)
        end
        epoch_time = time() - start_time

        total_loss /= total_samples

        @printf "Epoch: %3d \t lr: %.2e \t Train time: %.3f sec \t Loss: %.4f\n" epoch eta epoch_time total_loss

        total_loss = 0.0f0
        total_samples = 0

        st_ = Lux.testmode(train_state.states)
        for (x, y) in val_dataloader
            y_hat, st_ = model_compiled(x, train_state.parameters, st_)
            y_hat = y_hat |> cpu_dev
            y_true = y |> cpu_dev
            total_loss += loss_function(y_hat, y_true) * length(y)
            total_samples += length(y)
        end
        total_loss /= total_samples

        @printf "Validation:\tLoss: %.4f\n" total_loss

        if total_loss < best_loss
            best_loss, best_epoch = total_loss, epoch
            best = (train_state.parameters, train_state.states) |> cpu_dev
        end
    end
    @printf "Best validation loss %.4f at epoch %d\n" best_loss best_epoch
    return best
end


ps_trained, st_trained = train(model; rng=rng)

function predict(model, ps_trained, st_trained, X)
    st_ = Lux.testmode(st_trained)
    y_hat, st_ = model(pair_features(Float32.(X)), ps_trained, st_)
    return y_hat
end
