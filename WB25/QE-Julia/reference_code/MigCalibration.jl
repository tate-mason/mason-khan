module MigCalibration

using Random 
using LinearAlgebra
using Optim 
using StatsBase: Weights, sample 
using Plots 
using Distributions

# =========================
# 1. Environment Container
# =========================

"""
DataEnv will hold all exogenous objects for the model.

Conventions:
- Ages are discrete s.t. t = 40:80. Indexed 1:T
- Z locations are discrete s.t. z = 1:51. Indexed 1:Z
- Workers: Use wage and rent arrays
- Retirees: Use pension*wage and healthcare cost arrays
"""

struct DataEnv
  Z::Int                # number of states
  ages::Vector{Int}     # 40:80
  β::Float64            # discount factor
  σ::Float64            # CRRA coefficient
  pension_rate::Float64 # pension replacement rate

  amenW_base::Vector{Float64} # base amenity for workers by state
  amenR_base::Vector{Float64} # base amenity for retirees by state
  prob_death::Float64         # annual death probability (after 70)
  cbar::Float64               # minimum consumption for utility

  # state x age profiles
  wage::Matrix{Float64}     # pre-tax labor income potential
  tax::Matrix{Float64}      # average tax rate in [0,1]
  rent::Matrix{Float64}     # rental housing cost
  hc_cost::Matrix{Float64}  # healthcare cost for retirees

  # Initial distributions over locations
  init_dist_40::Vector{Float64}  # initial distribution at age 40
  δ_vals::Vector{Float64}       # expectation shocks
  δ_weights::Vector{Float64}    # weights for expectation shocks
end

function check_env(env::DataEnv)
  T = length(env.ages)
  @assert size(env.wage) == (env.Z, T)
  @assert size(env.tax) == (env.Z, T)
  @assert size(env.rent) == (env.Z, T)
  @assert size(env.hc_cost) == (env.Z, T)
  @assert length(env.amenW_base) == env.Z
  @assert length(env.amenR_base) == env.Z
  @assert length(env.init_dist_40) == env.Z
  @assert abs(sum(env.init_dist_40) - 1.0) < 1e-8
  @assert length(env.δ_vals) == length(env.δ_weights)
  @assert abs(sum(env.δ_weights) - 1.0) < 1e-8
end

# =========================
# 2. Preference and Utility
# =========================

"""
CRRA utility function. If c <= 0, return large negative utility.
"""

function uw_crra(c::Float64, σ::Float64, cbar::Float64)
  x = c - cbar
  if x <= 0.0
    return -1e10
  elseif σ == 1.0
    return log(x)
  else
    return (x^(1.0 - σ)) / (1.0 - σ)
  end
end

function ur_crra(c::Float64, σ::Float64)
  x = c
  if x <= 0.0
    return -1e10
  elseif σ == 1.0
    return log(x)
  else
    return (x^(1.0 - σ)) / (1.0 - σ)
  end
end

"""
Compute per-period flow utility for workers in location z at age t.

No shocks here. Shocks enter in logit choice.
"""

function worker_flow_utility(env::DataEnv, z::Int, t::Int, αW::Float64)
  # disposable income net of tax and rent
  y = env.wage[z, t] * (1.0 - env.tax[z, t])
  c = y - env.rent[z, t]
  return uw_crra(c, env.σ, env.cbar) + αW * env.amenW_base[z]
end

"""
Compute per-period flow utility for retirees in location z at age t.
"""

function retiree_flow_utility(env::DataEnv, z::Int, t::Int, αR::Float64)
  # disposable income net of tax and rent
  pension = env.pension_rate * env.wage[z, t]
  y = pension * (1.0 - env.tax[z, t])
  c = y - env.hc_cost[z, t]
  return ur_crra(c, env.σ) + αR * env.amenR_base[z]
end

function policies_by_delta(env::DataEnv, γW::Float64, γR::Float64,
                           αW::Float64, αR::Float64)
  D = length(env.δ_vals)

  αW0 = αW
  αR0 = αR + env.δ_vals[1]
  Vw0, Vr0, Pw0, Pr0 = solve_model(env, γW, γR, αW0, αR0)

  Vw_list = Vector{typeof(Vw0)}(undef, D)
  Vr_list = Vector{typeof(Vr0)}(undef, D)
  Pw_delta = Vector{typeof(Pw0)}(undef, D)
  Pr_delta = Vector{typeof(Pr0)}(undef, D)

  Vw_list[1] = Vw0
  Vr_list[1] = Vr0
  Pw_delta[1] = Pw0
  Pr_delta[1] = Pr0


  for d in 2:D
    αWd = αW
    αRd = αR + env.δ_vals[d]
    Vw, Vr, Pw, Pr = solve_model(env, γW, γR, αWd, αRd)
    Vw_list[d] = Vw
    Vr_list[d] = Vr
    Pw_delta[d] = Pw
    Pr_delta[d] = Pr
  end
  return Vw_list, Vr_list, Pw_delta, Pr_delta
end

# ============================
# 3. Value Function Iteration
# ============================

"""
Solve Dynamic Programming problem via Value Function Iteration.
Return:
- Vw[t,z]: value function for workers at age t in location z
- Vr[t,z]: value function for retirees at age t in location z
- Pw[t,z]: choice probabilities for workers at age t in location z
- Pr[t,z]: choice probabilities for retirees at age t in location z

Embed moving cost γ in choice specific utilities.
Choices follow logit with i.i.d T1 extreme value shocks.
Implies V = log(sum(exp(utilities))) + discounted continuation
"""

function solve_model(env::DataEnv, γW::Float64, γR::Float64,
                     αW::Float64, αR::Float64;)
  check_env(env)

  Z = env.Z
  ages = env.ages
  T = length(ages)
  β = env.β

  # retirement probabilities
  p_retire = zeros(Float64, T)
  for t in 1:T
    age = ages[t]
    if age < 62
      p_retire[t] = 0.0
    elseif age < 70
      p_retire[t] = (age - 62.0 + 1.0) / 8
    else
      p_retire[t] = 1.0
    end
  end

  # allocate
  Vw = zeros(Float64, T, Z)
  Vr = zeros(Float64, T, Z)

  # choice probabilities: (T,Z,Z)
  Pw = zeros(Float64, T, Z, Z)
  Pr = zeros(Float64, T, Z, Z)

  # terminal: no future utility at age 80
  # backward induction
  @inbounds for t in T:-1:1
    for z in 1:Z
      # -----Retiree-----
      u_stay_r = retiree_flow_utility(env, z, t, αR)
      cont_r_stay = 0.0
      if 70 <= t < T
        cont_r_stay = β * Vr[t + 1, z] * (1.0 - env.prob_death)
      elseif t < 70 && t < T
        cont_r_stay = β * Vr[t + 1, z]
      end
      util_r_stay = u_stay_r + cont_r_stay

      # moving options
      util_r = similar(Pr[t, z, :])
      denom_r = 0.0
      for zp in 1:Z
        if zp == z
          util_r[zp] = util_r_stay
        else
        u_move_r = retiree_flow_utility(env, zp, t, αR) - γR
          cont_r_move = 0.0
          if 70 <= t < T
            cont_r_move = β * Vr[t + 1, zp] * (1.0 - env.prob_death)
          elseif t < 70 && t < T 
            cont_r_move = β * Vr[t + 1, zp]
          end
          util_r[zp] = u_move_r + cont_r_move
        end
        # logit
        denom_r += exp(util_r[zp])
      end
      # log-sum-exp for value function
      Vr[t, z] = log(denom_r)
      # choice probabilities
      for zp in 1:Z
        Pr[t, z, zp] = exp(util_r[zp]) / denom_r
      end
      # -----Worker-----
      u_stay_w = worker_flow_utility(env, z, t, αW)

      cont_w_stay = 0.0
      if t < T
        p = p_retire[t]
        cont_w_stay = β * ((1.0 - p) * Vw[t + 1, z] + p * Vr[t + 1, z])
      end
      util_w_stay = u_stay_w + cont_w_stay

      util_w = similar(Pw[t, z, :])
      denom_w = 0.0
      for zp in 1:Z
        if zp == z
          util_w[zp] = util_w_stay
        else
          u_move_w = worker_flow_utility(env, zp, t, αW) - γW
          cont_w_move = 0.0
          if t < T
            p = p_retire[t]
            cont_w_move = β * ((1.0 - p) * Vw[t + 1, zp] + p * Vr[t + 1, zp])
          end
          util_w[zp] = u_move_w + cont_w_move
        end
        denom_w += exp(util_w[zp])
      end
      Vw[t, z] = log(denom_w)
      for zp in 1:Z
        Pw[t, z, zp] = exp(util_w[zp]) / denom_w
      end
    end
  end
  return Vw, Vr, Pw, Pr
end

# ============================
# 4. Simulate Agents
# ============================

"""
Simulate a panel of agents from 40 to 80. 

Inputs:
  - θ = (γW, γR, αW, αR)
  - env: DataEnv
  - N: number of agents to simulate
Outputs:
A dictionary with keys:
  - p_move_40_50: fraction of agents moving between age 40 and 50
  - p_move_65p: fraction of agents moving after age 65
  - P_40_50: (Z,Z) migration matrix between age 40 and 50
  - P_65p: (Z,Z) migration matrix after age 65
"""
function draw_state_from_probs(prob_view::AbstractVector{<:Real}, z_stay::Int, rng)
    # sum probs (allow for small numeric drift)
    s = 0.0
    @inbounds for k in eachindex(prob_view)
        s += prob_view[k]
    end

    # if something is numerically busted, just stay put
    if !(s > 0.0) || !isfinite(s)
        return z_stay
    end

    u = rand(rng) * s
    cum = 0.0
    @inbounds for k in eachindex(prob_view)
        cum += prob_view[k]
        if u <= cum
            return k
        end
    end
    return lastindex(prob_view)
end

function simulate_model_moments(theta::NTuple{4,Float64},
                                env::DataEnv;
                                n_agents::Int = 100_000,
                                rng::AbstractRNG = Random.default_rng())

    γW, γR, αW, αR = theta
    check_env(env)

    Z     = env.Z
    ages  = env.ages
    T     = length(ages)
    D     = length(env.δ_vals)

    p_retire = zeros(Float64, T)
    for t in 1:T
        age = ages[t]
        if age < 62
            p_retire[t] = 0.0
        elseif age < 70
            p_retire[t] = (age - 62.0 + 1.0) / 8
        else
            p_retire[t] = 1.0
        end
    end

    probs_vec = zeros(Float64, Z)
    w_probs   = Weights(probs_vec)

    # 1. Solve model once
    Vw_list, Vr_list, Pw_list, Pr_list = policies_by_delta(env, γW, γR, αW, αR)

    # 2. Counters
    move_40_50 = 0
    obs_40_50  = 0

    move_65p = 0
    obs_65p  = 0

    flows_40_50 = zeros(Float64, Z, Z)
    flows_65p   = zeros(Float64, Z, Z)

    amin = 42 
    amax = maximum(env.ages)

    n_pairs = Int(floor((amax - amin) / 2)) + 1
    exp_pair = zeros(Float64, n_pairs)
    mov_pair = zeros(Float64, n_pairs)

    agepair_index(age) = begin
      if age < amin || age > amax
        return 0
      end
      Int(floor((age - amin) / 2)) + 1
    end
  

    # initial location distribution at 40
    init_w = Weights(env.init_dist_40)
    delta_w = Weights(env.δ_weights)

    # 3. Simulate agents
    @inbounds for _ in 1:n_agents
        d = sample(rng, 1:D, delta_w)  # expectation shock index
        Pw = Pw_list[d]; Pr = Pr_list[d]

        z = sample(rng, 1:Z, init_w)   # start location

        is_retired = false

        for t in 1:T
            age = ages[t]
            k = agepair_index(age)

            # exposure counts
            if 40 <= age <= 50
                obs_40_50 += 1
            end
            if age >= 65
                obs_65p += 1
            end
            if k > 0
                exp_pair[k] += 1.0
            end

            # choice probabilities for next location (no allocation)
            prob_view = if is_retired
                @view Pr[t, z, :]
            else
                @view Pw[t, z, :]
            end

            # draw next location
            zp = draw_state_from_probs(prob_view, z, rng)
            moved = (zp != z)

            # record moves and flows by age group
            if 40 <= age <= 50 && moved
                move_40_50 += 1
                flows_40_50[z, zp] += 1.0
            end
            if age >= 65 && moved
                move_65p += 1
                flows_65p[z, zp] += 1.0
            end
            if k > 0 && moved
                mov_pair[k] += 1.0
            end

            # deterministic retirement transition at 65
            if !is_retired
              p = p_retire[t]
              if rand(rng) < p
                is_retired = true
              end
            end
            # update location
            z = zp
        end
    end

    # 4. Move probabilities
    p_move_40_50 = obs_40_50 > 0 ? move_40_50 / obs_40_50 : NaN
    p_move_65p   = obs_65p  > 0 ? move_65p   / obs_65p   : NaN
    p_move_pairs = [exp_pair[k] > 0 ? mov_pair[k] / exp_pair[k] : NaN for k in 1:n_pairs]

    agepair_starts = [amin + 2*(k-1) for k in 1:n_pairs]

    # 5. Normalize flows to get P(z'|z) among movers
    function normalize_flows(F::Matrix{Float64})
        P = copy(F)
        for i in 1:Z
            s = sum(P[i, :])
            if s > 0.0
                @inbounds P[i, :] ./= s
            end
        end
        return P
    end

    P_40_50 = normalize_flows(flows_40_50)
    P_65p   = normalize_flows(flows_65p)

    return Dict(
        :p_move_40_50 => p_move_40_50,
        :p_move_65p   => p_move_65p,
        :P_40_50      => P_40_50,
        :P_65p        => P_65p,
        :agepair_starts => agepair_starts,
        :p_move_pairs   => p_move_pairs,
        :exp_pair      => exp_pair,
        :mov_pair      => mov_pair
    )
end

# ============================
# 5. Calibration Objective
# ============================

"""
Quadratic loss between simulated and empirical moments.

Targets should be Dict or NamedTuple with:
  - :p_move_40_50
  - :p_move_65p
  - :P_40_50
  - :P_65p
"""

function calib_objective(
    θ_vec::Vector{Float64},
    targets,
    env::DataEnv;
    n_agents::Int = 100_000,
    w_rate_40::Float64 = 1.0,
    w_rate_65::Float64 = 1.0,
    w_flow_40::Float64 = 1.0,
    w_flow_65::Float64 = 1.0,
    w_ageprof::Float64 = 1.0,
)
    # big penalty if anything goes wrong
    PENALTY = 1e12

    # unpack θ
    θ = (θ_vec[1], θ_vec[2], θ_vec[3], θ_vec[4])

    moms = try
        simulate_model_moments(θ, env; n_agents = n_agents)
    catch e
        @warn "simulate_model_moments failed in calib_objective" exception = (e, catch_backtrace())
        return PENALTY
    end

    # check that key moments are finite
    for k in (:p_move_40_50, :p_move_65p, :P_40_50, :P_65p)
        v = moms[k]
        if v isa Number
            if !isfinite(v)
                return PENALTY
            end
        else
            if any(!isfinite, v)
                return PENALTY
            end
        end
    end

    # rate errors
    e40 = moms[:p_move_40_50] - targets[:p_move_40_50]
    e65 = moms[:p_move_65p]    - targets[:p_move_65p]

    # flow matrix errors
    Pm40 = moms[:P_40_50]
    Pd40 = targets[:P_40_50]
    Pm65 = moms[:P_65p]
    Pd65 = targets[:P_65p]

    @assert size(Pm40) == size(Pd40)
    @assert size(Pm65) == size(Pd65)

    eflow40 = 0.0
    eflow65 = 0.0

    Z = env.Z
    @inbounds for i in 1:Z, j in 1:Z
        if i != j
            eflow40 += (Pm40[i, j] - Pd40[i, j])^2
            eflow65 += (Pm65[i, j] - Pd65[i, j])^2
        end
    end

    e_age = 0.0
    if haskey(targets, :agepair_p)
      ages_model = moms[:agepair_starts]
      p_model    = moms[:p_move_pairs]
      ages_data = targets[:agepair_ages]
      p_data    = targets[:agepair_p]

      K = min(length(ages_model), length(ages_data))
      @inbounds for k in 1:K
        if isfinite(p_model[k]) && isfinite(p_data[k])
            e_age += (p_model[k] - p_data[k])^2
        end
      end
    end

    e_age_slope = 0.0
    if haskey(targets, :agepair_p)
      ages_model = moms[:agepair_starts]
      p_model    = moms[:p_move_pairs]
      ages_data = targets[:agepair_ages]
      p_data    = targets[:agepair_p]

      K = min(length(ages_model), length(ages_data))
      @inbounds for k in 2:K
        if isfinite(p_model[k]) && isfinite(p_model[k-1]) &&
            isfinite(p_data[k])  && isfinite(p_data[k-1])
            
            Δm = p_model[k] - p_model[k-1] # model change
            Δd = p_data[k]  - p_data[k-1]  # data
            e_age_slope += (Δm - Δd)^2
        end
      end
    end

    λ_smooth = 0.5
    e_age_total = e_age + λ_smooth * e_age_slope

    loss = w_rate_40 * e40^2 +
           w_rate_65 * e65^2 +
           w_flow_40 * eflow40 +
           w_flow_65 * eflow65 +
           w_ageprof * e_age_total

    if !isfinite(loss)
        return PENALTY
    end

    return loss
end

# ============================
# 6. Calibration Wrapper
# ============================

"""
Run calibration using Optim.jl.

  θ = (γW, γR, αW, αR) initialized at θ_0
  bounds and weights can be tuned.
"""

function run_calibration(
    θ_0::NTuple{4, Float64},
    targets,
    env::DataEnv;
    n_agents::Int = 100_000,
    w_rate_40::Float64,
    w_rate_65::Float64,
    w_flow_40::Float64,
    w_flow_65::Float64,
    w_ageprof::Float64,
)
    lower = [0.0, 0.0, -3.0, 0.0]
    upper = [10.0, 7.0, 3.0, 4.0]

    obj(θ_vec) = calib_objective(
        θ_vec,
        targets,
        env;
        n_agents   = n_agents,
        w_rate_40  = w_rate_40,
        w_rate_65  = w_rate_65,
        w_flow_40  = w_flow_40,
        w_flow_65  = w_flow_65,
        w_ageprof  = w_ageprof,
    )

    # derivative-free Nelder–Mead inside Fminbox
    method  = Fminbox(Optim.NelderMead())
    options = Optim.Options(iterations = 200)

    res = Optim.optimize(
        obj,          # objective
        lower,        # lower bounds
        upper,        # upper bounds
        collect(θ_0), # starting point as Vector
        method,
        options,
    )

    θ_hat = Optim.minimizer(res)
    return θ_hat, res
end

end # module


## Simulated data and testing
#
#using Random, LinearAlgebra, Statistics, .MigCalibration
#
## ---------------------------
## 1. Basic Grid and Params
## ---------------------------
#
#const STATES = collect(1:49) # 48 contiguous states + DC
#Z = length(STATES)
#ages = collect(40:80)
#T = length(ages)
#D = 3
#
## Parameters
#δ_vals = [-1, 0, 1] # expectation shocks
#δ_weights = [0.25, 0.5, 0.25]
#β = 0.96
#σ = 3.0
#prob_retire = 0.08
#prob_death = 0.04
#pension_rate = 0.5
#
## -------------------------------
## 2. Location Amentities (base)
## -------------------------------
#
## Simple cross-state pattern: center around 0
#amenW_base = range(-0.5, 0.5; length=Z) |> collect
#amenR_base = range(0.5, -0.5; length=Z) |> collect 
#
## ------------------------------------
## 3. Economic Profiles: w, r, τ, hc
## ------------------------------------
#
#wage = zeros(Float64, Z, T)
#tax = zeros(Float64, Z, T)
#rent = zeros(Float64, Z, T)
#hc_cost = zeros(Float64, Z, T)
#
#for (ti, age) in enumerate(ages)
#  age_factor = 1.0 + 0.02 * (age - 40)
#  for (zi, s) in enumerate(STATES)
#    base_w = 40_000.0 + 60_000.0 * (zi - 1)
#    wage[zi, ti] = base_w * age_factor^2
#
#    tax[zi, ti] = 0 + 0.08 * (zi - 1) / (Z - 1)
#
#    rent[zi, ti] = 8_000.0 + 16_000.0 * (Z - zi) / (Z - 1)
#    hc_cost[zi, ti] = 10_000.0 + 2_000.0 * (zi - 1) / (Z - 1)
#  end
#end
#
## -------------------------------
## 4. Initial Distribution at 40
## -------------------------------
#
## Start with more weight in mid-index states, normalized
#weights = [exp(-((zi - Z/2)^2) / (2 * (Z/6)^2)) for zi in 1:Z]
#init_dist_40 = normalize(weights, 1)
#
## -------------------------------
## 5. Create DataEnv
## -------------------------------
#
#env = MigCalibration.DataEnv(
#  Z,
#  ages,
#  β,
#  σ,
#  pension_rate,
#  amenW_base,
#  amenR_base,
#  cbar = 0.1 * median(vec(wage)),
#  wage,
#  tax,
#  rent,
#  hc_cost,
#  init_dist_40,
#  δ_vals,
#  δ_weights
#)
#MigCalibration.check_env(env)
#
## -------------------------------
## Plotting Move Probabilities by Age Pairs
## -------------------------------
#
#using Plots
#"""
#Plot move probabilities by:
#1. Overall (40-80)
#2. By 2-year age pairs
#3. Age group (40-50, 65+)
#"""
#
#function plot_move_probs(moms; targets=nothing)
#
#    # extract model
#    p40_m = moms[:p_move_40_50]
#    p65_m = moms[:p_move_65p]
#
#    # extract target if provided
#    if targets === nothing
#        x = ["40–50", "65+"]
#        vals = [p40_m, p65_m]
#        return bar(
#            x,
#            vals,
#            title = "Model Move Probabilities",
#            legend = false,
#            ylabel = "Probability"
#        )
#    else
#        p40_t = targets[:p_move_40_50]
#        p65_t = targets[:p_move_65p]
#
#        x = ["40–50", "65+"]
#        model_vals = [p40_m, p65_m]
#        target_vals = [p40_t, p65_t]
#
#        return bar(
#            x,
#            [model_vals target_vals],
#            label = ["Model" "Target"],
#            title = "Migration Probabilities",
#            ylabel = "Probability",
#            legend = :topright
#        )
#    end
#end
#
#function plot_age_profiles(moms; targets=nothing)
#
#    # model moments
#    ages_model = moms[:agepair_starts]
#    p_model    = moms[:p_move_pairs]
#
#    if targets === nothing
#        # just the model, French-style hazard by age
#        plt = plot(
#            ages_model,
#            p_model,
#            marker = :circle,
#            xlabel = "Age",
#            ylabel = "Probability of moving (2-year bin)",
#            title  = "Model age profiles of mobility",
#            legend = false,
#        )
#        return plt
#    else
#        # expect targets as Dict with :age_start and :p arrays
#        ages_t = targets[:age_start]
#        p_t    = targets[:p]
#
#        # restrict to overlapping age range, just in case
#        amin = max(minimum(ages_model), minimum(ages_t))
#        amax = min(maximum(ages_model), maximum(ages_t))
#
#        mask_m = (amin .<= ages_model) .& (ages_model .<= amax)
#        mask_t = (amin .<= ages_t)     .& (ages_t     .<= amax)
#
#        plt = plot(
#            ages_model[mask_m],
#            p_model[mask_m],
#            marker = :circle,
#            xlabel = "Age",
#            ylabel = "Probability of moving (2-year bin)",
#            title  = "Age profiles of mobility (French 2007 style)",
#            label  = "Model",
#        )
#        plot!(
#            plt,
#            ages_t[mask_t],
#            p_t[mask_t],
#            marker = :diamond,
#            linestyle = :dash,
#            label = "Data",
#        )
#        return plt
#    end
#end
## -------------------------------
## 6. Example Run of Simulation
## -------------------------------
#
##n_agents = 100_000
##θ_test = (5.8, 4.8 , 0.5, 0.5)
##moms = MigCalibration.simulate_model_moments(θ_test, env; n_agents=100_000)
##
##plot_move_probs(moms)
##
##println("Probability of moving between 40 and 50: ", moms[:p_move_40_50])
##println("Probability of moving after 65: ", moms[:p_move_65p])
##println("Move probabilities by 2-year age pairs starting at ages: ", moms[:agepair_starts])
##println("Move probabilities by age pairs: ", moms[:p_move_pairs])
#
