using CSV, DataFrames
using LinearAlgebra, StatsBase
include("MigCalibration.jl")

# --------- Load Targets -----------
rates = CSV.read("targets_move_rates.csv", DataFrame)
p40_data = rates.p_move_40_50[1]
p65_data = rates.p_move_65p[1]

# flow matrices: rows keyed by origin, columns string state codes

function load_P(path::String)
    df = CSV.read(path, DataFrame)
    states = names(df)[names(df) .!= "origin"]
    states_int = parse.(Int, String.(states))
    sortperm_states = sortperm(states_int)
    states_sorted = states_int[sortperm_states]

    origins = Int.(df.origin)
    origins_sorted = sort(unique(origins))
    Z = length(states_sorted)

    M = zeros(Float64, Z, Z)
    for r in eachrow(df)
      i = findfirst(==(Int(r.origin)), origins_sorted)
      for (jcol, sc) in enumerate(states_sorted)
          col = string(sc)
          M[i, jcol] = Float64(r[col])
      end
    end
    (origins_sorted, states_sorted, M)
end

# Robust conversion from CSV values to Float64
function as_float(x)
    if x isa Number
        return Float64(x)
    elseif x isa Missing
        return 0.0
    else
        s = strip(String(x))
        if isempty(s) || s == "NA"
            return 0.0
        else
            return parse(Float64, s)
        end
    end
end

orig40, dest40, P40_data = load_P("targets_P_40_50.csv")
orig65, dest65, P65_data = load_P("targets_P_65p.csv")
@assert orig40 == dest40 == orig65 == dest65 "State sets mismatched"

targets_agepar_df = CSV.read("targets_agepair_probs.csv", DataFrame)
ages_data = targets_agepar_df.age_start
p_data    = targets_agepar_df.p

states = orig40
Z = length(states)
ages = collect(40:80)
T = length(ages)

# ---------- Load Env Inputes ----------

profiles = CSV.read("env_state_age_profiles.csv", DataFrame)
init40 = CSV.read("env_init_dist_40.csv", DataFrame)
state_wage_means = CSV.read("state_age_mean.csv", DataFrame)

# map states & ages to indices
state_to_i = Dict(s => i for (i,s) in enumerate(states))
age_to_t = Dict(a => t for (t,a) in enumerate(ages))

# fill Z×T matrices
wage   = zeros(Float64, Z, T)
tax    = zeros(Float64, Z, T)
rent   = zeros(Float64, Z, T)
hccost = zeros(Float64, Z, T)


for r in eachrow(profiles)
    s = Int(r.destination)
    a = Int(r.age)
    if haskey(state_to_i, s) && haskey(age_to_t, a)
        i = state_to_i[s]
        t = age_to_t[a]
        tax[i,t]  = Float64(r.tax_rate)
        hccost[i, t] = Float64(r.hccost)
    end
end

wage .= 0.0
rent .= 0.0

for r in eachrow(state_wage_means)
    s = Int(r.destination)
    a = Int(r.age)
    if haskey(state_to_i, s) && haskey(age_to_t, a)
      i = state_to_i[s]
      t = age_to_t[a]
      wage[i,t] = Float64(r.mean_wage)
      rent[i,t] = Float64(r.mean_rentgrs) 
    end
end

has_hccost = :hccost in names(profiles)

# if no healthcare column in the data, generate a simple schedule:
#   - small burden pre-65
#   - larger burden 65+
if !has_hccost
    for i in 1:Z, t in 1:T
        age = ages[t]
        if age >= 65
            hccost[i,t] = 0.20 * rent[i,t]    # retirees: 20% of rent
        else
            hccost[i,t] = 0.05 * rent[i,t]    # workers: 5% of rent
        end
    end
end

using Statistics: median

function fill_gaps!(A::AbstractMatrix)
    B = Float64.(A)              # work on a float copy
    v = vec(B)

    # mask of valid entries for the median: non-NaN and > 0
    mask_valid = (.!isnan.(v)) .& (v .> 0.0)
    vals = v[mask_valid]

    m = isempty(vals) ? 1.0 : median(vals)

    for i in eachindex(B)
        if isnan(B[i]) || B[i] == 0.0
            B[i] = m
        end
    end

    A .= B
    return A
end
fill_gaps!(wage)
fill_gaps!(tax)
fill_gaps!(rent)
fill_gaps!(hccost)

# amenity baselines
amenW_base = zeros(Float64, Z)
amenR_base = zeros(Float64, Z)

try
    amen_df = CSV.read("env_state_amenities.csv", DataFrame)
    println("Loaded env_state_amenities.csv; first rows:")
    println(first(amen_df, 5))

    for r in eachrow(amen_df)
        # FIPS code (numeric or string)
        s_raw = r.statefip
        s = s_raw isa Number ? Int(s_raw) : parse(Int, String(s_raw))

        # amenity value
        a_R = as_float(r.amen_R)
        a_W = as_float(r.amen_W)

        if haskey(state_to_i, s)
            i = state_to_i[s]
            amenR_base[i] = a_R
            amenW_base[i] = a_W
        end
    end

    # center, but only if there's variation
    μ_R = mean(amenR_base)
    μ_W = mean(amenW_base)
    amenR_base .-= μ_R
    amenW_base .-= μ_W

    println("amenR_base: mean = ", mean(amenR_base),
            ", min = ", minimum(amenR_base),
            ", max = ", maximum(amenR_base))
    println("amenW_base: mean = ", mean(amenW_base),
            ", min = ", minimum(amenW_base),
            ", max = ", maximum(amenW_base))
catch err
    @warn "Failed to load env_state_amenities.csv; using zeros for amenR_base, amenW_base" err
    amenR_base .= 0.0
    amenW_base .= 0.0
end

# initial distribution at age 40
# See what columns we actually have
println("init40 columns: ", names(init40))

cols_sym = Symbol.(names(init40))

# Heuristic: pick the state column
state_col::Symbol = if :destination in cols_sym
    :destination
elseif :state in cols_sym
    :state
elseif :origin in cols_sym
    :origin
else
    # fallback: first column that is NOT :w or :p
    first(setdiff(cols_sym, [:w, :p]))
end

println("Using init40 state column: ", state_col)

# Build initial distribution
init_dist_40 = zeros(Float64, Z)

for r in eachrow(init40)
    # state code (numeric or string)
    s_raw = r[state_col]
    s = s_raw isa Number ? Int(s_raw) : parse(Int, String(s_raw))

    # probability p (numeric/string/"NA")
    p = as_float(r[:p])

    if haskey(state_to_i, s)
        i = state_to_i[s]
        init_dist_40[i] = p
    end
end

# Normalize so it sums to 1
total_p = sum(init_dist_40)
if total_p > 0
    init_dist_40 ./= total_p
else
    error("init_dist_40 sums to zero; check env_init_dist_40.csv")
end

println("init_dist_40 sum = ", sum(init_dist_40))

# ----------- Build DataEnv -----------
β = 0.96
σ = 2.0
pension_share = 0.3

prob_death = 0.03 

cbar = 1_000.0
n_agents = 5_000 

delta_vals = [-0.5, 0.0, 0.5]
delta_weights = [0.25, 0.5, 0.25]

env = MigCalibration.DataEnv(
  Z, ages, β, σ, pension_share,
  amenW_base, amenR_base, prob_death,
  cbar, wage, tax, rent, hccost,
  init_dist_40,
  delta_vals, delta_weights
)
MigCalibration.check_env(env)

targets = Dict(
  :p_move_40_50 => p40_data,
  :p_move_65p => p65_data,
  :P_40_50 => P40_data,
  :P_65p => P65_data,
  :agepair_ages => collect(ages_data),
  :agepair_p => collect(p_data),
)

# ----------- Run Calibration -----------

using Random
Random.seed!(0219)

θ_0 = (0.5, 1.0, 0.3, 0.2) # initial guess of disutilities and amenities
θ_hat, res = MigCalibration.run_calibration(
  θ_0, targets, env;
  n_agents  = n_agents,
  w_rate_40 = 10.0,
  w_rate_65 = 1.0,
  w_flow_40 = 2.0,
  w_flow_65 = 1.0,
  w_ageprof = 1.0,
)
println("Estimated parameters: ", θ_hat)

moms = MigCalibration.simulate_model_moments(tuple(θ_hat...), env; n_agents=5_000)

using CSV, DataFrames, Plots

# --- French (2007)-style age profile of mobility ---

ages_model = moms[:agepair_starts]
p_model    = moms[:p_move_pairs]

# Save age profile to CSV (model only)
ageprof_df = DataFrame(age = ages_model, p_move = p_model)
CSV.write("age_profiles_model.csv", ageprof_df)

# Plot: probability of moving by age (2-year bins)
ageprof_fig = plot(
    ages_model,
    p_model,
    marker = :circle,
    label  = "Model",
    xlabel = "Age",
    ylabel = "Prob. of moving (2-year bin)",
    title  = "Age Profiles of Mobility",
)

plot!(
    ageprof_fig,
    ages_data,
    p_data,
    marker    = :diamond,
    linestyle = :dash,
    label     = "Data",
)

savefig(ageprof_fig, "age_profiles_model_vs_data.pdf")

println("Probability of moving between ages 40-50: Model = ", moms[:p_move_40_50])
println("Probability of moving at age 65+: Model = ", moms[:p_move_65p])
