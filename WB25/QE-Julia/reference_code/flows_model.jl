using CSV, DataFrames
using Plots

include("data_cal.jl")

using Statistics

println("θ_hat = ", θ_hat)
γW_hat, γR_hat, αW_hat, αR_hat = θ_hat

println("γW_hat = $γW_hat, γR_hat = $γR_hat, αW_hat = $αW_hat, αR_hat = $αR_hat")

println("amenR: mean = ", mean(amenR_base),
        ", sd = ", std(amenR_base),
        ", min = ", minimum(amenR_base),
        ", max = ", maximum(amenR_base))

println("amenW: mean = ", mean(amenW_base),
        ", sd = ", std(amenW_base),
        ", min = ", minimum(amenW_base),
        ", max = ", maximum(amenW_base))
println("Loaded env and θ_hat from data_cal.jl")
println("θ_hat = ", θ_hat)

# Get Model Migration Matrices

P_40_50_model  = moms[:P_40_50]
P_65p_model    = moms[:P_65p]

Z = size(P_40_50_model, 1)
@assert Z == length(states)

# FIP -> name lookup for labels
const fips_to_name = Dict(
    1=>"Alabama", 2=>"Alaska", 4=>"Arizona", 5=>"Arkansas", 6=>"California",
    8=>"Colorado", 9=>"Connecticut", 10=>"Delaware", 11=>"District of Columbia",
    12=>"Florida", 13=>"Georgia", 15=>"Hawaii", 16=>"Idaho", 17=>"Illinois",
    18=>"Indiana", 19=>"Iowa", 20=>"Kansas", 21=>"Kentucky", 22=>"Louisiana",
    23=>"Maine", 24=>"Maryland", 25=>"Massachusetts", 26=>"Michigan",
    27=>"Minnesota", 28=>"Mississippi", 29=>"Missouri", 30=>"Montana",
    31=>"Nebraska", 32=>"Nevada", 33=>"New Hampshire", 34=>"New Jersey",
    35=>"New Mexico", 36=>"New York", 37=>"North Carolina", 38=>"North Dakota",
    39=>"Ohio", 40=>"Oklahoma", 41=>"Oregon", 42=>"Pennsylvania",
    44=>"Rhode Island", 45=>"South Carolina", 46=>"South Dakota",
    47=>"Tennessee", 48=>"Texas", 49=>"Utah", 50=>"Vermont", 51=>"Virginia",
    53=>"Washington", 54=>"West Virginia", 55=>"Wisconsin", 56=>"Wyoming"
)

fips_label(fips::Int) = get(fips_to_name, fips, string(fips))

# Extracting top N flows from migration matrix
function top_flows(P::AbstractMatrix{<:Real},
                   states::Vector{Int},
                   N::Int = 10)
    Z = size(P, 1)
    @assert size(P, 2) == Z "P must be square matrix"
   
   df = DataFrame(
        origin_idx = Int[],
        dest_idx   = Int[],
        origin_fips = Int[],
        dest_fips   = Int[],
        origin_name = String[],
        dest_name   = String[],
        prob        = Float64[],
   )

   for i in 1:Z, j in 1:Z
        if i == j
            continue  # skip diagonal
        end
        p = Float64(P[i,j])
        if p > 0
            ofips = states[i]
            dfips = states[j]
            push!(df, (
                i, j,
                ofips,
                dfips,
                fips_label(ofips),
                fips_label(dfips),
                p
            ))
        end
   end

    sort!(df, :prob, rev=true)
    return first(df, min(N, nrow(df)))
end

top_10_40_model = top_flows(P_40_50_model, states, 10)
top_10_65p_model = top_flows(P_65p_model, states, 10)

println("Top 10 Migration Flows Model 40-50:")
show(top_10_40_model, allrows=true, allcols=true)
println("\nTop 10 Migration Flows Model 65+:")
show(top_10_65p_model, allrows=true, allcols=true)

# Write to CSV
CSV.write("top_10_flows_40_50_model.csv", top_10_40_model)
CSV.write("top_10_flows_65p_model.csv", top_10_65p_model)
println("Wrote top 10 flows to CSV files.")

# Bar Plots
function flow_label(origin_name, dest_name)
    return string(origin_name, " → ", dest_name)
end

function plot_top_flows(df::DataFrame; age_label::AbstractString)
    df_plot = transform(df,
        [:origin_name, :dest_name] => ByRow(flow_label) => :flow_label
    )
 
    df_plot = sort(df_plot, :prob, rev=true)
    df_plot.flow_label = categorical(df_plot.flow_label, levels = df_plot.flow_label)

    bar(
        df_plot.flow_label,
        df_plot.prob;
        legend = false,
        xlabel = "Origin → Destination",
        ylabel = "Migration Probability (conditional on moving)",
        title = "Top 10 Migration Flows Model - " * age_label,
        orientation = :h,
    )
end

plot_40 = plot_top_flows(top_10_40_model; age_label="Ages 40-50 (Model)")
savefig(plot_40, "top_10_flows_40_50_model.pdf")
println("Saved top 10 flows plot for ages 40-50.")
plot_65p = plot_top_flows(top_10_65p_model; age_label="Ages 65+ (Model)")
savefig(plot_65p, "top_10_flows_65p_model.pdf")
println("Saved top 10 flows plot for ages 65+.")

    
