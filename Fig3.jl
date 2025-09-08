using Distributed
addprocs()                    # add workers (adjust # via addprocs(N))
@info "Workers: $(workers())"

@everywhere begin
    using Graphs
    using SparseArrays
    using LinearAlgebra
    using DifferentialEquations
    using Random
    using Statistics
end

# ---------------------
# Model helpers 
# ---------------------
@everywhere function build_triplets(A::AbstractMatrix{<:Real})
    N = size(A,1)
    i_idx = Int[]; j_idx = Int[]; k_idx = Int[]; val_idx = Float64[]
    @inbounds for i in 1:N, j in 1:N, k in 1:N
        val = A[i,j] * A[j,k] * A[k,i]
        if val != 0.0
            push!(i_idx, i); push!(j_idx, j); push!(k_idx, k); push!(val_idx, float(val))
        end
    end
    return i_idx, j_idx, k_idx, val_idx
end

@everywhere function compute_hoi(x::AbstractVector{<:Real}, i_idx, j_idx, k_idx, val_idx, N::Int)
    hoi = zeros(eltype(x), N)
    @inbounds @simd for n in eachindex(val_idx)
        i = i_idx[n]; j = j_idx[n]; k = k_idx[n]
        hoi[i] += val_idx[n] * x[j] * x[k]
    end
    return hoi
end

@everywhere function make_dynamics(A_sparse::SparseMatrixCSC{Float64,Int},
                                   i_idx, j_idx, k_idx, val_idx,
                                   r::Vector{Float64}, a::Float64, b::Float64, x0::Float64,
                                   d::Float64, d1::Float64)
    N = length(r)
    function f!(du, u, p, t)
        linear_term = A_sparse * u
        hoi_term    = compute_hoi(u, i_idx, j_idx, k_idx, val_idx, N)
        @inbounds @. du = -a * (u - x0)^3 + b * (u - x0) + r + d * linear_term + d1 * hoi_term
        return nothing
    end
    return f!
end

@everywhere function simulate_one(seed::Int,
                                  N::Int, p::Float64,
                                  d_vals, d1_vals,
                                  a::Float64, b::Float64, x0::Float64,
                                  r::Vector{Float64})
    # rng = MersenneTwister(seed)

    # Network (ER)
   G = erdos_renyi(N, p; is_directed=false)
   # G = complete_graph(N)
    A = Float64.(adjacency_matrix(G))
    @inbounds A[diagind(A)] .= 0.0
    A_sparse = sparse(A)

    # Triplets
    i_idx, j_idx, k_idx, val_idx = build_triplets(A)
    println("Starting simulation seed=$seed on worker $(myid())")

    # Sweep 
    local_sum = zeros(length(d1_vals), length(d_vals))
    for (ii, d1) in enumerate(d1_vals), (jj, d) in enumerate(d_vals)
        x_current = zeros(Float64, N)
    #    x_current = rand(Float64, N).*1e-6       # uniform [0,1)
        f!   = make_dynamics(A_sparse, i_idx, j_idx, k_idx, val_idx, r, a, b, x0, d, d1)
        prob = ODEProblem(f!, x_current, (1.0, 100.0))
        sol  = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10)

        uT = sol.u[end]
        final_tipped = count(>(0.5), uT)
        local_sum[ii, jj] += (final_tipped > 1) ? 1.0 : 0.0
    end
    println("Finished simulation seed=$seed on worker $(myid())")
    return local_sum
end

# ---------------------
# Parameters 
# ---------------------
N = 100
average_degree = 4.0
p = average_degree / (N - 1)   # valid probability

a, b, x0 = 4.0, 1.0, 0.5
r = zeros(Float64, N); r[1] = 0.2

d_vals  = range(0.0, 0.2; length=150)
d1_vals = range(-1.0, 0.0; length=150)

sim = 100
seeds = collect(1:sim)

# ---------------------
# Run in parallel
# ---------------------
@time partials = pmap(seeds) do s
    simulate_one(s, N, p, d_vals, d1_vals, a, b, x0, r)
end

sum_tipping_matrix = reduce(+, partials) ./ sim

# ---------------------
# Plot
# ---------------------
using Plots
default(size=(800,500))
heatmap(collect(d_vals), collect(d1_vals), sum_tipping_matrix;
        xlabel = "Linear coupling d",
        ylabel = "HOI coupling d₁",
        colorbar_title = "Fraction of runs tipped",
        framestyle = :box,
        c = :coolwarm,
        clims = (0,1))


using CSV, DataFrames

# Create a DataFrame with triplets: d, d1, value
rows = []
for i in 1:length(d1_vals)
    for j in 1:length(d_vals)
        push!(rows, (d = d_vals[j], d1 = d1_vals[i], value = sum_tipping_matrix[i, j]))
    end
end

df = DataFrame(rows)
CSV.write("result.csv", df)

