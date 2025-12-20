using Random, Statistics, DataFrames, QuantEcon, LinearAlgebra, Plots

"""
X_{t+1} = AX_{tb} + ΣW_{t+1}
  s.t. 
    X_{tb}, X_{t+1} are Nx1
    A is nxn
    Σ is nxk
    W_t is kx1 and {W_t} is iid ~(0,I)
S_{t+1} = AS_tA' + ΣΣ'
"""

A = [0.8 -0.2; -0.1 0.7]
Σ = [0.5 0.4; 0.4 0.6]

Q = Σ * Σ'

maximum(eigvals(A))

function comp_S(A, Σ;
                S0 = Σ * Σ',
                tol = 1e-6,
                maxiter = 500)
    Q = Σ * Σ'
    S = S0
    err = tol + 1
    iter = 1
    while err > tol && iter <= maxiter
      S_p = A * S * A' + Q
      err = norm(S - S_p)
      S = S_p
      iter += 1
    end
    return S
end

sol = comp_S(A, Σ)
println(sol)

diff_sol = norm(sol - solve_discrete_lyapunov(A, Q))
println(diff_sol)

"""
Exercise 2
"""

#T = 150
#y0 = 0
#θ = [0.8, 0.9, 0.98]
#σ = 1
#γ = 1
#wp = norm(0, 1)
#
#function simulate_y(T, y0, θ, σ, γ, wp;
#                    max_it = 200, tol = 1e-8)
#  y = y0
#  err = tol + 1
#  iter = 1
#  while err > tol && iter <= max_it
#    yt = γ .+ θ*y0 .+ σ*wp
#    err = norm(y .- yt)
#    y = yt
#    iter += 1
#    path_y = stephist(y)
#  end
#  return y
#  savefig("path_y.pdf")
#end
#
#println(simulate_y(T, y0, θ, σ, γ, wp))

# Corrected (AI)

T = 150
y0 = 0.0
θs = [0.8, 0.9, 0.98]
σ = 1
γ = 1
N = 200

Random.seed!(0219)

# Simulate singular path
function sim_path(T, y0, θ, σ, γ)
  y = zeros(T+1)
  y[1] = y0
  for t in 1:T
    y[t+1] = γ + θ*y[t] + σ*randn()
   end
   return y
end

# rolling mean
function rollmean(y)
  x = y[2:end]
  return cumsum(x) ./ (1:length(x))
end

# single path rolling mean plot
p1 = plot(xlabel = "τ", ylabel = "(1/τ)Σ y_t", title = "Rolling Mean")
for θ in θs
  y = sim_path(T, y0, θ, σ, γ)
  plot!(p1, 1:T, rollmean(y), label="θ=$(θ)")
end
savefig("p1.pdf")

# N-paths
p2 = plot(xlabel = "y_T", ylabel = "density", title = "Histogram of Path")
for θ in θs
  yT = [sim_path(T, y0, θ, σ, γ)[end] for _ in 1:N]
  histogram!(p2, yT, bins = 30, normalize=:pdf, alpha = 0.35, label = "θ=$(θ)")
  m = mean(yT)
  v = mean(yT .^ 2) - m^2
  println("θ=$(θ): mean = $(m), var = $(v)")
end
savefig("p2.pdf")

"""
Exercise 3
"""

a = 0.1
b = 0.2
c = 0.5
d = 1.0
σ = 0.1


N = 50
M = 20

#function sim_dgp(M, N, a, b, c, d, σ)
#  T = N * M
#
#  x1_vec = Vector(undef, T)
#  x2_vec = Vector(undef, T)
#  y_vec  = Vector(undef, T)
#
#  idx = 1
#
#  for n in 1:N
#    x1 = randn()
#    x2 = randn()
#    for m in 1:M
#      y = a*x1 + b*x1^2 + c*x2 + d + σ*randn(0,1)
#      x1_vec[idx] = x1
#      x2_vec[idx] = x2
#      y_vec[idx] = y
#      idx += 1
#    end
#  end
#  return x1_vec, x2_vec, y_vec
#end
# 
#function ols_manual(M, x1_vec, x2_vec, y_vec)
#  for m in 1:M
#    a = inv((x1_vec' * x1_vec)) * (x1_vec' * y_vec)
#    b = a
#    c = inv((x2_vec' * x2_vec)) * (x2_vec' * y_vec)
#    d = 0
#    σ = inv((randn(0,1)' * randn(0,1))) * (randn(0,1)' * y_vec)
#  end
#  return a, b, c, d, σ
#end
#
#histogram(a, normalize = :pdf)
#histogram(b, normalize = :pdf)
#histogram(c, normalize = :pdf)
#histogram(d, normalize = :pdf)
#histogram(σ, normalize = :pdf)

## Corrected

function sim_ols(N, M; a = 0.1, b = 0.2, c = 0.5, d = 1.0, σ = 0.1, seed = 0219)
  Random.seed!(seed)

  x1 = randn(N)
  x2 = randn(N)
  
  X = hcat(x1, x1.^2, x2, ones(N))
  k = size(X, 2)

  a_hat = Vector(undef, M)
  b_hat = Vector(undef, M)
  c_hat = Vector(undef, M)
  d_hat = Vector(undef, M)
  σ_hat = Vector(undef, M)

  XtX = X' * X
  XtX_inv = inv(XtX)

  for m in 1:M
    w = randn(N)
    y = a.*x1.+b.*(x1.^2).+c.*x2.+d.+σ.*w
    β_hat = XtX_inv * (X'*y)
    ε = y - X * β_hat

    σhat = sqrt(sum(ε.^2) / (N-k))

    a_hat[m] = β_hat[1]
    b_hat[m] = β_hat[2]
    c_hat[m] = β_hat[3]
    d_hat[m] = β_hat[4]
    σ_hat[m] = σhat
  end
  return(a_hat = a_hat, b_hat = b_hat, c_hat = c_hat, d_hat = d_hat,
         σ_hat = σ_hat)
end

est = sim_ols(N, M; a = a, b = b, c = c, d = d, σ = σ)

println("Estimates: ", est.a_hat, est.b_hat, est.c_hat, est.d_hat, est.σ_hat)

histogram(est.a_hat, title="â", normalize=:pdf)
histogram(est.b_hat, title="b̂", normalize=:pdf)
histogram(est.c_hat, title="ĉ", normalize=:pdf)
histogram(est.d_hat, title="d̂", normalize=:pdf)
histogram(est.σ_hat, title="σ̂", normalize=:pdf)
