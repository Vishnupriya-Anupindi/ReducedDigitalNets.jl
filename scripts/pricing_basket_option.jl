#=
    We aim to approximate 

        𝔼[Sⱼ(T)] = 1/bᵐ ∑ₖ Sⱼ(0) exp(-σ/2T + (Lxₖ)ⱼ√T)

    Sⱼ(0) = 100 
    j, s = 10 
    T = 1 
    K = 110 
    σ = 0.4 
    ρ = 0.2     

    R = 5 
    m = 25 

    wⱼ = min(⌊log₂(jᶜ)⌋, m)
    c ∈ {1, 0.5, 0}
=#
#import Pkg; Pkg.add("Distributions")
#Pkg.add("ProgressMeter")
using ReducedDigitalNets, LinearAlgebra, Statistics, Distributions, DataFrames, CSV, ProgressMeter


###########
# Parameters
use_sobol = true

b = 2
s = 10 
T = 1 
K = 110
S_init = 100
sigma = 0.4 
rho = 0.2     

R_ref = 5

if use_sobol
    R = 1
else 
    R = 5 
end

m_ref = 25
m_test = 10:20

###########
# Helper function to convert uniform to normal distribution:

dx = Float64(b)^(-m_ref)

function make_quantile(dx)
    return x -> quantile(Normal(), dx/2 + x)
end

# quantile function of the normal distribution 
randn_quantile = make_quantile(dx)


# this function tests if two random vectors are roughly similar:
stoch_equal(a, b, cut, thres) = abs(count(a .< cut) - count(b .< cut)) < length(a) * thres

# test the randn_quantile function
@assert stoch_equal(randn_quantile.(rand(100)), randn(100), 0.5, 0.1)



###########
# Helper function for loading Sobol matrices
function row_to_mat(row,b,m)
    return stack(first(digits(x, base = b, pad = m),m) for x in first(row,m))
end

function load_seq_mat(filename,b,m,s)
    df = CSV.read(filename, DataFrame, header = false, delim = ' ')
    K = Matrix(df[1:s,1:m])
    C = [row_to_mat(K[i,:],b,m) for i in 1:s]
    return C
end

# use like this: C = load_seq_mat("Data/sobol_Cs.txt",2,4,2)





###########
# Compute L, such that LL^T = Σ
Σ = diagm(-1 => fill(rho,s-1), 0 => fill(sigma,s), 1 => fill(rho,s-1))
# Σ[1,1] = rho
L = cholesky(Σ).L

@assert L * L' ≈ Σ


A = L' 

function kernel(xa, sigma, T, S_init)
    return S_init * exp(-sigma/(2*T) + xa * sqrt(T))
end

function expected_value(XA_prod, sigma, T, S_init)
    N = size(XA_prod, 1)
    s = size(XA_prod, 2)
    return [1/N * sum(kernel(XA_prod[k,j], sigma, T, S_init) for k in 1:N) for j in 1:s]
end


function pay_off(ES,K)
    return max(0.0, mean(ES) - K)
end

# comute reference value, here if we use a matrix product, we quickly run out of memory, therefore a manual for loop 

# ES_T_ref = zeros(s, Threads.nthreads())

# @showprogress Threads.@threads :static for k in 1:(R_ref * b^m_ref)
#     tid = Threads.threadid()
    
#     xa_k = L * randn(s)  # notice that we do not transpose L here!
#     for j in 1:s 
#         ES_T_ref[j,tid] += 1/(R_ref * b^m_ref) * kernel(xa_k[j], sigma, T, S_init)
#     end
# end
# ES_T_ref = sum(ES_T_ref, dims = 2)[:,1]

ES_T_ref = fill(100.0, s)

 R_ref = 1
 X = randn(R_ref * b^m_ref, s)
 XA_prod_ref = X * A
 ES_T_ref = expected_value(XA_prod_ref, sigma, T, S_init)
 HS_ref = pay_off(ES_T_ref, K)


# compute for different values of c 


ES = Array{Vector}(undef, length(m_test), 5, R)

# ES: 
# 1-dim: m-values 
# 2-dim: c-values: 0 (col/row), 0.5 (col), 1.0 (col), 0.5 (row), 1.0 (row)
# 3-dim: R-values 

for (i_m, m) in enumerate(m_test) 
    for (i_c, c) in enumerate([0,0.5,1.0])
        Threads.@threads for i_r in 1:R

            println("m = $m, c_col = $c, rep = $i_r") 

            if use_sobol
                C = load_seq_mat("sobol_Cs.txt",b, m, s)
            else
                C = [rand(0:1,m,m) for i in 1:s]        
            end

            P = DigitalNetGenerator(b,m,s,C)            
            w_s = [min(floor(Int,log2(j^c)), m) for j in 1:s]

            #Column reduced
            XA_prod_colred = colredmul(P, A, w_s, randn_quantile)
            ES_T_colred = expected_value(XA_prod_colred, sigma, T, S_init)
            @show ES_T_colred'

            ES[i_m, i_c, i_r] = ES_T_colred
        end 
    end

    for (i_c, c) in enumerate([0.5,1.0])
        Threads.@threads for i_r in 1:R
            println("m = $m, c_row = $c, rep = $i_r") 

            if use_sobol
                C = load_seq_mat("sobol_Cs.txt",b,m,s)
            else
                C = [rand(0:1,m,m) for i in 1:s]        
            end
            P = DigitalNetGenerator(b,m,s,C)                
            w_s = [min(floor(Int,log2(j^c)), m) for j in 1:s]

            #Row reduced
            Prr = rowredmatrices(P,w_s)
            # st = findlast(w_s .< m)
            pts = genpoints(Prr)
            XA_prod_rowred = rowredmul(P, A, w_s, pts, randn_quantile)
            ES_T_rowred = expected_value(XA_prod_rowred, sigma, T, S_init)
            @show ES_T_rowred'

            ES[i_m, 3 + i_c, i_r] = ES_T_rowred
        end
    end
end

# (a,b)
# mean(x) = a + b 
# norm(x,1) = |a| + |b| 


variant = 1

if variant == 1 
    ES_err = [abs(mean(es) .- mean(ES_T_ref)) for es in ES]
    variant_name = "abs_mean_diff"
elseif variant == 2
    ES_err = [1/s * norm(es .- ES_T_ref, 1) for es in ES]
    variant_name = "1_norm"
end 
mean_ES_err = mean(ES_err, dims = 3)[:,:,1]

HS_ref = pay_off(ES_T_ref, K)
HS = pay_off.(ES, K)  # issue, all payoffs are zero
HS_err = abs.(HS .- HS_ref)
# error_std_col = std(errs_col, dims = 3)[:,:,1]

using CairoMakie


err_quantitiy = mean_ES_err

begin 
    fig = Figure()

    ax = Axis(fig[1,1], yscale = log10, xlabel = "m", ylabel = "Mean error", yminorticksvisible = true, yminorgridvisible = true)

    c = 0
    lines!(ax, m_test, err_quantitiy[:, 1], linewidth = 2, label = "c = $(c)")
    scatter!(ax, m_test, err_quantitiy[:, 1], label = "c = $(c)")

    c = 0.5
    lines!(ax, m_test, err_quantitiy[:, 2], linewidth = 2, label = "Reduced c_col = $(c)")
    scatter!(ax, m_test, err_quantitiy[:, 2], label = "Reduced c_col = $(c)", marker = :rect)
    #row_red
    lines!(ax, m_test, err_quantitiy[:, 4], linewidth = 2, label = "Reduced c_row = $(c)", linestyle = :dash)
    scatter!(ax, m_test, err_quantitiy[:, 4], label = "Reduced c_row = $(c)", marker = :rect)

    c = 1.0
    lines!(ax, m_test, err_quantitiy[:, 3], linewidth = 2, label = "Reduced c_col = $(c)")
    scatter!(ax, m_test, err_quantitiy[:, 3], label = "Reduced c_col = $(c)", marker = :utriangle)
    #row_red
    lines!(ax, m_test, err_quantitiy[:, 5], linewidth = 2, label = "Reduced c_row = $(c)", linestyle = :dash)
    scatter!(ax, m_test, err_quantitiy[:, 5], label = "Reduced c_row = $(c)", marker = :utriangle)

    axislegend(ax, merge = true)

    save("Output/pricing_basket_option_$(variant_name).png", fig)
    save("Output/pricing_basket_option_$(variant_name).svg", fig)
    fig
end
