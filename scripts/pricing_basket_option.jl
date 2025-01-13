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

using ReducedDigitalNets, LinearAlgebra, Statistics

# Parameters

b = 2
s = 10 
T = 1 
K = 110
S_init = 100
sigma = 0.4 
rho = 0.2     

R_ref = 2
R = 10

m_ref = 20 
m_test = 10:20


# Compute L, such that LL^T = Σ
Σ = diagm(-1 => fill(rho,s-1), 0 => fill(sigma,s), 1 => fill(rho,s-1))
# Σ[1,1] = rho
L = cholesky(Σ).L

@assert L * L' ≈ Σ


A = L' 

function expected_value(XA_prod, sigma, T)

    N = size(XA_prod, 1)
    s = size(XA_prod, 2)
    return [1/N* sum(S_init* exp(-sigma/(2*T) + XA_prod[k,j] * sqrt(T)) for k in 1:N) for j in 1:s]
end


function pay_off(ES,K)
    s = size(ES,1)
    return max(0,1/s* sum(ES[i] for i in 1:s) - K)
end

# comute reference value 


X = rand(R_ref * b^m_ref, s)
XA_prod_ref = X * A
ES_T_ref = expected_value(XA_prod_ref, sigma, T)
HS_ref = pay_off(ES_T_ref, K)



# compute for different values of c 


errs_col = Array{Float64}(undef, length(m_test), 3, R)
errs_row = Array{Float64}(undef, length(m_test), 3, R)

for (i_m, m) in enumerate(m_test) 
    for (i_c, c) in enumerate([0,0.5,1.0])
        Threads.@threads for i_r in 1:R

            println("m = $m, c = $c, rep = $i_r") 

            C = [rand(0:1,m,m) for i in 1:s]
            P = DigitalNetGenerator(b,m,s,C)
            
            w_s = [min(floor(Int,log2(j^c)), m) for j in 1:s]

            #Column reduced
            XA_prod_colred = colredmul(P, A, w_s)
            ES_T_colred = expected_value(XA_prod_colred, sigma, T)
            HS_colred = pay_off(ES_T_colred,K)
            @show ES_T_colred'
            @show HS_colred

            errs_col[i_m, i_c, i_r] = norm(HS_colred - HS_ref)

            #Row reduced
            Prr = rowredmatrices(P,w_s)
            st = findlast(w_s.< m)
            pts = genpoints(Prr)
            XA_prod_rowred = rowredmul(P, A, w_s, pts)
            ES_T_rowred = expected_value(XA_prod_rowred, sigma, T)
            HS_rowred = pay_off(ES_T_rowred,K)
            @show ES_T_rowred'
            @show HS_rowred

            errs_row[i_m, i_c, i_r] = norm(HS_rowred - HS_ref)

        end
    end
end

mean_error_col = mean(errs_col, dims = 3)[:,:,1]
mean_error_row = mean(errs_col, dims = 3)[:,:,1]
error_std_col = std(errs_col, dims = 3)[:,:,1]

using CairoMakie

begin 
    fig = Figure()

    ax = Axis(fig[1,1], yscale = log10, xlabel = "m", ylabel = "Mean error", yminorticksvisible = true, yminorgridvisible = true)

    for (i_c, c) in enumerate([0, 0.5, 1])
        if i_c == 1
            lines!(ax, m_test, mean_error_col[:, i_c], linewidth = 2, label = "c = $(c)")
            scatter!(ax, m_test, mean_error_col[:, i_c], label = "c = $(c)")
        elseif i_c == 2
            lines!(ax, m_test, mean_error_col[:, i_c], linewidth = 2, label = "Reduced c_col = $(c)")
            scatter!(ax, m_test, mean_error_col[:, i_c], label = "Reduced c_col = $(c)", marker = :rect)
            #row_red
            lines!(ax, m_test, mean_error_row[:, i_c], linewidth = 2, label = "Reduced c_row = $(c)", linestyle = :dash)
            scatter!(ax, m_test, mean_error_row[:, i_c], label = "Reduced c_row = $(c)", marker = :rect)
        else
            lines!(ax, m_test, mean_error_col[:, i_c], linewidth = 2, label = "Reduced c_col = $(c)")
            scatter!(ax, m_test, mean_error_col[:, i_c], label = "Reduced c_col = $(c)", marker = :utriangle)
            #row_red
            lines!(ax, m_test, mean_error_row[:, i_c], linewidth = 2, label = "Reduced c_row = $(c)", linestyle = :dash)
            scatter!(ax, m_test, mean_error_row[:, i_c], label = "Reduced c_row = $(c)", marker = :utriangle)
        end


        #lines!(ax, m_test, mean_error[:, i_c], linewidth = 2, label = "c = $(c)")
        
        # band!(ax, m_test, 
        #         mean_error[:, i_c] - error_std[:, i_c] ./ 2 , 
        #         mean_error[:, i_c] + error_std[:, i_c] ./ 2 , 
        #         label = "c = $(c)")
    end

    axislegend(ax, merge = true)

    save("Output/pricing_basket_option_1_HStest1.png", fig)
    save("Output/pricing_basket_option_1_HStest1.svg", fig)
    fig
end
