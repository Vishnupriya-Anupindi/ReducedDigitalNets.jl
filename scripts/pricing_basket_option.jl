using ReducedDigitalNets, LinearAlgebra, Statistics

# Parameters

b = 2
j = 10
s = 10 
T = 1 
K = 110 
sigma = 0.4 
rho = 0.2     

R_ref = 5
R = 10

m_ref = 25 
m_test = 10:23


# Compute L, such that LL^T = Σ
Σ = diagm(-1 => fill(rho,s-1), 0 => fill(sigma,s), 1 => fill(rho,s-1))
# Σ[1,1] = rho
L = cholesky(Σ).L

@assert L * L' ≈ Σ


A = L' 

function expected_value(XA_prod, sigma, T)

    N = size(XA_prod, 1)
    s = size(XA_prod, 2)

    return [1/N* sum(-sigma/(2*T) + XA_prod[k,j] * sqrt(T) for k in 1:N) for j in 1:s]
end


# comute reference value 


X = rand(R_ref * b^m_ref, s)
XA_prod_ref = X * A
ES_T_ref = expected_value(XA_prod_ref, sigma, T)



# compute for different values of c 


errs = Array{Float64}(undef, length(m_test), 3, R)

for (i_m, m) in enumerate(m_test) 
    for (i_c, c) in enumerate([0,0.5,1.0])
        Threads.@threads for i_r in 1:R

            println("m = $m, c = $c, rep = $i_r") 

            C = [rand(0:1,m,m) for i in 1:s]
            P = DigitalNetGenerator(b,m,s,C)
            
            w_s = [min(floor(Int,log2(j^c)), m) for j in 1:s]
            XA_prod_colred = colredmul(P, A, w_s)
            ES_T_colred = expected_value(XA_prod_colred, sigma, T)

            @show ES_T_colred'

            errs[i_m, i_c, i_r] = norm(ES_T_colred - ES_T_ref)

        end
    end
end

mean_error = mean(errs, dims = 3)[:,:,1]
error_std = std(errs, dims = 3)[:,:,1]

using CairoMakie

begin 
    fig = Figure()

    ax = Axis(fig[1,1], yscale = log10, xlabel = "m", ylabel = "Mean error")

    for (i_c, c) in enumerate([0, 0.5, 1])
        if i_c == 1
            lines!(ax, m_test, mean_error[:, i_c], linewidth = 2, label = "c = $(c)")
            scatter!(ax, m_test, mean_error[:, i_c], label = "c = $(c)")
        elseif i_c == 2
            lines!(ax, m_test, mean_error[:, i_c], linewidth = 2, label = "Reduced c = $(c)")
            scatter!(ax, m_test, mean_error[:, i_c], label = "Reduced c = $(c)", marker = :rect)
        else
            lines!(ax, m_test, mean_error[:, i_c], linewidth = 2, label = "Reduced c = $(c)")
            scatter!(ax, m_test, mean_error[:, i_c], label = "Reduced c = $(c)", marker = :utriangle)
        end


        #lines!(ax, m_test, mean_error[:, i_c], linewidth = 2, label = "c = $(c)")
        
        # band!(ax, m_test, 
        #         mean_error[:, i_c] - error_std[:, i_c] ./ 2 , 
        #         mean_error[:, i_c] + error_std[:, i_c] ./ 2 , 
        #         label = "c = $(c)")
    end

    axislegend(ax, merge = true)

    save("Output/pricing_basket_option.png", fig)
    save("Output/pricing_basket_option.svg", fig)
    fig
end
