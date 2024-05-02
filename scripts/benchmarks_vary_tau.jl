using ReducedDigitalNets
using ReducedDigitalNets: stdmul
using BenchmarkTools
using Statistics
using CairoMakie
using GLM, StatsBase, DataFrames, CSV
using Colors

include("utils.jl")

mkpath("Output")

case = 32
b = 2
tau_range = 1000
step_size = 200
m = 12
s = 2000
fn_postfix = "case$(case)_m$(m)_s$(s)_tau$(tau_range)"


BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5
#BenchmarkTools.DEFAULT_PARAMETERS.samples = 2

#For linear s range
#s = collect(1:step_size:s_range)
#M = length(1:step_size:s_range) 

τ_s = collect(1:step_size:tau_range)
M = length(τ_s)

#For exponential s range
# s_exp = floor.(Int, 10 .^LinRange(0,4,n_steps))
# M = length(s_exp)

df = DataFrame(τ = τ_s, gen_pts=zeros(M), row_red = zeros(M), col_red = zeros(M), row_col_red = zeros(M), std_mat_cr = zeros(M), std_mat_rr = zeros(M), theo_col = zeros(M), theo_row = zeros(M))

begin 
   
        C = Matrix{Int64}[]
        for i in 1:s
            C_i = rand(0:1,m,m)
            push!(C,C_i)
        end

        P = DigitalNetGenerator(b,m,s,C)

        #w_s = @. min(floor(Int64,log2(1:s)),m)

        
        if case == 31
            w_s = @. min(floor(Int64,log2(1:s)),m)
        elseif case == 32
            w_s = @. min(floor(Int64,(log2(1:s))^(1/2)),m)
        elseif case == 33
            w_s = @. min(floor(Int64,(log2(1:s))^(1/4)),m)
        end
        

        Pcr = colredmatrices(P,w_s)
        Prr = rowredmatrices(P,w_s)
        st = findlast(w_s.< m)
        pts_cr = genpoints(Pcr)
        pts_rr = genpoints(Prr)
    for k in 1:M
        τ = df.τ[k]

        A_s = rand(s,τ)

        df.col_red[k] = @belapsed colredmul($P, $A_s, $w_s)

        df.row_col_red[k] = @belapsed redmul($P, $A_s, $w_s)

        df.std_mat_cr[k] = @belapsed stdmul($Pcr, $A_s)

        df.gen_pts[k] = @belapsed genpoints($Prr)
        df.row_red[k] = @belapsed rowredmul($P, $A_s, $w_s, $pts_rr)
        df.std_mat_rr[k] = @belapsed stdmul($Prr, $A_s)
        
        #df.theo_col[k] = runtime_theory_col(τ, b, m, s, w_s)
        #df.theo_row[k] = runtime_theory_row(τ, b, m, s, w_s)


        println("Finished τ=", τ)

    end
end


CSV.write("Output/runtime_$(fn_postfix).csv", df)

df = CSV.read("Output/runtime_$(fn_postfix).csv", DataFrame)

# df_reg = DataFrame(reg_row_red = regres_comp(df.τ,df.row_red), reg_col_red = regres_comp(df.τ,df.col_red), reg_red = regres_comp(df.τ,df.row_col_red), reg_std_mul = regres_comp(df.τ, df.std_mat) )

#CSV.write("Output/regression_$(fn_postfix).csv", df_reg)

colors = distinguishable_colors(7, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)[2:end]



begin
    fig = Figure()
    ax = Axis(fig[1,1], title = "", xlabel = "log τ", ylabel = "Runtime (log seconds)",xscale = log10, yscale = log10, xminorticksvisible = true, xminorgridvisible = true,
    xminorticks = IntervalsBetween(5),yminorticksvisible = true, yminorgridvisible = true,
    yminorticks = IntervalsBetween(5))
    
    plot_lines!(df.τ,df.col_red,"column reduced",:circle, colors[1])
    plot_lines!(df.τ,df.std_mat_cr,"standard_cr",:rect, colors[2])
    plot_lines!(df.τ,df.row_red,"row reduced", :xcross, colors[3])
    plot_lines!(df.τ,df.row_col_red,"row and column reduced",:rtriangle, colors[4])
    plot_lines!(df.τ,df.std_mat_rr,"standard rr",:rect, colors[5])
    plot_lines!(df.τ,df.gen_pts,"gen_pts",:rect, colors[6])


    # d_1,d_2 =  regres_theory(df.col_red, df.theo_col)
    # lines!(df.τ, d_2.*df.theo_col,linestyle = :dash, label="theoretical estimate \n column reduced",linewidth = 1.5, color = :black)

    # d_3,d_4 =  regres_theory(df.row_red, df.theo_row)
    # lines!(df.τ, d_4.*df.theo_row,linestyle = :dash, label="Theoretical estimate row reduced",linewidth = 1.5, color = :gray)

    axislegend("Matrix multiplication", merge = true, position = :lt)
    save("Output/logplot_$(fn_postfix).png", fig)
    save("Output/logplot_$(fn_postfix).svg", fig)
    fig
end


begin
    fig = Figure()
    ax = Axis(fig[1,1], title = "", xlabel = "τ", ylabel = "Runtime (seconds)" ,  yminorticksvisible = true, yminorgridvisible = true,
    yminorticks = IntervalsBetween(5))
    #ylims!(ax,0,0.07)

    plot_lines!(df.τ,df.col_red,"column reduced",:circle, colors[1])
    plot_lines!(df.τ,df.std_mat_cr,"standard",:rect, colors[2])
    plot_lines!(df.τ,df.row_red,"row reduced", :xcross, colors[3])
    plot_lines!(df.τ,df.row_col_red,"row and column reduced",:rtriangle, colors[4])
    #plot_lines!(df.τ,df.std_mat_rr,"standard rr",:rect, colors[5])
    #plot_lines!(df.τ,df.gen_pts,"gen_pts",:rect, colors[6])
    # d_1,d_2 =  regres_theory(df.col_red, df.theo_col)
    # lines!(df.τ, d_2.*df.theo_col,linestyle = :dash, label="Theoretical estimate \n column reduced",linewidth = 1.5, color = :black)

    # d_3,d_4 =  regres_theory(df.row_red, df.theo_row)
    # lines!(df.τ, d_4.*df.theo_row,linestyle = :dash, label="Theoretical estimate row reduced",linewidth = 1.5, color = :gray)
    
    axislegend("Matrix multiplication", merge = true, position = :lt)
    save("Output/linplot_$(fn_postfix).png", fig)
    save("Output/linplot_$(fn_postfix).svg", fig)
    fig
end