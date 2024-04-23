using ReducedDigitalNets
using ReducedDigitalNets: stdmul
using BenchmarkTools
using Statistics
using CairoMakie
using GLM, StatsBase, DataFrames, CSV
using Colors

include("utils.jl")

mkpath("Output")

case = 13
b = 2
s_range = 1600
n_steps = 9
step_size = 200
m = 12
fn_postfix = "case$(case)_m$(m)_s$(s_range)_b$(b)_exps"

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1
#BenchmarkTools.DEFAULT_PARAMETERS.samples = 2

τ = 20

#For linear s range
#s = collect(1:step_size:s_range)
#M = length(1:step_size:s_range) 

#For exponential s range
s_exp = floor.(Int, 10 .^LinRange(0,4,n_steps))
M = length(s_exp)

df = DataFrame(s = s_exp, gen_pts=zeros(M), row_red = zeros(M), col_red = zeros(M), row_col_red = zeros(M), std_mat = zeros(M), std_mat_pts = zeros(M), theo_col = zeros(M), theo_row = zeros(M))

begin 

    for k in 1:M
        s = df.s[k]

        C = Matrix{Int64}[]
        for i in 1:s
            C_i = rand(0:1,m,m)
            push!(C,C_i)
        end

        P = DigitalNetGenerator(b,m,s,C)

        A_s = rand(s,τ) 
        
        #w_s = @. min(floor(Int64,log2(1:s)),m)

        if case == 11
            w_s = @. min(floor(Int64,log2(1:s)),m)
        elseif case == 12
            w_s = @. min(floor(Int64,(log2(1:s))^(1/2)),m)
        elseif case == 13
            w_s = @. min(floor(Int64,(log2(1:s))^(1/4)),m)
        end

        Pcr = colredmatrices(P,w_s)
        Prr = rowredmatrices(P,w_s)
        st = findlast(w_s.< m)

        df.col_red[k] = @belapsed colredmul($P, $A_s, $w_s)

        df.row_col_red[k] = @belapsed redmul($P, $A_s, $w_s)

        df.std_mat[k] = @belapsed stdmul($Pcr, $A_s)

        pts = genpoints(Prr)
        df.gen_pts[k] = @belapsed genpoints($Prr)
        df.row_red[k] = @belapsed rowredmul($P, $A_s, $w_s, $pts)
        df.std_mat_pts[k] = @belapsed stdmul($Prr, $A_s, $pts)
        
        df.theo_col[k] = runtime_theory_col(τ, b, m, s, w_s)
        df.theo_row[k] = runtime_theory_row(τ, b, m, s, w_s)


        println("Finished s=", s)

    end
end


CSV.write("Output/runtime_$(fn_postfix).csv", df)

df = CSV.read("Output/runtime_$(fn_postfix).csv", DataFrame)

df_reg = DataFrame(reg_row_red = regres_comp(df.s,df.row_red), reg_col_red = regres_comp(df.s,df.col_red), reg_red = regres_comp(df.s,df.row_col_red), reg_std_mul = regres_comp(df.s, df.std_mat) )

CSV.write("Output/regression_$(fn_postfix).csv", df_reg)

colors = distinguishable_colors(5, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)[2:end]



begin
    fig = Figure()
    ax = Axis(fig[1,1], title = "", xlabel = "s (log scale)", ylabel = "Runtime in seconds (log scale)",xscale = log10, yscale = log10, xminorticksvisible = true, xminorgridvisible = true,
    xminorticks = IntervalsBetween(5),yminorticksvisible = true, yminorgridvisible = true,
    yminorticks = IntervalsBetween(5))
    
    plot_lines!(df.s,df.col_red,"column reduced",:circle, colors[1])
    plot_lines!(df.s,df.std_mat,"standard",:rect, colors[2])
    plot_lines!(df.s,df.row_red,"row reduced", :xcross, colors[3])
    plot_lines!(df.s,df.row_col_red,"row and column reduced",:rtriangle, colors[4])

    d_1,d_2 =  regres_theory(df.col_red, df.theo_col)
    lines!(df.s, d_2.*df.theo_col,linestyle = :dash, label="theoretical estimate \n column reduced",linewidth = 1.5, color = :black)

    # d_3,d_4 =  regres_theory(df.row_red, df.theo_row)
    # lines!(df.s, d_4.*df.theo_row,linestyle = :dash, label="Theoretical estimate row reduced",linewidth = 1.5, color = :gray)

    axislegend("Matrix multiplication", merge = true, position = :lt)
    save("Output/logplot_$(fn_postfix).png", fig)
    save("Output/logplot_$(fn_postfix).svg", fig)
    fig
end


# begin
#     fig = Figure()
#     ax = Axis(fig[1,1], title = "", xlabel = "s", ylabel = "Runtime (log seconds)" , yscale = log10, yminorticksvisible = true, yminorgridvisible = true,
#     yminorticks = IntervalsBetween(5))


#     plot_lines!(df.s,df.col_red,"Column reduced",:circle, colors[1])
#     plot_lines!(df.s,df.std_mat,"Standard",:rect, colors[2])
#     plot_lines!(df.s,df.row_red,"Row reduced", :xcross, colors[3])
#     plot_lines!(df.s,df.row_col_red,"Row and column reduced",:rtriangle, colors[4])
    
#     d_1,d_2 =  regres_theory(df.col_red, df.theo_col)
#     lines!(df.s, d_2.*df.theo_col,linestyle = :dash, label="Theoretical estimate \n column reduced",linewidth = 1.5, color = :black)

#     # d_3,d_4 =  regres_theory(df.row_red, df.theo_row)
#     # lines!(df.s, d_4.*df.theo_row,linestyle = :dash, label="Theoretical estimate row reduced",linewidth = 1.5, color = :gray)
    
#     axislegend("matrix multiplication", merge = true, position = :rc)
#     save("Output/semilog_plot_$(fn_postfix).png", fig)
#     fig
# end