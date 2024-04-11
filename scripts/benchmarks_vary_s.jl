using ReducedDigitalNets
using ReducedDigitalNets: stdmul
using BenchmarkTools
using Statistics
using CairoMakie
using GLM, StatsBase, DataFrames, CSV
using Colors

include("utils.jl")

mkpath("Output")

case = 11
b = 2
s_range = 1600
step_size = 200
m = 12
fn_postfix = "case$(case)_m$(m)_s$(s_range)_b$(b)"

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.1
#BenchmarkTools.DEFAULT_PARAMETERS.samples = 2

τ = 20
M = length(1:step_size:s_range) 

df = DataFrame(s = collect(1:step_size:s_range), gen_pts=zeros(M), row_red = zeros(M), col_red = zeros(M), row_col_red = zeros(M), std_mat = zeros(M), std_mat_pts = zeros(M), theo_col = zeros(M), theo_row = zeros(M))

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
    ax = Axis(fig[1,1], title = "", xlabel = "log s", ylabel = "Runtime (log seconds)",xscale = log10, yscale = log10, xminorticksvisible = true, xminorgridvisible = true,
    xminorticks = IntervalsBetween(5),yminorticksvisible = true, yminorgridvisible = true,
    yminorticks = IntervalsBetween(5))
    
    plot_lines!(df.s,df.col_red,"Column reduced mm product",:circle, colors[1])
    plot_lines!(df.s,df.std_mat,"Standard mm product",:rect, colors[2])
    plot_lines!(df.s,df.row_red,"Row reduced mm product", :xcross, colors[3])
    plot_lines!(df.s,df.row_col_red,"Row and column reduced mm product",:rtriangle, colors[4])

    d_1,d_2 =  regres_theory(df.col_red, df.theo_col)
    lines!(df.s, d_2.*df.theo_col,linestyle = :dash, label="Theoretical estimate column reduced",linewidth = 1.5, color = :black)

    d_3,d_4 =  regres_theory(df.row_red, df.theo_row)
    lines!(df.s, d_4.*df.theo_row,linestyle = :dash, label="Theoretical estimate row reduced",linewidth = 1.5, color = :gray)

    axislegend(ax, merge = true, position = :lt)
    save("Output/logplot_$(fn_postfix).png", fig)
    fig
end


begin
    fig = Figure()
    ax = Axis(fig[1,1], title = "", xlabel = "s", ylabel = "Runtime (log seconds)" , yscale = log10, yminorticksvisible = true, yminorgridvisible = true,
    yminorticks = IntervalsBetween(5))


    plot_lines!(df.s,df.col_red,"Column reduced mm product",:circle, colors[1])
    plot_lines!(df.s,df.std_mat,"Standard mm product",:rect, colors[2])
    plot_lines!(df.s,df.row_red,"Row reduced mm product", :xcross, colors[3])
    plot_lines!(df.s,df.row_col_red,"Row and column reduced mm product",:rtriangle, colors[4])
    
    d_1,d_2 =  regres_theory(df.col_red, df.theo_col)
    lines!(df.s, d_2.*df.theo_col,linestyle = :dash, label="Theoretical estimate column reduced",linewidth = 1.5, color = :black)

    d_3,d_4 =  regres_theory(df.row_red, df.theo_row)
    lines!(df.s, d_4.*df.theo_row,linestyle = :dash, label="Theoretical estimate row reduced",linewidth = 1.5, color = :gray)
    
    axislegend(ax, merge = true, position = :rb)
    save("Output/semilog_plot_$(fn_postfix).png", fig)
    fig
end