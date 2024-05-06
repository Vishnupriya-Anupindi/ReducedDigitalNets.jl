function runtime_theory_col(τ, b, m, s, w_s)
    st = findlast(w_s.< m)
    run_theory = 0.0
    for j in 1:st
        run_theory += τ*b^(m-w_s[j])
    end
    return run_theory
end 

function runtime_point_gen(τ, b, m, s, w_s)
    st = findlast(w_s.< m)
    run_theory = 0.0
    for j in 1:st
        run_theory += b^(m-w_s[j])*m*(m-w_s[j])
    end
    return run_theory
end 

function runtime_theory_row(τ, b, m, s, w_s)
    st = findlast(w_s.< m)
    run_theory_rr = 0.0
    for j in 1:st
        run_theory_rr += τ*b^(m-w_s[j]) + b^m
    end
    return run_theory_rr
end 

function regres_comp(s_val,T_val)
    df = DataFrame(x = log.(s_val),y = log.(T_val))
    ols = lm(@formula(y ~ x), df)
    return [exp(coef(ols)[1]), coef(ols)[2]]
end

function regres_theory(T_val, T_theory1, T_theory2)
    df = DataFrame(x = T_val,y = T_theory1, z = T_theory2)
    ols = lm(@formula(x ~ y + z), df)
    return coef(ols)[1], coef(ols)[2], coef(ols)[3]
end

function regres_theory(T_val, T_theory)
    df = DataFrame(x = T_val,y = T_theory)
    ols = lm(@formula(x ~ y), df)
    return coef(ols)[1], coef(ols)[2]
end

function plot_lines!(s_val,T_val,label,ptstyle, colour)
    lines!(s_val,T_val; linestyle = :solid, color = colour, label , linewidth = 1.5)
    scatter!(s_val,T_val; marker = ptstyle, color = colour, label , markersize = 12)
    #c_1,c_2 = regres_comp(s_val,T_val)
    #lines!(s_val,c_1.*(s_val.^c_2), color = "light gray", linestyle = :dot)
end