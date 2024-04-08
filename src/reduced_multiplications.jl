
function norm_coord(v,b,bf=float(b))
    v_1 = 0.0
    for i in eachindex(v)
        v_1 += v[i] * bf^(-i)
    end
    return v_1
end


function redmatrices(P::DigitalNetGenerator,rows,cols)
    C = [P.C[i][1:end-rows[i],1:end-cols[i]] for i in eachindex(P.C)]
    return DigitalNetGenerator(P.b,P.m,P.s,C)
end

function colredmatrices(P::DigitalNetGenerator,cols)
    C = [P.C[i][:,1:end-cols[i]] for i in eachindex(P.C)]
    return DigitalNetGenerator(P.b,P.m,P.s,C)
end

function rowredmatrices(P::DigitalNetGenerator,rows)
    C = [P.C[i][1:end-rows[i],:] for i in eachindex(P.C)]
    return DigitalNetGenerator(P.b,P.m,P.s,C)
end


function colredmul(P::DigitalNetGenerator, A, w)
    (;b,m,s,C) = colredmatrices(P,w)
    τ = size(A,2)
    st = findlast(w.< m)
    P_j = zeros(1,τ)
    Cn = zeros(Int64, m)
    base_n = zeros(Int64, m)
    X_j = zeros(Float64,b^(m - minimum(w))) 
    for j in st:-1:1
        resize!(base_n, m-w[j])
        C_j = C[j]
        for i in 1:b^(m-w[j])
            digits!(base_n, i-1, base=b)
            mul!(Cn,C_j,base_n)
            @. Cn = Cn % b  
            X_j[i] = norm_coord(Cn,b)
        end
        @views q_j = X_j[1:b^(m-w[j])]*A[j:j,:] 

        # computing P_j
        n_w = min( get(w,j+1,m) , m) - w[j]
        P_j = repeat(P_j,b^n_w) + q_j      
        # @show size(P_j) n_w q_j 
    end
    return P_j
end