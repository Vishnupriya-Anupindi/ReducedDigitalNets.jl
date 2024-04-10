
function norm_coord(v,b)
    v_1 = 0.0
    for i in eachindex(v)
        v_1 += v[i] / b^i
    end
    return v_1
end


function redmatrices(P::DigitalNetGenerator,rows,cols)
    C = [P.C[i][1:end-rows[i],1:end-cols[i]] for i in eachindex(P.C)]
    return DigitalNetGenerator(P.b,P.m,P.s,C)
end

colredmatrices(P,cols) = redmatrices(P, zeros(Int,P.s), cols)
rowredmatrices(P,rows) = redmatrices(P, rows, zeros(Int,P.s))

function repeat!(A, rows, nrep)
    for i in 2:nrep
        A[rows*(i-1)+1:rows*i,:].= @view A[1:rows,:]
    end
end


function colredmul(P::DigitalNetGenerator, A, w)
    (;b,m,s,C) = colredmatrices(P,w)

    inv_base= [ 1 / b^(i) for i in 1:m]

    τ = size(A,2)
    st = findlast(w.< m)

    w_min = minimum(w)
    P_j = zeros(b^(m - w_min), τ)
    P_j_rows = 1 

    Cn = zeros(Int64, m)
    base_n = zeros(Int64, m)
    A_j = similar(A, τ)

    #X_j = zeros(Float64,b^(m - minimum(w))) 

    @inbounds for j in st:-1:1

        # resizing P_j
        n_w = min(get(w,j+1,m), m) - w[j]
        repeat!(P_j, P_j_rows, b^n_w)
        P_j_rows *= b^n_w
        
        #Computing X_j
        resize!(base_n, m-w[j])
        C_j = C[j]
        @views A_j .= A[j,:]

        for i in 1:b^(m-w[j])
            digits!(base_n, i-1, base=b)
            mul!(Cn,C_j,base_n)
            @. Cn = Cn % b  
            X_ij = dot(Cn, inv_base)
            for k in 1:τ
                P_j[i,k] += X_ij * A_j[k]
            end
        end
        #@views q_j = X_j[1:b^(m-w[j])]*A[j:j,:] 

        # computing P_j
        #n_w = min( get(w,j+1,m) , m) - w[j]
        #P_j = repeat(P_j,b^n_w) + q_j      
        # @show size(P_j) n_w q_j 
    end
    return P_j
end


function rowredmul(P::DigitalNetGenerator, A, w, pts)
    (;b,m,s,C) = rowredmatrices(P,w)

    τ = size(A,2)
    st = findlast(w.< m)
    
    P_j = zeros(b^m,τ)
    c = [zeros(1,τ) for i in 1:b^(m- minimum(w))]
    #c = Vector{Matrix{Float64}}(undef,b^(m- minimum(w))) 
    @inbounds for j in st:-1:1
        #Computing row vectors
        for k in 0:b^(m-w[j])-1
           @views @. c[k+1] = k/(b^(m-w[j]))*A[j:j,:] # Julia has a problem with index
        end
        #Compute P_j
        for i in 1:b^m
            k_i = floor(Int,pts[i][j]*b^(m-w[j]))
            @views @. P_j[i:i,:] = P_j[i:i,:] + c[k_i + 1]
        end
    end
    return P_j
end