struct DigitalNetGenerator
    b::Int
    m::Int
    s::Int
    C::Vector{Matrix{Int64}}
end

""" 
This function computes the dot product of vector v with ``(\\frac{1}{b},\\dots,\\frac{1}{b^m})``
"""
function norm_coord(v,b)
    v_1 = 0.0
    for i in eachindex(v)
        v_1 += v[i] / b^i
    end
    return v_1
end

""" 
This function generates the points of the digital net using the generating matrices.
"""
function genpoints(P::DigitalNetGenerator)
    (;b,m,s,C) = P
    Cn = zeros(Int64, m)
    base_n = zeros(Int64, m)
    pts = [zeros(s) for i in 1:b^m]
    for k in eachindex(pts) 
        digits!(base_n, k-1, base=b)
        for i in eachindex(C)
            resize!(Cn, size(C[i],1))
            mul!(Cn, C[i], @view base_n[1:size(C[i],2)])
            @. Cn = Cn % b
            pts[k][i] = norm_coord(Cn, b)
        end
    end
    return pts
end


function redmatrices(P::DigitalNetGenerator,rows,cols)
    C = [P.C[i][1:end-rows[i],1:end-cols[i]] for i in eachindex(P.C)]
    return DigitalNetGenerator(P.b,P.m,P.s,C)
end

colredmatrices(P,cols) = redmatrices(P, zeros(Int,P.s), cols)
rowredmatrices(P,rows) = redmatrices(P, rows, zeros(Int,P.s))