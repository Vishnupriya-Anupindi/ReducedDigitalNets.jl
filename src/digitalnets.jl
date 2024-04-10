struct DigitalNetGenerator
    b::Int
    m::Int
    s::Int
    C::Vector{Matrix{Int64}}
end

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