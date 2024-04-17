using BenchmarkTools
using StaticArrays

b=2
m=12
w = 8
τ = 20
Nt = b^(m-w) 
#Example c_k vector
c = [@SVector rand(τ) for k in 1:Nt]

P = [@SVector zeros(τ) for k in 1:b^m]

y = rand(b^m)

function linearacc(c,P,Nt,b,m)
    for j in 1:b^m
        P[j] = P[j] + c[mod(j,1:Nt)]
    end
    return nothing
end

@btime linearacc($c,$P,$Nt,$b,$m)
# 15.772 μs (0 allocations: 0 bytes)

ridx = [rand(1:Nt) for i in 1:b^m]

function randacc(c,P,Nt,b,m, ridx)
    for j in 1:b^m
        P[j] = P[j] + c[ridx[j]]
    end
    return nothing
end

@btime rand(1:$Nt)
#4.714 ns (0 allocations: 0 bytes)
@btime randacc($c,$P,$Nt,$b,$m, $ridx)
# 28.455 μs (0 allocations: 0 bytes)
# 15.553 μs (0 allocations: 0 bytes)


function randmul(c,P,Nt,b,m,y,ridx)
    for j in 1:b^m
        P[j] = P[j] + c[ridx[j]]*y[j]
    end
    return nothing
end

@btime randmul($c,$P,$Nt,$b,$m, $y, $ridx)
# 31.524 μs (0 allocations: 0 bytes)
# 15.850 μs (0 allocations: 0 bytes)


# For m=16,τ = 100, w=4
# 3.704 ms (0 allocations: 0 bytes)
# 4.595 ns (0 allocations: 0 bytes)
# 3.652 ms (0 allocations: 0 bytes)
# 3.701 ms (0 allocations: 0 bytes)

function randshortmul(c,P,Nt,b,m, ridx, y)
    for k in 1:Nt
        c[k] = c[k]*y[k]
    end

    for j in 1:b^m
        P[j] = P[j] + c[ridx[j]]
    end
    return nothing
end

@btime randshortmul($c,$P,$Nt,$b,$m, $ridx, $y)
# 32.510 μs (0 allocations: 0 bytes)
# m=16,τ = 100, w=4: 3.885 ms (0 allocations: 0 bytes)

# m=12, w = 4, τ = 20
# 15.668 μs (0 allocations: 0 bytes)
# 4.633 ns (0 allocations: 0 bytes)
# 15.698 μs (0 allocations: 0 bytes)
# 16.363 μs (0 allocations: 0 bytes)
# 34.690 μs (0 allocations: 0 bytes)


# m=12, w = 8, τ = 20
# 13.539 μs (0 allocations: 0 bytes)
# 4.597 ns (0 allocations: 0 bytes)
# 13.474 μs (0 allocations: 0 bytes)
# 13.935 μs (0 allocations: 0 bytes)
# 14.656 μs (0 allocations: 0 bytes)

