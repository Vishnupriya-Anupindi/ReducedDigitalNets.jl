using ReducedDigitalNets
using ReducedDigitalNets: stdmul
using Test

#@testset "ReducedDigitalNets.jl" begin
#    # Write your tests here.
#end

@testset "point generation" begin
    C = [[1 0; 0 1], [0 1; 1 0]]
    b = 2
    m = 2
    s = 2
    P = DigitalNetGenerator(b,m,s,C)
    pts = genpoints(P)
    @test pts == [[0.0, 0.0], [0.5, 0.25],[0.25; 0.5], [0.75, 0.75]]
    
end

@testset "ReducedMatrices" begin
    P = DigitalNetGenerator(2,3,2, [[1 0 1; 0 1 1; 1 1 1], [1 1 0; 0 0 1; 1 1 1]])
    rows = [0,1]
    cols = [0,2]
    P_1 = redmatrices(P,rows,cols)
    @test all(size(P_1.C[i]) == (P.m - rows[i], P.m-cols[i]) for i in eachindex(P.C))
end

@testset "column reduced computation" begin 
    P = DigitalNetGenerator(2,2,2,[[1 0; 0 1], [0 1; 1 0]])
    A = [1 2 3; 2 5 1]
    w = [0,1]
    prod_alg = colredmul(P, A, w)
    PT = colredmatrices(P,w)
    @test prod_alg == stdmul(PT,A)
end


@testset "row reduced multiplication" begin 
    P = DigitalNetGenerator(2,2,2,[[1 0; 0 1], [0 1; 1 0]])
    A = [1 2 3; 2 5 1]
    w = [0,1]
    PT = rowredmatrices(P,w)
    ptsred = genpoints(PT)
    prod_alg = rowredmul(P, A, w, ptsred)
    @test prod_alg == stdmul(PT,A)
end

@testset "row+column reduced computation" begin 
    P = DigitalNetGenerator(2,2,2,[[1 0; 0 1], [0 1; 1 0]])
    A = [1 2 3; 2 5 1]
    w = [0,1]
    prod_alg = redmul(P, A, w)
    PT = redmatrices(P,w,w)
    @test prod_alg == stdmul(PT,A)
end

