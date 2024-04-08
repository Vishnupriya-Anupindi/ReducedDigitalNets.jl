module ReducedDigitalNets

using LinearAlgebra


include("digitalnets.jl")
include("reduced_multiplications.jl")

export DigitalNetGenerator
export redmatrices, colredmatrices, rowredmatrices
export colredmul, rowredmul, redmul
export genpoints, redmat


end

## Use using Revise everytime you start. 