## 08/18/2023
## Measure the bond dimension and von Neumann entanglment entropy
using ITensors
using ITensors: orthocenter, sites, copy, complex, real

include("Entanglement.jl")
include("ObtainBond.jl")

# Meausre the bond dimension chi and von Neumann entanglment Entropy
function measure_chi_and_SvN!(input_ψ :: MPS, input_index :: Int, sample_index :: Int, input_SvN, input_bond)
    input_SvN[
        sample_index, 
        (input_index - 1) * (N - 1) + 1 : input_index * (N - 1),
    ] = entanglement_entropy(input_ψ, N)

    input_bond[
        sample_index, 
        (input_index - 1) * (N - 1) + 1 : input_index * (N - 1),
    ] = obtain_bond_dimension(input_ψ, N)
end