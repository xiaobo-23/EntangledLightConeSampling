## 08/18/2023
## Measure the bond dimension and von Neumann entanglment entropy
using ITensors
using ITensors: orthocenter, sites, copy, complex, real

include("Entanglement.jl")
include("ObtainBond.jl")
include("Sample.jl")

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

# Measure bond dimension and von Neumann entanglement entropy before and after measuring the state
# Generate bitstrings and strings of fractional numbers

function sample_and_measure!(input_ψ :: MPS, input_index :: Int, sample_index :: Int, input_measurement_type :: AbstractString,
    input_SvN, input_bond, input_samples, input_samples_bitstring)
    
    # Obtain bond dimension and von Neumann entanglment entropy
    measure_chi_and_SvN!(input_ψ, input_index, sample_index, input_SvN, input_bond)
    
    # Generate samples from wavefunction
    ## To-do list: right now we are only able to sample Sz, which is hard-coded 
    ##             we need to update this procedure later to be able to sample Sx and Sy
    input_samples[sample_index, input_index : input_index + 1] = (
        expect(input_ψ, input_measurement_type; sites = input_index : input_index + 1)
    )
    input_samples_bitstring[sample_index, input_index : input_index + 1] = (
        sample(input_ψ, input_index)
    )
    normalize!(input_ψ)

    # Obtain bond dimension and von Neumann entanglement entropy 
    measure_chi_and_SvN!(input_ψ, input_index + 1, sample_index, input_SvN, input_bond)
end