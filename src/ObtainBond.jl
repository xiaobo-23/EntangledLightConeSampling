## 05/11/2023
## Obtain Bond Dimension
using ITensors
using ITensors: orthocenter, sites, copy, complex, real

# Measure bond dimensions of an input MPS
function obtain_bond_dimension(ψ_input::MPS, length_input::Int)
    bond = []

    for index = 1:length_input-1
        push!(bond, dim(linkind(ψ_input, index)))
    end

    return bond
end
