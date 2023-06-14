## 05/11/2023
## Layers of gates that are used in the holoQUADS circuits. 
## Including the left light cone, diagonal parts, right light cone and layers of kicked gates

using ITensors
using ITensors: orthocenter, sites, copy, complex, real

function build_a_layer_of_gates(
    starting_index::Int,
    ending_index::Int,
    upper_bound::Int,
    amplitude::Real,
    delta_tau::Real,
    tmp_sites::AbstractVector,
)
    tmp_gates = []
    for ind = starting_index:2:ending_index
        s1 = tmp_sites[ind]
        s2 = tmp_sites[ind+1]

        if (ind - 1 < 1E-8)
            tmp1 = 2
            tmp2 = 1
        elseif (abs(ind - (upper_bound - 1)) < 1E-8)
            tmp1 = 1
            tmp2 = 2
        else
            tmp1 = 1
            tmp2 = 1
        end

        # hj = π * op("Sz", s1) * op("Sz", s2) + tmp1 * amplitude * op("Sz", s1) * op("Id", s2) + tmp2 * amplitude * op("Id", s1) * op("Sz", s2) 
        hj =
            π / 2 * op("Sz", s1) * op("Sz", s2) +
            tmp1 * amplitude * op("Sz", s1) * op("Id", s2) +
            tmp2 * amplitude * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * delta_tau * hj)
        push!(tmp_gates, Gj)
    end
    return tmp_gates
end

# Build a sequence of one-site kick gates
function build_kick_gates(
    tmp_site::AbstractVector,
    starting_index::Integer,
    ending_index::Integer,
)
    tmp_gates = ITensor[]
    for index = starting_index:ending_index
        tmpS = tmp_site[index]
        tmpHamiltonian = π / 2 * op("Sx", tmpS)
        tmpGate = exp(-1.0im * tmpHamiltonian)
        push!(tmp_gates, tmpGate)
    end
    return tmp_gates
end
