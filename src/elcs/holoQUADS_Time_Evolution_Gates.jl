## 05/11/2023
## Layers of gates that are used in the holoQUADS circuits. 
## Including the left light cone, diagonal parts, right light cone and layers of kicked gates

using ITensors
using ITensorMPS

# Contruct layers of two-site gates including the Ising interaction and longitudinal fileds in the left light cone.
function left_light_cone(
    number_of_gates::Int,
    parity::Int,
    longitudinal_field::Float64,
    Δτ::Float64,
    tmp_sites,
)
    gates = ITensor[]

    for ind = 1:number_of_gates
        tmp_index = 2 * ind - parity
        s1 = tmp_sites[tmp_index]
        s2 = tmp_sites[tmp_index+1]

        if abs(tmp_index - 1) < 1E-8
            coeff = 2
            @show tmp_index, coeff
        else
            coeff = 1
            # @show coeff
        end

        # hj = coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        # hj = π * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        hj = π * op("Sz", s1) * op("Sz", s2) + coeff * longitudinal_field * op("Sz", s1) * op("Id", s2) + longitudinal_field * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * Δτ * hj)
        push!(gates, Gj)
    end
    return gates
end


# Construct a two-site gate in the right light cone
function diagonal_right_edge(
    input_index::Int,
    total_sites::Int,
    longitudinal_field::Float64,
    Δτ::Float64,
    tmp_sites,
)
    gates = ITensor[]
    s1 = tmp_sites[input_index-1]
    s2 = tmp_sites[input_index]

    if abs(input_index - total_sites) < 1E-8
        coeff = 2
        @show input_index, coeff
    else
        coeff = 1
    end

    # hj = longitudinal_field * op("Sz", s1) * op("Id", s2) + longitudinal_field * op("Id", s1) * op("Sz", s2)
    # hj = π * op("Sz", s1) * op("Sz", s2) + longitudinal_field * op("Sz", s1) * op("Id", s2) + longitudinal_field * op("Id", s1) * op("Sz", s2)
    hj = π * op("Sz", s1) * op("Sz", s2) + longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff * longitudinal_field * op("Id", s1) * op("Sz", s2)
    Gj = exp(-1.0im * Δτ * hj)
    push!(gates, Gj)

    return gates
end


# # Construct layers of two-site gate including the Ising interaction and longitudinal fields in the right light cone
# function right_light_cone(starting_index :: Int, number_of_gates :: Int, edge_index :: Int, longitudinal_field :: Float64, Δτ :: Float64, tmp_sites)
#     gates = ITensor[]

#     for ind in 1 : number_of_gates
#         tmp_start = starting_index - 2 * (ind - 1)
#         tmp_end = tmp_start - 1

#         s1 = tmp_sites[tmp_end]
#         s2 = tmp_sites[tmp_start]

#         # Consider the finite-size effect on the right edge
#         if abs(tmp_start - edge_index) < 1E-8
#             coeff₁ = 1
#             coeff₂ = 2
#         else
#             coeff₁ = 1
#             coeff₂ = 1
#         end

#         # hj = coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
#         # hj = π * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
#         hj = π/2 * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
#         Gj = exp(-1.0im * Δτ * hj)
#         push!(gates, Gj)
#     end
#     return gates
# end


# Construct multiple one-site gates to apply the transverse Ising fields.
function build_kick_gates(starting_index::Int, ending_index::Int, tmp_sites)
    kick_gate = ITensor[]

    for ind = starting_index:ending_index
        s1 = tmp_sites[ind]
        hamilt = π / 2 * op("Sx", s1)
        tmpG = exp(-1.0im * hamilt)
        push!(kick_gate, tmpG)
    end

    return kick_gate
end