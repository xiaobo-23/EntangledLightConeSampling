## 08/15/2023
## Set up the left lightcone, diagonal parts, and right lightcone

using ITensors
using ITensors: orthocenter, sites, copy, complex, real

## Construct a layer of two-site gates with an arbitrary number of gates
function construct_layers_of_gates(starting_index :: Int, number_of_gates :: Int, tmp_sites, input_gates)
    # gates = ITensor[]
    for j = 1 : number_of_gates
        index1 = starting_index + 2 * (j - 1)
        index2 = index1 + 1
       
        tmp1 = tmp_sites[index1]
        tmp2 = tmp_sites[index2]

        tmp_hj = op("Sz", tmp1) * op("Sz", tmp2) + 1/2 * op("S+", tmp1) * op("S-", tmp2) + 1/2 * op("S-", tmp1) * op("S+", tmp2)
        tmp_Gj = exp(-1.0im * Δτ * tmp_hj)
        push!(input_gates , tmp_Gj)
        # push!(gates, tmp_Gj)
    end
    # return gates
end

# Construct the left lightcone part of the holoQUADS circuit
function construct_left_lightcone(input_ψ :: MPS, tmp_time_slices :: Int, input_sites)
    @time for ind₁ in 1:div(tmp_time_slices, 2)
        @show ind₁
        gates = ITensor[]
        tmp_number_of_gates = div(tmp_time_slices, 2) - (ind₁ - 1)
        for tmp_index in [2, 1]
            construct_layers_of_gates(tmp_index, tmp_number_of_gates, input_sites, gates)
        end
        input_ψ = apply(gates, input_ψ; running_cutoff)
    end
end


# Construct the diagonal parts of the holoQUADS circuits
function construct_diagonal_part(input_ψ :: MPS, tmp_time_slices :: Int, input_index :: Int, input_sites)
    @show input_index
    gates = ITensor[]
    for temporary_index in 1:tmp_time_slices
        starting_index = unit_cell_size + 2 * (input_index - 1) - (temporary_index - 1)
        construct_layers_of_gates(starting_index, 1, input_sites, gates)
    end
    input_ψ = apply(gates, input_ψ; running_cutoff)
    normalize!(input_ψ) 
end


# Construct the right lightcone of the holoQUADS circuit
function construct_right_lightcone(input_ψ :: MPS, input_index :: Int, tmp_time_slices :: Int, input_sites)
    
    @show input_index
    gates = ITensor[]
    tmp_number_of_gates = tmp_time_slices - 2 * input_index + 1
    for tmp_index=1:tmp_number_of_gates
        starting_index = N - tmp_index
        construct_layers_of_gates(starting_index, 1, input_sites, gates)
    end
    input_ψ = apply(gates, input_ψ; running_cutoff)
    normalize!(input_ψ)
end


# # The long-range two-site gate is only used when the recycle procedure is turned on
# function long_range_gate(tmp_site, position_index::Int, Δτ::Float64)
#     s1 = tmp_site[1]
#     s2 = tmp_site[position_index]

#     # Define the two-site Hamiltonian and set up a long-range gate
#     hj =
#         op("Sz", s1) * op("Sz", s2) +
#         1 / 2 * op("S+", s1) * op("S-", s2) +
#         1 / 2 * op("S-", s1) * op("S+", s2)
#     Gj = exp(-1.0im * Δτ * hj)

#     # Benchmark gate that employs swap operations
#     benchmarkGate = ITensor[]
#     push!(benchmarkGate, Gj)

#     U, S, V = svd(Gj, (tmp_site[1], tmp_site[1]'))
#     # @show norm(U*S*V - Gj)
#     # @show S
#     # @show U
#     # @show V

#     # Absorb the S matrix into the U matrix on the left
#     U = U * S
#     # @show U

#     # Make a vector to store the bond indices
#     bondIndices = Vector(undef, position_index - 1)

#     # Grab the bond indices of U and V matrices
#     if hastags(inds(U)[3], "Link,v") != true    # The original tag of this index of U matrix should be "Link,u".  But we absorbed S matrix into the U matrix.
#         error("SVD: fail to grab the bond indice of matrix U by its tag!")
#     else
#         replacetags!(U, "Link,v", "i1")
#     end
#     # @show U
#     bondIndices[1] = inds(U)[3]

#     if hastags(inds(V)[3], "Link,v") != true
#         error("SVD: fail to grab the bond indice of matrix V by its tag!")
#     else
#         replacetags!(V, "Link,v", "i" * string(position_index))
#     end
#     # @show V
#     # @show position_index
#     bondIndices[position_index-1] = inds(V)[3]
#     # @show (bondIndices[1], bondIndices[n - 1])

#     #####################################################################################################################################
#     # Construct the long-range two-site gate as an MPO
#     longrangeGate = MPO(position_index)
#     longrangeGate[1] = U

#     for ind = 2:position_index-1
#         # Set up site indices
#         if abs(ind - (position_index - 1)) > 1E-8
#             bondString = "i" * string(ind)
#             bondIndices[ind] = Index(4, bondString)
#         end

#         # Make the identity tensor
#         # @show s[ind], s[ind]'
#         tmpIdentity =
#             delta(tmp_site[ind], tmp_site[ind]') *
#             delta(bondIndices[ind-1], bondIndices[ind])
#         longrangeGate[ind] = tmpIdentity
#     end

#     # @show typeof(V), V
#     longrangeGate[position_index] = V
#     #####################################################################################################################################
#     return longrangeGate
# end