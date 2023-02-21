## 02/07/2023
## Implement the holographic quantum circuit for the Heisenberg model

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product, Float64
using Random
ITensors.disable_warn_order()




 # # Sample and reset one two-site MPS
# function sample(m::MPS, j::Int)
#     mpsLength = length(m)

#     # Move the orthogonality center of the MPS to site j
#     orthogonalize!(m, j)
#     if orthocenter(m) != j
#         error("sample: MPS m must have orthocenter(m) == 1")
#     end
    
#     # Check the normalization of the MPS
#     if abs(1.0 - norm(m[j])) > 1E-8
#         error("sample: MPS is not normalized, norm=$(norm(m[1]))")
#     end
 
#     '''
#         # Take measurements of two-site MPS and reset the MPS to its initial state
#     '''
#     projn_up_matrix = [
#         1  0 
#         0  0
#     ]
    
#     S⁻_matrix = [
#         0  0 
#         1  0
#     ]

#     projn_dn_matrix = [
#         0  0 
#         0  1
#     ] 
    
#     S⁺_matrix = [
#         0  1 
#         0  0
#     ]
    
#     result = zeros(Int, 2)
#     A = m[j]
#     for ind in j:j+1
#         tmpS = siteind(m, ind)
#         d = dim(tmpS)
#         pdisc = 0.0
#         r = rand()

#         n = 1 
#         An = ITensor()
#         pn = 0.0

#         while n <= d
#             projn = ITensor(tmpS)
#             projn[tmpS => n] = 1.0
#             An = A * dag(projn)
#             pn = real(scalar(dag(An) * An))
#             pdisc += pn

#             (r < pdisc) && break
#             n += 1
#         end
#         result[ind - j + 1] = n
#         # @show result[ind - j + 1]
#         # @show An

#         if ind < mpsLength
#             A = m[ind + 1] * An
#             A *= (1. / sqrt(pn))
#         end

#         # Resetting procedure
#         if ind % 2 == 1
#             if n - 1 < 1E-8
#                 tmpReset = ITensor(projn_up_matrix, tmpS', tmpS)
#             else
#                 tmpReset = ITensor(S⁺_matrix, tmpS', tmpS)
#             end
#         else
#             if n - 1 < 1E-8
#                 tmpReset = ITensor(S⁻_matrix, tmpS', tmpS)
#             else
#                 tmpReset = ITensor(projn_dn_matrix, tmpS', tmpS)
#             end
#         end
        
#         m[ind] *= tmpReset
#         noprime!(m[ind])
#         # @show m[ind]
#     end
#     return result
# end 


# Sample and reset one two-site MPS
function sample(m::MPS, j::Int)
    mpsLength = length(m)

    # Move the orthogonality center of the MPS to site j
    orthogonalize!(m, j)
    if orthocenter(m) != j
        error("sample: MPS m must have orthocenter(m) == 1")
    end
    # Check the normalization of the MPS
    if abs(1.0 - norm(m[j])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[1]))")
    end

    # Take measurements and reset the two-site MPS to |up, down> Neel state
    # Need to be modified based on the initialization of MPS
    projn_up_matrix = [
        1  0 
        0  0
    ] 
    S⁻_matrix = [
        0  0
        1  0
    ]
    projn_dn_matrix = [
        0  0 
        0  1
    ] 
    S⁺_matrix = [
        0  1 
        0  0
    ]
    
    result = zeros(Int, 2)
    A = m[j]
    for ind in j:j+1
        tmpS = siteind(m, ind)
        # println("Before taking measurements")
        # @show(m[ind])
        d = dim(tmpS)
        pdisc = 0.0
        r = rand()

        n = 1 
        An = ITensor()
        pn = 0.0

        while n <= d
            projn = ITensor(tmpS)
            projn[tmpS => n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind - j + 1] = n
        # @show result[ind - j + 1]
        # @show An

        if ind < mpsLength
            A = m[ind + 1] * An
            A *= (1. / sqrt(pn))
        end

        '''
            # 01/27/2022
            # Comment: the reset procedure needs to be revised 
            # Use a product state of entangled (two-site) pairs and reset the state to |Psi (t=0)> instead of |up, down>. 
        '''
        
        # n denotes the corresponding physical state: n=1 --> |up> and n=2 --> |down>
        if ind % 2 == 1
            if n - 1 < 1E-8             
                tmpReset = ITensor(projn_up_matrix, tmpS', tmpS)
            else
                tmpReset = ITensor(S⁺_matrix, tmpS', tmpS)
            end
        else
            if n - 1 < 1E-8
                tmpReset = ITensor(S⁻_matrix, tmpS', tmpS)
            else
                tmpReset = ITensor(projn_dn_matrix, tmpS', tmpS)
            end
        end
        m[ind] *= tmpReset
        noprime!(m[ind])
        # println("After resetting")
        # @show m[ind]
    end
    # println("")
    # println("")
    # println("Measure sites $j and $(j+1)!")
    # println("")
    # println("")
    return result
end 


# Construct layers of two-site gates for the corner part of the holoQAUDS circuit
function construct_corner_layer(starting_index :: Int, ending_index :: Int, temp_sites, Δτ :: Float64)
    gates = ITensor[]
    for j in starting_index : 2 : ending_index
        @show starting_index, j, j + 1
        temp_s1 = temp_sites[j]
        temp_s2 = temp_sites[j + 1]

        temp_hj = op("Sz", temp_s1) * op("Sz", temp_s2) + 1/2 * op("S+", temp_s1) * op("S-", temp_s2) + 1/2 * op("S-", temp_s1) * op("S+", temp_s2)
        temp_Gj = exp(-1.0im * Δτ * temp_hj)
        push!(gates, temp_Gj)
    end
    return gates
end


# Construct layers of two-site gates for the diagonal part of the holoQUADS circuit
function construct_diagonal_layer(starting_index :: Int, ending_index :: Int, temp_sites, Δτ :: Float64)
    gates = ITensor[]
    if starting_index - 1 < 1E-8
        @show starting_index, ending_index
        tmp_gate = long_range_gate(temp_sites, ending_index, Δτ)
        return tmp_gate
    else
        @show starting_index, ending_index
        temp_s1 = temp_sites[starting_index]
        temp_s2 = temp_sites[ending_index]
        temp_hj = op("Sz", temp_s1) * op("Sz", temp_s2) + 1/2 * op("S+", temp_s1) * op("S-", temp_s2) + 1/2 * op("S-", temp_s1) * op("S+", temp_s2)
        temp_Gj = exp(-1.0im * Δτ * temp_hj)
        push!(gates, temp_Gj)
    end
    return gates
end


# 02/14/2023
# Modify the long-range two-site gate: prefactor in the exponentiation
function long_range_gate(tmp_site, position_index::Int, Δτ :: Float64)
    s1 = tmp_site[1]
    s2 = tmp_site[position_index]
    
    # Define the two-site Hamiltonian and set up a long-range gate
    hj = op("Sz", s1) * op("Sz", s2) + 1/2 * op("S+", s1) * op("S-", s2) + 1/2 * op("S-", s1) * op("S+", s2)
    Gj = exp(-1.0im * Δτ * hj)

    # Benchmark gate that employs swap operations
    benchmarkGate = ITensor[]
    push!(benchmarkGate, Gj)

    U, S, V = svd(Gj, (tmp_site[1], tmp_site[1]'))
    # @show norm(U*S*V - Gj)
    # @show S
    # @show U
    # @show V

    # Absorb the S matrix into the U matrix on the left
    U = U * S
    # @show U

    # Make a vector to store the bond indices
    bondIndices = Vector(undef, position_index - 1)

    # Grab the bond indices of U and V matrices
    if hastags(inds(U)[3], "Link,v") != true    # The original tag of this index of U matrix should be "Link,u".  But we absorbed S matrix into the U matrix.
        error("SVD: fail to grab the bond indice of matrix U by its tag!")
    else 
        replacetags!(U, "Link,v", "i1")
    end
    # @show U
    bondIndices[1] = inds(U)[3]

    if hastags(inds(V)[3], "Link,v") != true
        error("SVD: fail to grab the bond indice of matrix V by its tag!")
    else
        replacetags!(V, "Link,v", "i" * string(position_index))
    end
    # @show V
    # @show position_index
    bondIndices[position_index - 1] = inds(V)[3]
    # @show (bondIndices[1], bondIndices[n - 1])

    #####################################################################################################################################
    # Construct the long-range two-site gate as an MPO
    longrangeGate = MPO(position_index)
    longrangeGate[1] = U

    for ind in 2 : position_index - 1
        # Set up site indices
        if abs(ind - (position_index - 1)) > 1E-8
            bondString = "i" * string(ind)
            bondIndices[ind] = Index(4, bondString)
        end

        # Make the identity tensor
        # @show s[ind], s[ind]'
        tmpIdentity = delta(tmp_site[ind], tmp_site[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]) 
        longrangeGate[ind] = tmpIdentity
    end

    # @show typeof(V), V
    longrangeGate[position_index] = V
    #####################################################################################################################################
    return longrangeGate
end


# Compute the overlap between the time-evolved wavefunction and 
function compute_overlap(tmp_ψ₁::MPS, tmp_ψ₂::MPS)
    overlap = abs(inner(tmp_ψ₁, tmp_ψ₂))
    println("")
    println("")
    @show overlap
    println("")
    println("")
    return overlap
end


let 
    #####################################################################################################################################
    ##### Define parameters used in the holoQUADS circuit
    ##### Given the light-cone structure of the real-time dynamics, circuit depth and number of sites are related/intertwined
    floquet_time = 1.5
    tau = 0.05                                                                                  # time step used for Trotter decomposition
    N_time_slice = Int(floquet_time / tau) * 2
    N = N_time_slice + 2
    N_half_infinite = N; N_diagonal_circuit = div(N_half_infinite - 2, 2)
    cutoff = 1E-8
    @show floquet_time, typeof(floquet_time)
    num_measurements = 1 
    #####################################################################################################################################
    
    # Make an array of 'site' indices && quantum numbers are CONSERVED for the Heisenberg model
    s = siteinds("S=1/2", N; conserve_qns = false)
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = MPS(s, states)
    Sz₀ = expect(ψ, "Sz"; sites = 1 : N)
    
    # Random.seed!(1234567)
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # Sz₀ = expect(ψ, "Sz"; sites = 1 : N)                    # Take measurements of the initial random MPS
    # Random.seed!(8000000)

    #####################################################################################################################################
    # Sample from the time-evolved wavefunction and store the measurements
    #####################################################################################################################################
    # Sx = complex(zeros(timeSlices, N))
    # Sy = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(timeSlices, N))
    # Cxx = complex(zeros(timeSlices, N))
    # Czz = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(num_measurements, N))
    Sz_sample = real(zeros(num_measurements, N_half_infinite))

    #####################################################################################################################################
    # Sample from the time-evolved wavefunction and store the measurements
    #####################################################################################################################################    
    Sx = complex(zeros(div(N_half_infinite, 2), N))
    Sy = complex(zeros(div(N_half_infinite, 2), N))
    Sz = complex(zeros(div(N_half_infinite, 2), N))
    Sz_Reset = complex(zeros(div(N_half_infinite, 2), N))

    # ## Construct the holoQUADS circuit 
    # ## Consider to move this part outside the main function in the near future
    
    # Random.seed!(2000)
    for measure_ind in 1 : num_measurements
        println("")
        println("")
        println("############################################################################")
        println("#########  PERFORMING MEASUREMENTS LOOP #$measure_ind ")
        println("############################################################################")
        println("")
        println("")

        # Compute the overlap between the original and time evolved wavefunctions
        ψ_copy = deepcopy(ψ)
        ψ_overlap = Complex{Float64}[]
        
        tmp_overlap = abs(inner(ψ, ψ_copy))
        println("The overlap of wavefuctions @T=0 is: $tmp_overlap")
        append!(ψ_overlap, tmp_overlap)

        @time for ind₁ in 1 : div(N_time_slice, 2)
            number_of_gates = div(N_time_slice, 2) - (ind₁ - 1); @show number_of_gates
            for tmp_index in [2, 1]
                tmp_starting_index = tmp_index
                tmp_ending_index = tmp_starting_index + 2 * number_of_gates - 1
                corner_gate = construct_corner_layer(tmp_starting_index, tmp_ending_index, s, tau)
                ψ_copy = apply(corner_gate, ψ_copy; cutoff)
            end
        end
        normalize!(ψ_copy) 

        if measure_ind - 1 < 1E-8
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
            Sz[1, :] = tmp_Sz
        end
        # Sz_sample[measure_ind, 1 : 2] = sample(ψ_copy, 1)
        # if measure_ind - 1 < 1E-8
        #     tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
        #     Sz_Reset[1, :] = tmp_Sz
        # end
        # normalize!(ψ_copy)
        

        @time for ind₁ in 1 : N_diagonal_circuit
            gate_seeds = []
            for gate_ind in 1 : N_time_slice
                tmp_ind = (2 * ind₁ - gate_ind + N) % N
                if tmp_ind == 0
                    tmp_ind = N
                end
                push!(gate_seeds, tmp_ind)
            end
            println("")
            println("")
            println("#########################################################################################")
            @show size(gate_seeds)[1]
            println("#########################################################################################")
            println("")
            println("")


            for ind₂ in 1 : N_time_slice
                tmp_starting_index = gate_seeds[ind₂]
                if tmp_starting_index - 1 < 1E-8
                    tmp_ending_index = N
                else
                    tmp_ending_index = tmp_starting_index - 1
                end
                @show tmp_starting_index, tmp_ending_index
                diagonal_gate = construct_diagonal_layer(tmp_starting_index, tmp_ending_index, s, tau)
                ψ_copy = apply(diagonal_gate, ψ_copy; cutoff)
            end
            normalize!(ψ_copy) 

            if measure_ind - 1 < 1E-8
                tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
                Sz[ind₁ + 1, :] = tmp_Sz
                println(""); @show tmp_Sz
            end
            # Sz_sample[measure_ind, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, 2 * ind₁ + 1)
            # if measure_ind - 1 < 1E-8
            #     tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
            #     Sz_Reset[ind₁ + 1, :] = tmp_Sz
            #     println(""); @show tmp_Sz
            # end
            # normalize!(ψ_copy)
        end
    end
    replace!(Sz_sample, 1.0 => 0.5, 2.0 => -0.5)
     

    println("################################################################################")
    println("################################################################################")
    println("Projection in the Sz basis of the initial MPS")
    @show Sz₀
    println("################################################################################")
    println("################################################################################")
    
    # Store data in hdf5 file
    file = h5open("Data/holoQUADS_Circuit_Heisenberg_N$(N)_T$(floquet_time)_tau$(tau)_AFM_Initialization_Sample&Reset.h5", "w")
    write(file, "Initial Sz", Sz₀)
    # write(file, "Sx", Sx)
    # write(file, "Sy", Sy)
    write(file, "Sz", Sz)
    # write(file, "Cxx", Cxx)
    # write(file, "Cyy", Cyy)
    # write(file, "Czz", Czz)
    write(file, "Sz_sample", Sz_sample)
    # write(file, "Wavefunction Overlap", ψ_overlap)
    close(file)

    return
end  