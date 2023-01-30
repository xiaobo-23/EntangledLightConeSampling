## 11/28/2022
## Implement the holographic quantum circuit for the kicked Ising model
## Test first two sites which is built based on the corner case

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product
using Random

ITensors.disable_warn_order()


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
    projn_up_matrix = [1  0; 0  0]; lower_up_matrix = [0  0; 1  0]
    projn_dn_matrix = [0  0; 0  1]; raise_dn_matrix = [0  1; 0  0]
    
    result = zeros(Int, 2)
    A = m[j]
    for ind in j:j+1
        tmpS = siteind(m, ind)
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
        @show m[ind]
        if n - 1 < 1E-8
            if ind % 2 == 0
                tmpReset = ITensor(projn_up_matrix, tmpS, tmpS')
            else
                tmpReset = ITensor(lower_up_matrix, tmpS, tmpS')
            end
        else
            if ind % 2 == 1
                tmpReset = ITensor(projn_dn_matrix, tmpS, tmpS')
            else
                tmpReset = ITensor(raise_dn_matrix, tmpS, tmpS')
            end
        end
        m[ind] *= tmpReset
        noprime!(m[ind])
        # @show m[ind]
    end
    println("")
    println("")
    println("Measure sites $j and $(j+1)!")
    println("")
    println("")
    return result
end 

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

#     projn0_Matrix = [1  0; 0  0]
#     projnLower_Matrix = [0  0; 1  0]
#     # @show projectionMatrix, sizeof(projectionMatrix)
#     result = zeros(Int, 2)
#     A = m[j]
#     # @show A
#     # @show m[j]

#     for ind in j:j+1
#         tmpS = siteind(m, ind)
#         d = dim(tmpS)
#         pdisc = 0.0
#         r = rand()

#         n = 1
#         An = ITensor()
#         pn = 0.0

#         while n <= d
#             # @show A
#             # @show m[ind]
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

#         '''
#             # 1/11/2022
#             # Comment: the reset procedure needs to be revised 
#             # Use a product state of entangled (two-site) pairs and reset the state to |Psi (t=0)> instead of |up, up>. 
#         '''
#         @show m[ind]
#         if n - 1 < 1E-8
#             # tmpReset = ITensor(projn0_Matrix, s, s')
#             tmpReset = ITensor(projn0_Matrix, tmpS, tmpS')
#         else
#             # tmpReset = ITensor(projnLower_Matrix, s, s')
#             tmpReset = ITensor(projnLower_Matrix, tmpS, tmpS')
#         end
#         m[ind] *= tmpReset
#         noprime!(m[ind])
#         # @show m[ind]
#     end
#     println("")
#     println("")
#     println("Measure sites $j and $(j+1)!")
#     println("")
#     println("")
#     return result
# end 


# # Implement a long-range two-site gate
# function long_range_gate(tmp_s, position_index::Int)
#     s1 = tmp_s[1]
#     s2 = tmp_s[position_index]
    
#     # Notice the difference in coefficients due to the system is half-infinite chain
#     hj = π * op("Sz", s1) * op("Sz", s2) + 2 * h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
#     Gj = exp(-1.0im * tau / 2 * hj)
#     # @show hj
#     # @show Gj
#     @show inds(Gj)

#     # Benchmark gate that employs swap operations
#     benchmarkGate = ITensor[]
#     push!(benchmarkGate, Gj)
    
#     # for ind in 1 : n
#     #     @show s[ind], s[ind]'
#     # end

#     U, S, V = svd(Gj, (tmp_s[1], tmp_s[1]'))
#     @show norm(U*S*V - Gj)
#     # @show S
#     # @show U
#     # @show V

#     # Absorb the S matrix into the U matrix on the left
#     U = U * S
#     # @show U

#     # Make a vector to store the bond indices
#     bondIndices = Vector(undef, n - 1)

#     # Grab the bond indices of U and V matrices
#     if hastags(inds(U)[3], "Link,v") != true           # The original tag of this index of U matrix should be "Link,u".  But we absorbed S matrix into the U matrix.
#         error("SVD: fail to grab the bond indice of matrix U by its tag!")
#     else 
#         replacetags!(U, "Link,v", "i1")
#     end
#     # @show U
#     bondIndices[1] = inds(U)[3]

#     if hastags(inds(V)[3], "Link,v") != true
#         error("SVD: fail to grab the bond indice of matrix V by its tag!")
#     else
#         replacetags!(V, "Link,v", "i" * string(n))
#     end
#     # @show V
#     @show position_index
#     bondIndices[position_index - 1] = inds(V)[3]
#     # @show (bondIndices[1], bondIndices[n - 1])

    

#     #####################################################################################################################################
#     # Construct the long-range two-site gate as an MPO
#     longrangeGate = MPO(n)
#     longrangeGate[1] = U

#     for ind in 2 : position_index - 1
#         # Set up site indices
#         if abs(ind - (position_index - 1)) > 1E-8
#             bondString = "i" * string(ind)
#             bondIndices[ind] = Index(4, bondString)
#         end

#         # Make the identity tensor
#         # @show s[ind], s[ind]'
#         tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]) 
#         longrangeGate[ind] = tmpIdentity

#         # @show sizeof(longrangeGate)
#         # @show longrangeGate
#     end

#     @show typeof(V), V
#     longrangeGate[position_index] = V
#     # @show sizeof(longrangeGate)
#     # @show longrangeGate
#     @show typeof(longrangeGate), typeof(benchmarkGate)
#     #####################################################################################################################################
# end

let 
    N = 8
    cutoff = 1E-8
    tau = 1.0
    h = 0.2                                     # an integrability-breaking longitudinal field h 
    
    # Set up the circuit (e.g. number of sites, \Delta\tau used for the TEBD procedure) based on
    floquet_time = 3.0                                        # floquet time = Δτ * circuit_time
    circuit_time = 2 * Int(floquet_time)
    # circuit_time = Int(floquet_time / (0.5 * tau))
    @show floquet_time, circuit_time
    num_measurements = 1

    # 01/27/2023
    # Modify the long-range two-site gate: prefactor in the exponentiation
    function long_range_gate(tmp_s, position_index::Int)
        s1 = tmp_s[1]
        s2 = tmp_s[position_index]
        
        # Notice the difference in coefficients due to the system is half-infinite chain
        # hj = π * op("Sz", s1) * op("Sz", s2) + 2 * h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
        hj = π * op("Sz", s1) * op("Sz", s2) + h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * tau * hj)
        # Gj = exp(-1.0im * tau / 2 * hj)
        # @show hj
        # @show Gj
        # @show inds(Gj)

        # Benchmark gate that employs swap operations
        benchmarkGate = ITensor[]
        push!(benchmarkGate, Gj)
        
        # for ind in 1 : n
        #     @show s[ind], s[ind]'
        # end

        U, S, V = svd(Gj, (tmp_s[1], tmp_s[1]'))
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
        if hastags(inds(U)[3], "Link,v") != true           # The original tag of this index of U matrix should be "Link,u".  But we absorbed S matrix into the U matrix.
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
            tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]) 
            longrangeGate[ind] = tmpIdentity

            # @show sizeof(longrangeGate)
            # @show longrangeGate
        end

        # @show typeof(V), V
        longrangeGate[position_index] = V
        # @show sizeof(longrangeGate)
        # @show longrangeGate
        # @show typeof(longrangeGate), typeof(benchmarkGate)
        #####################################################################################################################################
        return longrangeGate
    end
    
    
    ###############################################################################################################################
    ## Constructing gates used in the TEBD algorithm
    ###############################################################################################################################
    # # Construct a two-site gate that implements Ising interaction and longitudinal field
    # gates = ITensor[]
    # for ind in 1:(N - 1)
    #     s1 = s[ind]
    #     s2 = s[ind + 1]

    #     if (ind - 1 < 1E-8)
    #         tmp1 = 2 
    #         tmp2 = 1
    #     elseif (abs(ind - (N - 1)) < 1E-8)
    #         tmp1 = 1
    #         tmp2 = 2
    #     else
    #         tmp1 = 1
    #         tmp2 = 1
    #     end

    #     println("")
    #     println("Coefficients are $(tmp1) and $(tmp2)")
    #     println("Site index is $(ind) and the conditional sentence is $(ind - (N - 1))")
    #     println("")

    #     hj = π * op("Sz", s1) * op("Sz", s2) + tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2)
    #     Gj = exp(-1.0im * tau / 2 * hj)
    #     push!(gates, Gj)
    # end
    # # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    # append!(gates, reverse(gates))


    # # Construct the transverse field as the kicked gate
    # ampo = OpSum()
    # for ind in 1 : N
    #     ampo += π/2, "Sx", ind
    # end
    # Hₓ = MPO(ampo, s)
    # Hamiltonianₓ = Hₓ[1]
    # for ind in 2 : N
    #     Hamiltonianₓ *= Hₓ[ind]
    # end
    # expHamiltoinianₓ = exp(-1.0im * Hamiltonianₓ)
    

    #################################################################################################################################################
    # Construct the holographic quantum dynamics simulation (holoQUADS) circuit
    #################################################################################################################################################
    # Construct time evolution for one floquet time step
    # function timeEvolutionCorner(numGates :: Int, numSites :: Int, tmp_gates)
    #     # In the corner case, two-site gates are applied to site 1 --> site N
    #     # gates = ITensor[]
    #     if 2 * numGates >= numSites
    #         error("the number of time evolution gates is larger than what can be accommodated based on the number of sites!")
    #     end

    #     for ind₁ in 1:2
    #         for ind₂ in 1:numGates
    #             parity = (ind₁ - 1) % 2
    #             s1 = s[2 * ind₂ - parity]; @show inds(s1)
    #             s2 = s[2 * ind₂ + 1 - parity]; @show inds(s2)

    #             if 2 * ind₂ - parity - 1 < 1E-8
    #                 coeff₁ = 2
    #                 coeff₂ = 1
    #             else
    #                 coeff₁ = 1
    #                 coeff₂ = 1
    #             end

    #             hj = (π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2))
    #             Gj = exp(-1.0im * tau / 2 * hj)
    #             push!(tmp_gates, Gj)
    #         end
    #     end
    #     # return gates
    # end


    # Construct time evolution for one floquet time step
    function time_evolution_corner(num_gates :: Int, parity :: Int)
        # Time evolution using TEBD for the corner case
        gates = ITensor[]

        for ind₁ in 1 : num_gates
            s1 = s[2 * ind₁ - parity]
            s2 = s[2 * ind₁ + 1 - parity]
            @show inds(s1)
            @show inds(s2)

            if 2 * ind₁ - parity - 1 < 1E-8
                coeff₁ = 2
                coeff₂ = 1
            else
                coeff₁ = 1
                coeff₂ = 1
            end

            # hj = (π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2))
            # Gj = exp(-1.0im * tau / 2 * hj)
            hj = coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2)
            Gj = exp(-1.0im * tau * hj)
            push!(gates, Gj)
        end
        return gates
    end

    
    function time_evolution(initialPosition :: Int, numSites :: Int, tmp_sites)
        # General time evolution using TEBD
        gates = ITensor[]
        if initialPosition - 1 < 1E-8
            tmpGate = long_range_gate(tmp_sites, numSites)
            # push!(gates, tmpGate)
            return tmpGate
        else
            # if initialPosition - 2 < 1E-8
            #     coeff₁ = 1
            #     coeff₂ = 2
            # else
            #     coeff₁ = 1
            #     coeff₂ = 1
            # end
            s1 = tmp_sites[initialPosition]
            s2 = tmp_sites[initialPosition - 1]
            # hj = (π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2))
            # hj = (π * op("Sz", s1) * op("Sz", s2) + h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2))
            hj = h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
            Gj = exp(-1.0im * tau * hj)                   # Correcting the factor in the exponentiation
            push!(gates, Gj)
        end
        return gates
    end

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false)

    # # Construct the kicked gate that applies transverse Ising fields at integer time using single-site gate
    # kick_gate = ITensor[]
    # for ind in 1 : N
    #     s1 = s[ind]
    #     hamilt = π / 2 * op("Sx", s1)
    #     tmpG = exp(-1.0im * hamilt)
    #     push!(kick_gate, tmpG)
    # end
    
    # Construct a multi-site kicked gate to apply the transverse Ising fields at integer times
    # Constructed for the corner part of the holoQAUDS circuit
    function build_kick_gates(starting_index :: Int, ending_index :: Int)
        kick_gate = ITensor[]
        for ind in starting_index : ending_index
            s1 = s[ind]; @show ind
            hamilt = π / 2 * op("Sx", s1)
            tmpG = exp(-1.0im * hamilt)
            push!(kick_gate, tmpG)
        end
        return kick_gate
    end

    # Construct a two-site kick gate to apply the transverse Ising fields at integer times
    # Constructed for the diagonal parts of the holoQUADS circuit
    function build_two_site_kick_gate(starting_index :: Int, period :: Int)
        # Index arranged in decreasing order due to the speciifc structure of the diagonal parity
        two_site_kick_gate = ITensor[]
        
        ending_index = (starting_index - 1 + period) % period
        if ending_index == 0
            ending_index = period
        end
        index_list = [starting_index, ending_index]

        for index in index_list
            s1 = s[index]
            hamilt = π / 2 * op("Sx", s1)
            tmpG = exp(-1.0im * hamilt)
            push!(two_site_kick_gate, tmpG)
        end
        return two_site_kick_gate
    end    

    '''
        # Sample from the time-evolved wavefunction and store the measurements
    '''
    # timeSlices = Int(floquet_time / tau) + 1; println("Total number of time slices that need to be saved is : $(timeSlices)")
    # Sx = complex(zeros(timeSlices, N))
    # Sy = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(timeSlices, N))
    # Cxx = complex(zeros(timeSlices, N))
    # Czz = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(num_measurements, N))
    Sz_sample = real(zeros(num_measurements, N))

    '''
        # Measure expectation values of the wavefunction during time evolution
    '''
    Sx = complex(zeros(Int(floquet_time) * Int(N / 2), N))
    Sy = complex(zeros(Int(floquet_time) * Int(N / 2), N))
    Sz = complex(zeros(Int(floquet_time) * Int(N / 2), N))
    # Sx = Vector{Float64}[]
    # Sy = Vector{Float64}[]
    # Sz = Vector{Float64}[]


    # Initialize the wavefunction
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = MPS(s, states)
    Sz₀ = expect(ψ, "Sz"; sites = 1 : N)
    # Random.seed!(10000)

    # ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # @show eltype(ψ), eltype(ψ[1])
    
    # # Initializa a random MPS
    # Random.seed!(200); 
    # initialization_s = siteinds("S=1/2", 8; conserve_qns = false)
    # initialization_states = [isodd(n) ? "Up" : "Dn" for n = 1 : 8]
    # initialization_ψ = randomMPS(initialization_s, initialization_states, linkdims = 2)
    # ψ = initialization_ψ[1 : N]
    # # @show maxlinkdim(ψ)

    # Random.seed!(200)
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # # states = [isodd(n) ? "X+" : "X-" for n = 1 : N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # Sz₀ = expect(ψ, "Sz"; sites = 1 : N)                    # Take measurements of the initial random MPS
    # Random.seed!(10000)


    for measure_ind in 1 : num_measurements
        println("")
        println("")
        println("############################################################################")
        println("#########           PERFORMING MEASUREMENTS LOOP #$measure_ind           ##############")
        println("############################################################################")
        println("")
        println("")

        # Compute the overlap between the original and time evolved wavefunctions
        ψ_copy = deepcopy(ψ)
        ψ_overlap = Complex{Float64}[]
        
        @time for ind in 1 : circuit_time
            tmp_overlap = abs(inner(ψ, ψ_copy))
            println("The inner product is: $tmp_overlap")
            append!(ψ_overlap, tmp_overlap)

            # # Local observables e.g. Sx, Sz
            # tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); @show tmpSx; # Sx[index, :] = tmpSx
            # tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); @show tmpSy; # Sy[index, :] = tmpSy
            # tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); @show tmpSz; # Sz[index, :] = tmpSz

            # # Apply kicked gate at integer times
            # if ind % 2 == 1
            #     ψ_copy = apply(kick_gate, ψ_copy; cutoff)
            #     normalize!(ψ_copy)
            #     println("")
            #     println("")
            #     println("Applying the kicked Ising gate at time $(ind)!")
            #     tmp_overlap = abs(inner(ψ, ψ_copy))
            #     @show tmp_overlap
            #     println("")
            #     println("")
            # end

            # Apply a sequence of two-site gates
            tmp_parity = (ind - 1) % 2
            tmp_num_gates = Int(circuit_time / 2) - floor(Int, (ind - 1) / 2) 
            print(""); @show tmp_num_gates; print("")

            # Apply kicked gate at integer times
            if ind % 2 == 1
                tmp_kick_gate = build_kick_gates(1, 2 * tmp_num_gates + 1); @show 2 * tmp_num_gates + 1
                # tmp_kick_gate = build_kick_gates(1, N)
                ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                normalize!(ψ_copy)
                # ψ_copy = apply(kick_gate, ψ_copy; cutoff)

                println("")
                println("")
                println("Applying the kicked Ising gate at time $(ind)!")
                tmp_overlap = abs(inner(ψ, ψ_copy))
                @show tmp_overlap
                println("")
                println("")
            end
            
            tmp_two_site_gates = ITensor[]
            tmp_two_site_gates = time_evolution_corner(tmp_num_gates, tmp_parity)
            println("")
            println("")
            @show sizeof(tmp_two_site_gates)
            println("")
            println("")
            println("Appling the Ising gate plus longitudinal fields.")
            println("")
            println("")
            # println("")
            # @show tmp_two_site_gates 
            # @show typeof(tmp_two_site_gates)
            # println("")

            # println("")
            # println("")
            # tmp_overlap = abs(inner(ψ, ψ_copy))
            # @show tmp_overlap
            # println("")
            # println("")

            ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)

            if ind % 2 == 0
                Sx[Int(ind / 2), :] = expect(ψ_copy, "Sx"; sites = 1 : N)
                Sy[Int(ind / 2), :] = expect(ψ_copy, "Sy"; sites = 1 : N) 
                Sz[Int(ind / 2), :] = expect(ψ_copy, "Sz"; sites = 1 : N); @show Sz[Int(ind / 2), :]
            end
            # if ind % 2 == 0
            #     push!(Sx, expect(ψ_copy, "Sx"; sites = 1 : N))
            #     push!(Sy, expect(ψ_copy, "Sy"; sites = 1 : N))
            #     push!(Sz, expect(ψ_copy, "Sz"; sites = 1 : N))
            # end
        end

        println("")
        println("")
        tmp_overlap = abs(inner(ψ, ψ_copy))
        @show tmp_overlap
        println("")
        Sz_sample[measure_ind, 1:2] = sample(ψ_copy, 1)
        println("")
        tmp_overlap = abs(inner(ψ, ψ_copy))
        @show tmp_overlap
        println("")
        println("")

        # Sx[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sx"; sites = 1 : N);
        # Sy[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sy"; sites = 1 : N); 
        # Sz[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sz"; sites = 1 : N); # @show real(Sz[4, :]) 

        @time for ind₁ in 1 : 3
            gate_seeds = []
            for gate_ind in 1 : circuit_time
                tmp_ind = (2 * ind₁ - gate_ind + N) % N
                if tmp_ind == 0
                    tmp_ind = N
                end
                push!(gate_seeds, tmp_ind)
            end
            println("")
            println("")
            println("#########################################################################################")
            @show gate_seeds
            println("#########################################################################################")
            println("")
            println("")

            for ind₂ in 1 : circuit_time
                # Apply the kicked gate at integer time
                if ind₂ % 2 == 1
                    tmp_kick_gate₁ = build_two_site_kick_gate(gate_seeds[ind₂], N)
                    ψ_copy = apply(tmp_kick_gate₁, ψ_copy; cutoff)
                    normalize!(ψ_copy)
                end

                # Apply the Ising interaction and longitudinal fields using a sequence of two-site gates
                tmp_two_site_gates = time_evolution(gate_seeds[ind₂], N, s)
                ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
                normalize!(ψ_copy)
                # println("")
                # println("")
                # @show tmp_two_site_gates 
                # @show typeof(tmp_two_site_gates)
                # println("")
                # println("")

                if ind₂ % 2 == 0
                    Sx[Int(ind₂ / 2) + Int(floquet_time) * ind₁, :] = expect(ψ_copy, "Sx"; sites = 1 : N);
                    Sy[Int(ind₂ / 2) + Int(floquet_time) * ind₁, :] = expect(ψ_copy, "Sy"; sites = 1 : N); 
                    Sz[Int(ind₂ / 2) + Int(floquet_time) * ind₁, :] = expect(ψ_copy, "Sz"; sites = 1 : N);
                    @show Int(ind₂/2) + Int(floquet_time) * ind₁
                    # @show real(Sz[Int(ind₂ / 2) + 3 * ind₁ + 1, :]) 
                end

                # if ind₂ % 2 == 0
                #     push!(Sx, expect(ψ_copy, "Sx"; sites = 1 : N))
                #     push!(Sy, expect(ψ_copy, "Sy"; sites = 1 : N))
                #     push!(Sz, expect(ψ_copy, "Sz"; sites = 1 : N))
                # end
            end
            Sz_sample[measure_ind, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, 2 * ind₁ + 1)
        end

        # @time for ind₁ in 1 : Int(N / 2) - 1
        #     gate_seeds = []
        #     for gate_ind in 1 : circuit_time
        #         tmp_ind = (2 * ind₁ - gate_ind + N) % N
        #         if tmp_ind == 0
        #             tmp_ind = N
        #         end
        #         push!(gate_seeds, tmp_ind)
        #     end
        #     println("")
        #     println("")
        #     @show gate_seeds
        #     println("")
        #     println("")

        #     for ind₂ in 1 : circuit_time
        #         # tmp_overlap = abs(inner(ψ, ψ_copy))
        #         # println("The inner product is: $tmp_overlap")
        #         # append!(ψ_overlap, tmp_overlap)

        #         # # Local observables e.g. Sx, Sz
        #         # tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); @show tmpSx; # Sx[index, :] = tmpSx
        #         # tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); @show tmpSy; # Sy[index, :] = tmpSy
        #         # tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); @show tmpSz; # Sz[index, :] = tmpSzz

        #         # Apply kicked gate at integer times
        #         if ind₂ % 2 == 1
        #             ψ_copy = apply(kick_gate, ψ_copy; cutoff)
        #             normalize!(ψ_copy)
        #             println("")
        #             tmp_overlap = abs(inner(ψ, ψ_copy))
        #             @show tmp_overlap
        #             println("")
        #         end

        #         # Apply a sequence of two-site gates
        #         # tmp_two_site_gates = ITensor[]
        #         tmp_two_site_gates = time_evolution(gate_seeds[ind₂], N, s)
        #         # println("")
        #         # @show tmp_two_site_gates 
        #         # @show typeof(tmp_two_site_gates)
        #         # println("")

        #         ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
        #         normalize!(ψ_copy)
        #         # println("")
        #         # tmp_overlap = abs(inner(ψ, ψ_copy))
        #         # @show tmp_overlap
        #         # println("")
        #     end
        #     Sz[measure_ind, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, 2 * ind₁ + 1) 
        #     # println("")
        #     # @show abs(inner(ψ, ψ_copy))
        #     # println("")
        # end
    end
    # replace!(Sz_sample, 1.0 => 0.5, 2.0 => -0.5)
     

    println("################################################################################")
    println("################################################################################")
    println("Information of the initial random MPS")
    @show Sz₀
    println("################################################################################")
    println("################################################################################")
    
    # Store data in hdf5 file
    file = h5open("Data/holoQUADS_Circuit_N$(N)_h$(h)_T$(floquet_time)_Rotations_Only_AFM.h5", "w")
    write(file, "Initial Sz", Sz₀)
    write(file, "Sx", Sx)
    write(file, "Sy", Sy)
    write(file, "Sz", Sz)
    # write(file, "Cxx", Cxx)
    # write(file, "Czz", Czz)
    # write(file, "Wavefunction Overlap", ψ_overlap)
    close(file)

    return
end  