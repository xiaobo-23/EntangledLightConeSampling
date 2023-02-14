## 02/07/2023
## Implement the holographic quantum circuit for the Heisenberg model

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product, Float64
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
 
    '''
        # Take measurements of two-site MPS and reset the MPS to its initial state
    '''
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

        # Resetting procedure
        if ind % 2 == 1
            if n == 1
                tmpReset = ITensor(projn_up_matrix, tmpS', tmpS)
            else
                tmpReset = ITensor(raise_dn_matrix, tmpS', tmpS)
            end
        else
            if n == 1
                tmpReset = ITensor(lower_up_matrix, tmpS', tmpS)
            else
                tmpReset = ITensor(projn_dn_matrix, tmpS', tmpS)
            end
        end
        
        m[ind] *= tmpReset
        noprime!(m[ind])
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
    for j in starting_index : ending_index
        temp_s1 = temp_site[j]
        temp_s2 = temp_site[j + 1]

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
        tmp_gate = long_range_gate(starting_index, ending_index, Δτ)
        return tmp_gate
    else
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
        tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]) 
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
    ##### Define parameters that will be used in the simulation
    N = 8; floquet_time = Float((N-2)//2); circuit_time = 2 * Int(floquet_time)
    tau = 0.1                                               # time step used for Trotter decomposition
    cutoff = 1E-8
    @show floquet_time, circuit_time
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
    # timeSlices = Int(floquet_time / tau) + 1; println("Total number of time slices that need to be saved is : $(timeSlices)")
    # Sx = complex(zeros(timeSlices, N))
    # Sy = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(timeSlices, N))
    # Cxx = complex(zeros(timeSlices, N))
    # Czz = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(num_measurements, N))
    Sz_sample = real(zeros(num_measurements, 2 * N))

    #####################################################################################################################################
    # Sample from the time-evolved wavefunction and store the measurements
    #####################################################################################################################################    
    Sx = complex(zeros(Int(floquet_time) * Int(N / 2), N))
    Sy = complex(zeros(Int(floquet_time) * Int(N / 2), N))
    Sz = complex(zeros(Int(floquet_time) * Int(N / 2), N))

    ## Construct the holoQUADS circuit 
    ## Consider to move this part outside the main function in the near future
    
    # Random.seed!(2000)
    for measure_ind in 1 : num_measurements
        println("")
        println("")
        println("############################################################################")
        println("#########          PERFORMING MEASUREMENTS LOOP #$measure_ind          ##############")
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
                ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                normalize!(ψ_copy)
                # ψ_copy = apply(kick_gate, ψ_copy; cutoff)
                
                println("Applying the kicked Ising gate at time $(ind)!")
                compute_overlap(ψ, ψ_copy)
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

            compute_overlap(ψ,ψ_copy)
            ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)

            if measure_ind == 1 && ind % 2 == 0
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

        compute_overlap(ψ,ψ_copy)
        Sz_sample[measure_ind, 1:2] = sample(ψ_copy, 1)
        compute_overlap(ψ,ψ_copy)

        # Sx[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sx"; sites = 1 : N);
        # Sy[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sy"; sites = 1 : N); 
        # Sz[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sz"; sites = 1 : N); # @show real(Sz[4, :]) 

        @time for ind₁ in 1 : 11
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

                if measure_ind == 0 && ind₂ % 2 == 0
                    Sx[Int(ind₂ / 2) + Int(floquet_time) * ind₁, :] = expect(ψ_copy, "Sx"; sites = 1 : N);
                    Sy[Int(ind₂ / 2) + Int(floquet_time) * ind₁, :] = expect(ψ_copy, "Sy"; sites = 1 : N); 
                    Sz[Int(ind₂ / 2) + Int(floquet_time) * ind₁, :] = expect(ψ_copy, "Sz"; sites = 1 : N);
                    @show Int(ind₂/2) + Int(floquet_time) * ind₁
                    # @show real(Sz[Int(ind₂ / 2) + 3 * ind₁ + 1, :]) 
                end
            end
            index_to_sample = (2 * ind₁ + 1) % N
            println("############################################################################")
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
            @show index_to_sample
            @show tmp_Sz[index_to_sample : index_to_sample + 1]
            println("****************************************************************************")
            Sz_sample[measure_ind, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, index_to_sample)
            # Sz_sample[measure_ind, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, 2 * ind₁ + 1)
            println("############################################################################")
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
            @show tmp_Sz[index_to_sample : index_to_sample + 1]
            println("****************************************************************************")
        end
    end
    replace!(Sz_sample, 1.0 => 0.5, 2.0 => -0.5)
     

    println("################################################################################")
    println("################################################################################")
    println("Projection in the Sz basis of the initial MPS")
    @show Sz₀
    println("################################################################################")
    println("################################################################################")
    
    # # Store data in hdf5 file
    # file = h5open("Data/holoQUADS_Circuit_N$(N)_h$(h)_T$(floquet_time)_Rotations_Only_Sample_AFM_v1.h5", "w")
    # write(file, "Initial Sz", Sz₀)
    # write(file, "Sx", Sx)
    # write(file, "Sy", Sy)
    # write(file, "Sz", Sz)
    # # write(file, "Cxx", Cxx)
    # # write(file, "Cyy", Cyy)
    # # write(file, "Czz", Czz)
    # write(file, "Sz_sample", Sz_sample)
    # # write(file, "Wavefunction Overlap", ψ_overlap)
    # close(file)

    return
end  