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
        error("sample: MPS m must have orthocenter(m) == j")
    end
    # Check the normalization of the MPS
    if abs(1.0 - norm(m[j])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[j]))")
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


# Construct the kicked gates to apply transverse Ising fields in the right corner of the holoQUADS circuit
function kick_gates_right_corner(starting_index :: Int, number_of_gates :: Int, period :: Int, tmp_sites)
    kick_gate = ITensor[]

    # Creat a vector of site indices to deal with *periodic* boundary condition in physical sites
    index_container = []
    for ind in 1 : 2 * number_of_gates + 1
        index_to_add = (starting_index - ind + 1 + period) % period
        if index_to_add < 1E-8
            index_to_add = period
        end
        push!(index_container, index_to_add)
    end

    println("")
    println("")
    println("***************************************************************************")
    println("Applying kick gats at sites #")
    @show index_container
    println("***************************************************************************")
    println("")
    println("")

    # Loop through all the sites in the container 
    for ind in index_container
        # println("Apply kicked gates on sites")
        @show ind
        s1 = tmp_sites[ind]
        hamilt = π / 2 * op("Sx", s1)
        tmpG = exp(-1.0im * hamilt)
        push!(kick_gate, tmpG)
    end
    return kick_gate
end


# Check the overlap between time-evolved wavefunction and the original wavefunction
function compute_overlap(tmp_ψ₁::MPS, tmp_ψ₂::MPS)
    overlap = abs(inner(tmp_ψ₁, tmp_ψ₂))
    println("")
    println("")
    @show overlap
    println("")
    println("")
    return overlap
end


# Constructing the gate that applies the transverse Ising fields to multiple sites
# Used in the corner part of the holoQUADS circuit
function build_kick_gates(starting_index :: Int, ending_index :: Int, tmp_sites)
    kick_gate = ITensor[]
    for ind in starting_index : ending_index
        s1 = tmp_sites[ind]; @show ind
        hamilt = π / 2 * op("Sx", s1)
        tmpG = exp(-1.0im * hamilt)
        push!(kick_gate, tmpG)
    end
    return kick_gate
end


let 
    floquet_time = 4.0                                                                 # floquet time = Δτ * circuit_time
    circuit_time = 2 * Int(floquet_time)
    N = 2 * Int(floquet_time) + 2       # the size of an unit cell that is determined by time and the lightcone structure
    N_diagonal = 9                                                              # the number of diagonal parts of circuit
    N_total = N + 2 * N_diagonal; site_tensor_index = 0
    cutoff = 1E-8
    tau = 1.0
    h = 0.2                                                              # an integrability-breaking longitudinal field h 
    
    # Set up the circuit (e.g. number of sites, \Delta\tau used for the TEBD procedure) based on
    
    # @show floquet_time, circuit_time
    num_measurements = 1

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false)

    # Construct the gate to apply transverse Ising fields to two sites at integer times
    # Used in the diaognal part of the holoQUADS circuit
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


    # Construct two-site gates to apply the Ising interaction and longitudinal gates in the right corner of the holoQUADS circuit 
    function layers_right_corner(starting_index :: Int, edge_index :: Int, number_of_gates :: Int, period :: Int, tmp_sites)
        # gates = ITensor[]
        gates = Any[]
        for ind in 1 : number_of_gates
            tmp_start = (starting_index - 2 * (ind - 1) + period) % period
            tmp_end = (tmp_start - 1 + period) % period 

            if tmp_start < 1E-8
                tmp_start = period
            elseif tmp_end < 1E-8
                tmp_end = period
            end

            println("Apply two-site gates to sites $(tmp_start) and $(tmp_end)")
            s1 = tmp_sites[tmp_end]
            s2 = tmp_sites[tmp_start]

            if abs(tmp_start - edge_index) < 1E-8
                println("********************************************************************************")
                println("Yeah!")
                @show tmp_start
                println("********************************************************************************")
                println("")
                coeff₁ = 1
                coeff₂ = 2
            else
                coeff₁ = 1
                coeff₂ = 1
            end

            if abs(tmp_start - 1) < 1E-8
                Gj = long_range_gate(tmp_sites, period)
            else
                @show tmp_start, tmp_end
                # hj = coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2)
                hj = π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2)
                Gj = exp(-1.0im * tau * hj)
            end
            push!(gates, Gj)
        end
        return gates
    end

    
    # Construct the left corner of the holoQUADS circuit for the holoQUADS model
    function time_evolution_corner(num_gates :: Int, parity :: Int, tmp_sites)
        gates = ITensor[]

        for ind₁ in 1 : num_gates
            s1 = tmp_sites[2 * ind₁ - parity]
            s2 = tmp_sites[2 * ind₁ + 1 - parity]
            @show inds(s1)
            @show inds(s2)

            if 2 * ind₁ - parity - 1 < 1E-8
                coeff₁ = 2
                coeff₂ = 1
            else
                coeff₁ = 1
                coeff₂ = 1
            end

            # hj = coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2)
            hj = π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2)
            Gj = exp(-1.0im * tau * hj)
            push!(gates, Gj)
        end
        return gates
    end

    
    ## Construct the diagonal part of the holoQUADS circuit for the SDKI model.
    function time_evolution(initial_position :: Int, num_sites :: Int, tmp_sites)
        gates = ITensor[]

        if initial_position - 1 < 1E-8
            # Generate a long-range two-site gate
            tmp_gate = long_range_gate(tmp_sites, num_sites)
            return tmp_gate
        else
            # Generate a local two-site gate 
            s1 = tmp_sites[initial_position]
            s2 = tmp_sites[initial_position - 1]

            # hj = h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
            hj = π * op("Sz", s1) * op("Sz", s2) + h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
            Gj = exp(-1.0im * tau * hj)                 
            push!(gates, Gj)
        end
        return gates
    end


    ## Long-range two-site gate 
    function long_range_gate(tmp_s, position_index::Int)
        s1 = tmp_s[1]
        s2 = tmp_s[position_index]
        
        println("")
        println("")
        println("***************************************************************************")
        println("Applying a long-range two-site gate!!")
        println("***************************************************************************")
        println("")
        println("")

        # Use bulk coefficients to define this long-range gate
        # hj = h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
        hj = π * op("Sz", s1) * op("Sz", s2) + h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * tau * hj)
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
            tmpIdentity = delta(tmp_s[ind], tmp_s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]) 
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


    # # Construct the kicked gate that applies transverse Ising fields at integer time using single-site gate
    # kick_gate = ITensor[]
    # for ind in 1 : N
    #     s1 = s[ind]
    #     hamilt = π / 2 * op("Sx", s1)
    #     tmpG = exp(-1.0im * hamilt)
    #     push!(kick_gate, tmpG)
    # end
        
    # '''
    #     # Sample from the time-evolved wavefunction and store the measurements
    # '''
    # timeSlices = Int(floquet_time / tau) + 1; println("Total number of time slices that need to be saved is : $(timeSlices)")
    # Sx = complex(zeros(timeSlices, N))
    # Sy = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(timeSlices, N))
    # Cxx = complex(zeros(timeSlices, N))
    # Czz = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(num_measurements, N))
    Sz_sample = real(zeros(num_measurements, N_total))

    # '''
    #     # Measure expectation values of the wavefunction during time evolution
    # '''
    # Sz_Reset = complex(zeros(Int(N/2), N))
    # Sx = complex(zeros(3, N_total))
    # Sy = complex(zeros(3, N_total))
    # Sz = complex(zeros(3, N_total))
    Sx = complex(zeros(N_total))
    Sy = complex(zeros(N_total))
    Sz = complex(zeros(N_total))

    # Initialize the wavefunction
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = MPS(s, states)
    Sz₀ = expect(ψ, "Sz"; sites = 1 : N)
    # Random.seed!(10000)

    # ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # @show eltype(ψ), eltype(ψ[1])
    
    # # Initializa a random MPS
    # # initialization_s = siteinds("S=1/2", N; conserve_qns = false)
    # initialization_states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # Random.seed!(87900) 
    # ψ = randomMPS(s, initialization_states, linkdims = 2)
    # # ψ = initialization_ψ[1 : N]
    # Sz₀ = expect(ψ, "Sz"; sites = 1 : N)
    # # @show maxlinkdim(ψ)

    # Random.seed!(1234567)
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # # states = [isodd(n) ? "X+" : "X-" for n = 1 : N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # Sz₀ = expect(ψ, "Sz"; sites = 1 : N)                  
    # Random.seed!(8000000)
    # Random.seed!(6789200)
    for measure_ind in 1 : num_measurements
        println("############################################################################")
        println("#########   PERFORMING MEASUREMENTS LOOP #$measure_ind                      ")
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
                tmp_kick_gate = build_kick_gates(1, 2 * tmp_num_gates + 1, s); # @show 2 * tmp_num_gates + 1
                ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                normalize!(ψ_copy)
                
                println("Applying the kicked Ising gate at time $(ind)!")
                compute_overlap(ψ, ψ_copy)
            end
            
            tmp_two_site_gates = ITensor[]
            tmp_two_site_gates = time_evolution_corner(tmp_num_gates, tmp_parity, s)
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

            # if measure_ind == 1 && ind % 2 == 0
            #     Sx[Int(ind / 2), :] = expect(ψ_copy, "Sx"; sites = 1 : N)
            #     Sy[Int(ind / 2), :] = expect(ψ_copy, "Sy"; sites = 1 : N) 
            #     Sz[Int(ind / 2), :] = expect(ψ_copy, "Sz"; sites = 1 : N); @show Sz[Int(ind / 2), :]
            # end
            # if ind % 2 == 0
            #     push!(Sx, expect(ψ_copy, "Sx"; sites = 1 : N))
            #     push!(Sy, expect(ψ_copy, "Sy"; sites = 1 : N))
            #     push!(Sz, expect(ψ_copy, "Sz"; sites = 1 : N))
            # end
        end

        # compute_overlap(ψ, ψ_copy)
        if measure_ind - 1 < 1E-8
            # Measure Sx on each site
            tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N)
            Sx[1:2] = tmp_Sx[1:2]

            # Measure Sy on each site
            tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N)
            Sy[1:2] = tmp_Sy[1:2]

            # Measure Sz on each site
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
            Sz[1:2] = tmp_Sz[1:2]
            # Sz_Reset[1, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
        end
        
        # compute_overlap(ψ, ψ_copy)
        Sz_sample[measure_ind, 1:2] = sample(ψ_copy, 1)
        site_tensor_index = (site_tensor_index + 1) % div(N, 2)
        if site_tensor_index < 1E-8
            site_tensor_index = div(N, 2)
        end

        # Running the diagonal part of the circuit 
        if N_diadonal > 1E-8
            @time for ind₁ in 1 : N_diagonal
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
                @show gate_seeds, ind₁
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
                end
    
                
                ## Make local measurements using the wavefunction 
                tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N)
                tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N)
                tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
    
                tmp_measure_index = (2 * ind₁ + 1) % N
                Sx[2 * ind₁ + 1 : 2 * ind₁ + 2] = tmp_Sx[tmp_measure_index : tmp_measure_index + 1] 
                Sy[2 * ind₁ + 1 : 2 * ind₁ + 2] = tmp_Sy[tmp_measure_index : tmp_measure_index + 1]
                Sz[2 * ind₁ + 1 : 2 * ind₁ + 2] = tmp_Sz[tmp_measure_index : tmp_measure_index + 1]
                
    
                index_to_sample = (2 * ind₁ + 1) % N
                println("############################################################################")
                @show tmp_Sz[index_to_sample : index_to_sample + 1]
                println("****************************************************************************")
    
                # println("############################################################################")
                # tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
                # @show index_to_sample
                # @show tmp_Sz[index_to_sample : index_to_sample + 1]
                # println("****************************************************************************")
                Sz_sample[measure_ind, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, index_to_sample)
                site_tensor_index = (site_tensor_index + 1) % div(N, 2)
                if site_tensor_index < 1E-8
                    site_tensor_index = div(N, 2)
                end
                # if measure_ind - 1 < 1E-8 
                #     Sz_Reset[ind₁ + 1, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
                # end
                # println("")
                # println("")
                # println("Yeah!")
                # @show Sz_Reset[1, :]
                # println("")
                # println("")
                # Sz_sample[measure_ind, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, 2 * ind₁ + 1)
            end
        end
        # #**************************************************************************************************************************************
        # # Code up the right corner for the specific case without diagonal part. 
        # # Generalize the code later 
        # @time for ind in 1 : circuit_time
        #     tmp_gates_number = div(ind, 2)
        #     if ind % 2 == 1
        #         tmp_kick_gates = kick_gates_right_corner(N, tmp_gates_number, N, s)
        #         ψ_copy = apply(tmp_kick_gates, ψ_copy; cutoff)
        #         normalize!(ψ_copy)

        #         println("Applying transverse Ising fields at time slice $(ind)")
        #         compute_overlap(ψ, ψ_copy)
        #     end

        #     if ind % 2 == 1
        #         tmp_edge = N - 1
        #     else
        #         tmp_edge = N
        #     end
            

        #     if tmp_gates_number > 1E-8
        #         println("Applying longitudinal Ising fields and Ising interaction at time slice $(ind)")
        #         println("")
        #         tmp_two_site_gates = layers_right_corner(tmp_edge, tmp_gates_number, s)
        #         ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
        #         normalize!(ψ_copy)
        #         compute_overlap(ψ, ψ_copy)
        #     end
        # end
        # #**************************************************************************************************************************************
        
        starting_tensor = (site_tensor_index - 1) % div(N, 2)
        if starting_tensor < 1E-8
            starting_tensor = div(N, 2)
        end
        starting_site = 2 * starting_tensor

        @time for ind in 1 : circuit_time
            tmp_gates_number = div(ind, 2)
            # @show ind, tmp_gates_number
            
            if ind % 2 == 1
                println("")
                println("")
                println("Applying transverse Ising fields at time slice $(ind)")
                println("")
                println("")
                @show expect(ψ_copy, "Sx"; sites = 1 : N)
                @show expect(ψ_copy, "Sy"; sites = 1 : N)
                @show expect(ψ_copy, "Sz"; sites = 1 : N)

                tmp_kick_gates = kick_gates_right_corner(starting_site, tmp_gates_number, N, s)
                ψ_copy = apply(tmp_kick_gates, ψ_copy; cutoff)
                normalize!(ψ_copy)

                compute_overlap(ψ, ψ_copy)
                @show expect(ψ_copy, "Sx"; sites = 1 : N)
                @show expect(ψ_copy, "Sy"; sites = 1 : N)
                @show expect(ψ_copy, "Sz"; sites = 1 : N)
            end

            # Set up the starting index for a sequence of two-site gates
            if ind % 2 == 1
                tmp_edge = starting_site - 1
            else
                tmp_edge = starting_site
            end

            if tmp_gates_number > 1E-8
                println("Applying longitudinal Ising fields and Ising interaction at time slice $(ind)")
                println("")
                tmp_two_site_gates = layers_right_corner(tmp_edge, starting_site, tmp_gates_number, N, s)
                for temporary_gate in tmp_two_site_gates
                    # @show temporary_gate
                    ψ_copy = apply(temporary_gate, ψ_copy; cutoff)
                    normalize!(ψ_copy)
                end
                # ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
                # normalize!(ψ_copy)
                compute_overlap(ψ, ψ_copy)
            end
            @show expect(ψ_copy, "Sx"; sites = 1 : N)
            @show expect(ψ_copy, "Sy"; sites = 1 : N)
            @show expect(ψ_copy, "Sz"; sites = 1 : N)
        end

        # measurement_starting_site = (starting_site + 3) % N
        # measurement_interval = (N - measurement_starting_site + 1) % N
        # if measurement_starting_site - 1 < 1E-8
        #     measurement_interval = N - 2
        # end

        if measure_ind - 1 < 1E-8
            tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N)
            tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N)
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
 
            if abs(site_tensor_index - div(N, 2)) < 1E-8
                println("")
                println("")
                println("")
                println("")
                println("$(site_tensor_index)")
                println("")
                println("")
                println("")
                println("")
                Sx[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sx[1 : N - 2]
                Sy[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sy[1 : N - 2]
                Sz[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sz[1 : N - 2]
            elseif abs(site_tensor_index - 1) < 1E-8
                Sx[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sx[3 : N]
                Sy[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sy[3 : N]
                Sz[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sz[3 : N]
            else
                interval = 2 * (div(N, 2) - site_tensor_index)
                @show site_tensor_index, interval
                Sx[2 * (N_diagonal + 1) + 1 : 2 * (N_diagonal + 1) + interval] = tmp_Sx[2 * site_tensor_index + 1 : N]
                Sy[2 * (N_diagonal + 1) + 1 : 2 * (N_diagonal + 1) + interval] = tmp_Sy[2 * site_tensor_index + 1 : N]
                Sz[2 * (N_diagonal + 1) + 1 : 2 * (N_diagonal + 1) + interval] = tmp_Sz[2 * site_tensor_index + 1 : N]

                Sx[2 * (N_diagonal + 1) + interval + 1 : N_total] = tmp_Sx[1 : starting_site]
                Sy[2 * (N_diagonal + 1) + interval + 1 : N_total] = tmp_Sy[1 : starting_site]
                Sz[2 * (N_diagonal + 1) + interval + 1 : N_total] = tmp_Sz[1 : starting_site]
            end
            @show tmp_Sx, tmp_Sy, tmp_Sz
        end

        # Create a vector of sites that need to be measured in the right lightcone        
        # sites_to_measure = Vector{Int}
        sites_to_measure = []
        for ind in 1 : Int(floquet_time)
            tmp_site = (starting_site + 2 * ind + 1) % N
            push!(sites_to_measure, tmp_site)
        end
        @show sites_to_measure

        sample_index = 0
        for ind in sites_to_measure
            Sz_sample[measure_ind, 2 * (N_diagonal + 1) + 2 * sample_index + 1 : 2 * (N_diagonal + 1) + 2 * sample_index + 2] = sample(ψ_copy, ind)
            normalize!(ψ_copy)
            sample_index += 1
        end

        # Sx[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sx"; sites = 1 : N);
        # Sy[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sy"; sites = 1 : N); 
        # Sz[Int(floquet_time) + 1, :] = expect(ψ_copy, "Sz"; sites = 1 : N); # @show real(Sz[4, :]) 
    end
    replace!(Sz_sample, 1.0 => 0.5, 2.0 => -0.5)
     

    println("################################################################################")
    println("################################################################################")
    println("Projection in the Sz basis of the initial MPS")
    @show Sz₀
    println("################################################################################")
    println("################################################################################")
    
    # Store data in hdf5 file
    file = h5open("Data_Benchmark/holoQUADS_Circuit_Finite_N$(N_total)_T$(floquet_time)_AFM.h5", "w")
    write(file, "Initial Sz", Sz₀)
    write(file, "Sx", Sx)
    write(file, "Sy", Sy)
    write(file, "Sz", Sz)
    # write(file, "Cxx", Cxx)
    # write(file, "Cyy", Cyy)
    # write(file, "Czz", Czz)
    write(file, "Sz_sample", Sz_sample)
    # write(file, "Sz_Reset", Sz_Reset)
    # write(file, "Wavefunction Overlap", ψ_overlap)
    close(file)

    return
end  