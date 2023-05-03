## 05/02/2023
## IMPLEMENT THE HOLOQAUDS CIRCUITS WITHOUT RECYCLING AND LONG-RANGE GATES.

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product
using Random
ITensors.disable_warn_order()




# Sample a two-site MPS to compute Sx, Sy or Sz
function sample(m :: MPS, j :: Int, observable_type :: AbstractString)
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

    # Define projectors in the Sz basis
    Sx_projn = 1/sqrt(2) * [[1, 1], [1, -1]]
    Sy_projn = 1/sqrt(2) * [[1, 1.0im], [1, -1.0im]]
    Sz_projn = [[1, 0], [0, 1]]
    
    if observable_type == "Sx"
        tmp_projn = Sx_projn
    elseif observable_type == "Sy"
        tmp_projn = Sy_projn
    elseif observable_type == "Sz"
        tmp_projn = Sz_projn
    else
        error("sample: Measurement type doesn't exist")
    end
    

    # Sample the target observables
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
            projn[tmpS => 1] = tmp_projn[n][1]
            projn[tmpS => 2] = tmp_projn[n][2]
        
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind - j + 1] = n

        if ind < mpsLength
            A = m[ind + 1] * An
            A *= (1. / sqrt(pn))
        end
    end
    return result
end 

# Contruct layers of two-site gates including the Ising interaction and longitudinal fileds in the left light cone.
function left_light_cone(number_of_gates :: Int, parity :: Int, longitudinal_field :: Float64, Δτ :: Float64, tmp_sites)
    gates = ITensor[]

    for ind in 1 : number_of_gates
        tmp_index = 2 * ind - parity
        s1 = tmp_sites[tmp_index]
        s2 = tmp_sites[tmp_index + 1]
       
        if tmp_index - 1 < 1E-8
            coeff₁ = 2
            coeff₂ = 1
        else
            coeff₁ = 1
            coeff₂ = 1
        end

        # hj = coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        # hj = π * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        hj = π/2 * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * Δτ * hj)
        push!(gates, Gj)
    end
    return gates
end

# Construct multiple two-site gate(s) to apply the Ising interaction and the longitudinal fields in the diagonal parts of the circuit
function time_evolution(starting_index :: Int, longitudinal_field :: Float64, Δτ :: Float64, tmp_sites)
    gates = ITensor[]

    # Generate a local two-site gate 
    s1 = tmp_sites[starting_index]
    s2 = tmp_sites[starting_index - 1]

    # hj = longitudinal_field * op("Sz", s1) * op("Id", s2) + longitudinal_field * op("Id", s1) * op("Sz", s2)
    # hj = π * op("Sz", s1) * op("Sz", s2) + longitudinal_field * op("Sz", s1) * op("Id", s2) + longitudinal_field * op("Id", s1) * op("Sz", s2)
    hj = π/2 * op("Sz", s1) * op("Sz", s2) + longitudinal_field * op("Sz", s1) * op("Id", s2) + longitudinal_field * op("Id", s1) * op("Sz", s2)
    Gj = exp(-1.0im * Δτ * hj)                 
    push!(gates, Gj)

    return gates
end


# Construct layers of two-site gate including the Ising interaction and longitudinal fields in the right light cone
function right_light_cone(starting_index :: Int, number_of_gates :: Int, edge_index :: Int, longitudinal_field :: Float64, Δτ :: Float64, tmp_sites)
    gates = ITensor[]

    for ind in 1 : number_of_gates
        tmp_start = starting_index - 2 * (ind - 1)
        tmp_end = tmp_start - 1

        s1 = tmp_sites[tmp_end]
        s2 = tmp_sites[tmp_start]

        # Consider the finite-size effect on the right edge
        if abs(tmp_start - edge_index) < 1E-8
            coeff₁ = 1
            coeff₂ = 2
        else
            coeff₁ = 1
            coeff₂ = 1
        end

        # hj = coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        # hj = π * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        
        hj = π/2 * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * Δτ * hj)
        push!(gates, Gj)
    end
    return gates
end


# Construct multiple one-site gates to apply the transverse Ising fields.
function build_kick_gates(starting_index :: Int, ending_index :: Int, tmp_sites)
    kick_gate = ITensor[]
    for ind in starting_index : ending_index
        s1 = tmp_sites[ind] 

        hamilt = π / 2 * op("Sx", s1)
        tmpG = exp(-1.0im * hamilt)
        push!(kick_gate, tmpG)
    end
    return kick_gate
end

# Assemble the holoQUADS circuits 
let 
    floquet_time = 4.0                                                                  
    circuit_time = 2 * Int(floquet_time)
    cutoff = 1E-8
    tau = 1.0
    h = 0.2                                                              # an integrability-breaking longitudinal field h 
    number_of_samples = 1

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    N_corner = 2 * Int(floquet_time) + 2       
    N_diagonal = 20                                                             # the number of diagonal parts of circuit
    N_total = N_corner + 2 * N_diagonal
    s = siteinds("S=1/2", N_total; conserve_qns = false)
    

    # entropy = complex(zeros(2, N - 1))
    Sx = complex(zeros(N_total))
    Sy = complex(zeros(N_total))
    Sz = complex(zeros(N_total))
    samples = real(zeros(number_of_samples, N_total))


    # Initialize the wavefunction
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N_total]
    ψ = MPS(s, states)
    Sz₀ = expect(ψ, "Sz"; sites = 1 : N_total)
    Random.seed!(123)
    
    # # Initializa a random MPS
    # # initialization_s = siteinds("S=1/2", N; conserve_qns = false)
    # initialization_states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # Random.seed!(87900) 
    # ψ = randomMPS(s, initialization_states, linkdims = 2)
    # # ψ = initialization_ψ[1 : N]
    # Sz₀ = expect(ψ, "Sz"; sites = 1 : N)
    # # @show maxlinkdim(ψ)


    for measure_index in 1 : number_of_samples
        println("")
        println("")
        println("############################################################################")
        println("#########   PERFORMING MEASUREMENTS LOOP #$measure_index                    ")
        println("############################################################################")
        println("")
        println("")

        # Make a copy of the original wavefunction for each sample
        ψ_copy = deepcopy(ψ)
        tensor_pointer = 1
        
        @time for tmp_ind in 1 : circuit_time
            # Apply a sequence of two-site gates
            tmp_parity = (tmp_ind - 1) % 2
            tmp_number_of_gates = Int(floquet_time) - floor(Int, (tmp_ind - 1) / 2) 

            # APPLY ONE-SITE GATES
            if tmp_ind % 2 == 1
                tmp_kick_gate = build_kick_gates(1, 2 * tmp_number_of_gates + 1, s)
                ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                normalize!(ψ_copy)
            end
            
            # APPLY TWO-SITE GATES
            tmp_two_site_gates = left_light_cone(tmp_number_of_gates, tmp_parity, h, tau, s)
            ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
        end

        
        if measure_index - 1 < 1E-8
            # Measure Sx on each site
            tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N_total)
            Sx = tmp_Sx

            # Measure Sy on each site
            tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N_total)
            Sy = tmp_Sy
            
            # Measure Sz on each site
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N_total)
            Sz = tmp_Sz


            # for site_index in 1 : N - 1 
            #     orthogonalize!(ψ_copy, site_index)
            #     if abs(site_index - 1) < 1E-8
            #         i₁ = siteind(ψ_copy, site_index)
            #         _, C1, _ = svd(ψ_copy[site_index], i₁)
            #     else
            #         i₁, j₁ = siteind(ψ_copy, site_index), linkind(ψ_copy, site_index - 1)
            #         _, C1, _ = svd(ψ_copy[site_index], i₁, j₁)
            #     end
            #     C1 = matrix(C1)
            #     SvN₁ = compute_entropy(C1)
               
            #     @show site_index, SvN₁
            #     entropy[1, site_index] = SvN₁
            # end
        end
       
        samples[measure_index, 2 * tensor_pointer - 1 : 2 * tensor_pointer] = sample(ψ_copy, 1, "Sz")
        normalize!(ψ_copy)
    #     site_tensor_index += 1 

    #     if measure_index - 1 < 1E-8
    #         for site_index in 2 : N - 1 
    #             orthogonalize!(ψ_copy, site_index)
    #             if abs(site_index - 1) < 1E-8
    #                 i₁ = siteind(ψ_copy[site_index], i₁)
    #                 _, C, _ = svd(ψ_copy[site_index], i₁)
    #             else
    #                 i₁, j₁ = siteind(ψ_copy, site_index), linkind(ψ_copy, site_index - 1)
    #                 _, C1, _ = svd(ψ_copy[site_index], i₁, j₁)
    #             end
    #             C1 = matrix(C1)
    #             SvN₁ = compute_entropy(C1)
                
    #             @show site_index, SvN₁
    #             entropy[2, site_index] = SvN₁
    #         end
    #     end

        # Running the diagonal part of the circuit 
        if N_diagonal > 1E-8
            @time for ind₁ in 1 : N_diagonal
                tensor_pointer += 1

                gate_seeds = []
                for ind₂ in 1 : circuit_time
                    tmp_index = N_corner + 2 * ind₁ - ind₂
                    push!(gate_seeds, tmp_index)
                end
                # println("")
                # println("")
                # println("#########################################################################################")
                # @show gate_seeds, ind₁
                # println("#########################################################################################")
                # println("")
                # println("")

                for ind₃ in 1 : circuit_time
                    # Apply the kicked gate at integer time
                    if ind₃ % 2 == 1
                        tmp_kick_gate = build_kick_gates(gate_seeds[ind₃] - 1, gate_seeds[ind₃], s)
                        ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                        normalize!(ψ_copy)
                    end

                    # Apply the Ising interaction and longitudinal fields using a sequence of two-site gates
                    tmp_two_site_gates = time_evolution(gate_seeds[ind₃], h, tau, s)
                    ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
                    normalize!(ψ_copy)
                end

                
                ## Measuring local observables directly from the wavefunction
                tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N_total)
                tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N_total)
                tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N_total)

                Sx[2 * tensor_pointer - 1 : 2 * tensor_pointer] = tmp_Sx[2 * tensor_pointer - 1 : 2 * tensor_pointer] 
                Sy[2 * tensor_pointer - 1 : 2 * tensor_pointer] = tmp_Sy[2 * tensor_pointer - 1 : 2 * tensor_pointer]
                Sz[2 * tensor_pointer - 1 : 2 * tensor_pointer] = tmp_Sz[2 * tensor_pointer - 1 : 2 * tensor_pointer]
                samples[measure_index, 2 * tensor_pointer - 1 : 2 * tensor_pointer] = sample(ψ_copy, 2 * tensor_pointer - 1, "Sz")
            end
        end

        
        # Set up and apply the right light cone
        @time for ind in 1 : circuit_time
            tmp_gates_number = div(ind, 2)

            # Apply a sequence of one-site gates
            ending_index = N_total
            starting_index = N_total - ind + 1
            if ind % 2 == 1
                tmp_kick_gates = build_kick_gates(starting_index, ending_index, s)
                ψ_copy = apply(tmp_kick_gates, ψ_copy; cutoff)
                normalize!(ψ_copy)
            end

            # Apply a sequence of two-site gates
            if ind % 2 == 1
                tmp_edge = ending_index - 1
            else
                tmp_edge = ending_index
            end
            # println("")
            # println("")
            # @show tmp_edge, starting_site
            # println("")
            # println("")

            if tmp_gates_number > 1E-8
                tmp_two_site_gates = right_light_cone(tmp_edge, tmp_gates_number, N_total, h, tau, s)
                ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
                normalize!(ψ_copy)
            end
        end

        # Create a vector of sites that need to be measured in the right lightcone 
        sites_to_measure = []
        for ind in 1 : Int(floquet_time)
            tmp_site = 2 * tensor_pointer + 2 * ind - 1
            push!(sites_to_measure, tmp_site)
        end
        @show sites_to_measure

        # tensor_pointer += 1
        if measure_index - 1 < 1E-8
            tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N_total)
            tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N_total)
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N_total)

            Sx[2 * tensor_pointer + 1 : N_total] = tmp_Sx[2 * tensor_pointer + 1 : N_total]
            Sy[2 * tensor_pointer + 1 : N_total] = tmp_Sy[2 * tensor_pointer + 1 : N_total]
            Sz[2 * tensor_pointer + 1 : N_total] = tmp_Sz[2 * tensor_pointer + 1 : N_total]
        end

        for ind in sites_to_measure
            samples[measure_index, ind : ind + 1] = sample(ψ_copy, ind, "Sz")
            normalize!(ψ_copy)
        end
    end

    replace!(samples, 1.0 => 0.5, 2.0 => -0.5)

    println("################################################################################")
    println("################################################################################")
    println("Measure Sz of the time-evolved wavefunction")
    @show Sz
    println("################################################################################")
    println("################################################################################")
    
    # @show Sz_sample
    # Store data in hdf5 file
    file = h5open("Data_Test/holoQUADS_SDKI_N$(N_total)_T$(floquet_time).h5", "w")
    write(file, "Initial Sz", Sz₀)
    write(file, "Sx", Sx)
    write(file, "Sy", Sy)
    write(file, "Sz", Sz)
    # write(file, "Samples", Samples)
    # write(file, "Entropy", entropy)
    close(file)

    return
end  