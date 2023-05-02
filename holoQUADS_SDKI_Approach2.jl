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
    
    if observable_type == "Sx":
        tmp_projn = Sx_projn
    else if observable_type == "Sy":
        tmp_projn = Sy_projn
    else if observable_type == "Sz":
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
    number_of_samples = 2000

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    N_corner = 2 * Int(floquet_time) + 2       
    N_diagonal = 11                                                             # the number of diagonal parts of circuit
    N_total = N_corner + 2 * N_diagonal
    s = siteinds("S=1/2", N_total; conserve_qns = false)
    
    # '''
    #     # Sample from the time-evolved wavefunction and store the measurements
    # '''
    # Sx_sample = real(zeros(number_of_samples, N_total))
    # Sz_sample = real(zeros(number_of_samples, N_total))
    samples = real(zeros(number_of_samples, N_total))
    # entropy = complex(zeros(2, N - 1))
    
    # '''
    #     # Measure expectation values of the wavefunction during time evolution
    # '''
    Sx = complex(zeros(N_total))
    Sy = complex(zeros(N_total))
    Sz = complex(zeros(N_total))

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
        # site_tensor_index = 0
        
        @time for tmp_ind in 1 : circuit_time
            # Apply a sequence of two-site gates
            tmp_parity = (tmp_ind - 1) % 2
            tmp_num_gates = Int(circuit_time / 2) - floor(Int, (tmp_ind - 1) / 2) 

            # APPLY ONE-SITE GATES
            if tmp_ind % 2 == 1
                tmp_kick_gate = build_kick_gates(1, 2 * tmp_num_gates + 1, s)
                ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                normalize!(ψ_copy)
            end
            
            # APPLY TWO-SITE GATES
            tmp_two_site_gates = time_evolution_corner(tmp_num_gates, tmp_parity, h, tau, s)
            ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
        end

        
    #     if measure_index - 1 < 1E-8
    #         # Measure Sx on each site
    #         tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N)
    #         Sx[1:2] = tmp_Sx[1:2]

    #         # Measure Sy on each site
    #         tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N)
    #         Sy[1:2] = tmp_Sy[1:2]

    #         # Measure Sz on each site
    #         tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
    #         Sz[1:2] = tmp_Sz[1:2]
    #         # Sz_Reset[1, :] = expect(ψ_copy, "Sz"; sites = 1 : N)

    #         for site_index in 1 : N - 1 
    #             orthogonalize!(ψ_copy, site_index)
    #             if abs(site_index - 1) < 1E-8
    #                 i₁ = siteind(ψ_copy, site_index)
    #                 _, C1, _ = svd(ψ_copy[site_index], i₁)
    #             else
    #                 i₁, j₁ = siteind(ψ_copy, site_index), linkind(ψ_copy, site_index - 1)
    #                 _, C1, _ = svd(ψ_copy[site_index], i₁, j₁)
    #             end
    #             C1 = matrix(C1)
    #             SvN₁ = compute_entropy(C1)
               
    #             @show site_index, SvN₁
    #             entropy[1, site_index] = SvN₁
    #         end
    #     end
       
    #     # Sx_sample[measure_index, 1:2] = sample(ψ_copy, 1)
    #     # Sz_sample[measure_index, 1:2] = sample(ψ_copy, 1)
    #     Samples[measure_index, 1:2] = sample(ψ_copy, 1)
    #     println("")
    #     println("")
    #     @show expect(ψ_copy, "Sz"; sites = 1 : 2)
    #     println("")
    #     println("")
    #     normalize!(ψ_copy)
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

    #     # Running the diagonal part of the circuit 
    #     if N_diagonal > 1E-8
    #         @time for ind₁ in 1 : N_diagonal
    #             gate_seeds = []
    #             for gate_ind in 1 : circuit_time
    #                 tmp_ind = (2 * ind₁ - gate_ind + N) % N
    #                 if tmp_ind == 0
    #                     tmp_ind = N
    #                 end
    #                 push!(gate_seeds, tmp_ind)
    #             end
    #             println("")
    #             println("")
    #             println("#########################################################################################")
    #             @show gate_seeds, ind₁
    #             println("#########################################################################################")
    #             println("")
    #             println("")

    #             for ind₂ in 1 : circuit_time
    #                 # Apply the kicked gate at integer time
    #                 if ind₂ % 2 == 1
    #                     tmp_kick_gate₁ = build_two_site_kick_gate(gate_seeds[ind₂], N, s)
    #                     ψ_copy = apply(tmp_kick_gate₁, ψ_copy; cutoff)
    #                     normalize!(ψ_copy)
    #                 end

    #                 # Apply the Ising interaction and longitudinal fields using a sequence of two-site gates
    #                 tmp_two_site_gates = time_evolution(gate_seeds[ind₂], N, h, tau, s)
    #                 ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
    #                 normalize!(ψ_copy)
    #             end

                
    #             ## Make local measurements using the wavefunction 
    #             tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N)
    #             tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N)
    #             tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)

    #             tmp_measure_indexex = (2 * ind₁ + 1) % N
    #             Sx[2 * ind₁ + 1 : 2 * ind₁ + 2] = tmp_Sx[tmp_measure_indexex : tmp_measure_indexex + 1] 
    #             Sy[2 * ind₁ + 1 : 2 * ind₁ + 2] = tmp_Sy[tmp_measure_indexex : tmp_measure_indexex + 1]
    #             Sz[2 * ind₁ + 1 : 2 * ind₁ + 2] = tmp_Sz[tmp_measure_indexex : tmp_measure_indexex + 1]
                

    #             index_to_sample = (2 * ind₁ + 1) % N
    #             # println("############################################################################")
    #             # @show tmp_Sz[index_to_sample : index_to_sample + 1]
    #             # println("****************************************************************************")

    #             # println("############################################################################")
    #             # tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
    #             # @show index_to_sample
    #             # @show tmp_Sz[index_to_sample : index_to_sample + 1]
    #             # println("****************************************************************************")
    #             # Sz_sample[measure_index, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, index_to_sample)
    #             Samples[measure_index, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, index_to_sample)
    #             println("")
    #             println("")
    #             # tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
    #             @show expect(ψ_copy, "Sz"; sites = (2 * ind₁ + 1) % N : (2 * ind₁ + 2) % N)
    #             println("")
    #             println("")
                
    #             site_tensor_index = (site_tensor_index + 1) % div(N, 2)
    #             if site_tensor_index < 1E-8
    #                 site_tensor_index = div(N, 2)
    #             end
    #             # if measure_index - 1 < 1E-8 
    #             #     Sz_Reset[ind₁ + 1, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
    #             # end
    #             # println("")
    #             # println("")
    #             # println("Yeah!")
    #             # @show Sz_Reset[1, :]
    #             # println("")
    #             # println("")
    #             # Sz_sample[measure_index, 2 * ind₁ + 1 : 2 * ind₁ + 2] = sample(ψ_copy, 2 * ind₁ + 1)
    #         end
    #     end

    #     # #**************************************************************************************************************************************
    #     # # Code up the right corner for the specific case without diagonal part. 
    #     # # Generalize the code later 
    #     # @time for ind in 1 : circuit_time
    #     #     tmp_gates_number = div(ind, 2)
    #     #     if ind % 2 == 1
    #     #         tmp_kick_gates = kick_gates_right_corner(N, tmp_gates_number, N, s)
    #     #         ψ_copy = apply(tmp_kick_gates, ψ_copy; cutoff)
    #     #         normalize!(ψ_copy)

    #     #         println("Applying transverse Ising fields at time slice $(ind)")
    #     #         compute_overlap(ψ, ψ_copy)
    #     #     end

    #     #     if ind % 2 == 1
    #     #         tmp_edge = N - 1
    #     #     else
    #     #         tmp_edge = N
    #     #     end
            

    #     #     if tmp_gates_number > 1E-8
    #     #         println("Applying longitudinal Ising fields and Ising interaction at time slice $(ind)")
    #     #         println("")
    #     #         tmp_two_site_gates = layers_right_corner(tmp_edge, tmp_gates_number, s)
    #     #         ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
    #     #         normalize!(ψ_copy)
    #     #         compute_overlap(ψ, ψ_copy)
    #     #     end
    #     # end
    #     # #**************************************************************************************************************************************
        
    #     starting_tensor = (site_tensor_index - 1) % div(N, 2)
    #     if starting_tensor < 1E-8
    #         starting_tensor = div(N, 2)
    #     end
    #     starting_site = 2 * starting_tensor

    #     @time for ind in 1 : circuit_time
    #         tmp_gates_number = div(ind, 2)
    #         # @show ind, tmp_gates_number
            
    #         if ind % 2 == 1
    #             println("")
    #             println("")
    #             println("Applying transverse Ising fields at time slice $(ind)")
    #             println("")
    #             println("")
    #             # @show expect(ψ_copy, "Sx"; sites = 1 : N)
    #             # @show expect(ψ_copy, "Sy"; sites = 1 : N)
    #             # @show expect(ψ_copy, "Sz"; sites = 1 : N)

    #             tmp_kick_gates = kick_gates_right_corner(starting_site, tmp_gates_number, N, s)
    #             ψ_copy = apply(tmp_kick_gates, ψ_copy; cutoff)
    #             normalize!(ψ_copy)

    #             compute_overlap(ψ, ψ_copy)
    #             # @show expect(ψ_copy, "Sx"; sites = 1 : N)
    #             # @show expect(ψ_copy, "Sy"; sites = 1 : N)
    #             # @show expect(ψ_copy, "Sz"; sites = 1 : N)
    #         end

    #         # Set up the starting index for a sequence of two-site gates
    #         if ind % 2 == 1
    #             tmp_edge = starting_site - 1
    #         else
    #             tmp_edge = starting_site
    #         end
    #         println("")
    #         println("")
    #         @show tmp_edge, starting_site
    #         println("")
    #         println("")

    #         if tmp_gates_number > 1E-8
    #             println("Applying longitudinal Ising fields and Ising interaction at time slice $(ind)")
    #             println("")
    #             println("")

    #             tmp_two_site_gates = layers_right_corner(tmp_edge, starting_site, tmp_gates_number, N, h, tau, s)
    #             for temporary_gate in tmp_two_site_gates
    #                 # @show temporary_gate
    #                 ψ_copy = apply(temporary_gate, ψ_copy; cutoff)
    #                 normalize!(ψ_copy)
    #             end
    #             # ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
    #             # normalize!(ψ_copy)
    #             compute_overlap(ψ, ψ_copy)
    #         end
    #         # @show expect(ψ_copy, "Sx"; sites = 1 : N)
    #         # @show expect(ψ_copy, "Sy"; sites = 1 : N)
    #         # @show expect(ψ_copy, "Sz"; sites = 1 : N)
    #     end

    #     if measure_index - 1 < 1E-8
    #         tmp_Sx = expect(ψ_copy, "Sx"; sites = 1 : N)
    #         tmp_Sy = expect(ψ_copy, "Sy"; sites = 1 : N)
    #         tmp_Sz = expect(ψ_copy, "Sz"; sites = 1 : N)
 
    #         if abs(site_tensor_index - div(N, 2)) < 1E-8
    #             println("")
    #             println("")
    #             println("")
    #             println("")
    #             println("Edge case site=$(site_tensor_index)")
    #             println("")
    #             println("")
    #             println("")
    #             println("")
    #             Sx[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sx[1 : N - 2]
    #             Sy[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sy[1 : N - 2]
    #             Sz[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sz[1 : N - 2]
    #         elseif abs(site_tensor_index - 1) < 1E-8
    #             println("")
    #             println("")
    #             println("")
    #             println("")
    #             println("Edge case site=$(site_tensor_index)")
    #             println("")
    #             println("")
    #             println("")
    #             println("")
    #             Sx[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sx[3 : N]
    #             Sy[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sy[3 : N]
    #             Sz[2 * (N_diagonal + 1) + 1 : N_total] = tmp_Sz[3 : N]
    #         else
    #             interval = 2 * (div(N, 2) - site_tensor_index)
    #             @show site_tensor_index, interval
    #             Sx[2 * (N_diagonal + 1) + 1 : 2 * (N_diagonal + 1) + interval] = tmp_Sx[2 * site_tensor_index + 1 : N]
    #             Sy[2 * (N_diagonal + 1) + 1 : 2 * (N_diagonal + 1) + interval] = tmp_Sy[2 * site_tensor_index + 1 : N]
    #             Sz[2 * (N_diagonal + 1) + 1 : 2 * (N_diagonal + 1) + interval] = tmp_Sz[2 * site_tensor_index + 1 : N]

    #             Sx[2 * (N_diagonal + 1) + interval + 1 : N_total] = tmp_Sx[1 : starting_site]
    #             Sy[2 * (N_diagonal + 1) + interval + 1 : N_total] = tmp_Sy[1 : starting_site]
    #             Sz[2 * (N_diagonal + 1) + interval + 1 : N_total] = tmp_Sz[1 : starting_site]
    #         end
    #         @show tmp_Sz
    #     end

    #     # Create a vector of sites that need to be measured in the right lightcone        
    #     # sites_to_measure = Vector{Int}
    #     sites_to_measure = []
    #     for ind in 1 : Int(floquet_time)
    #         tmp_site = (starting_site + 2 * ind + 1) % N
    #         push!(sites_to_measure, tmp_site)
    #     end
    #     @show sites_to_measure

    #     sample_index = 0
    #     for ind in sites_to_measure
    #         # Sz_sample[measure_index, 2 * (N_diagonal + 1) + 2 * sample_index + 1 : 2 * (N_diagonal + 1) + 2 * sample_index + 2] = sample(ψ_copy, ind)
    #         Samples[measure_index, 2 * (N_diagonal + 1) + 2 * sample_index + 1 : 2 * (N_diagonal + 1) + 2 * sample_index + 2] = sample(ψ_copy, ind)
    #         normalize!(ψ_copy)
    #         sample_index += 1
    #     end
    # end
    # # replace!(Sz_sample, 1.0 => 0.5, 2.0 => -0.5)
    # replace!(Samples, 1.0 => 0.5, 2.0 => -0.5)

    # println("################################################################################")
    # println("################################################################################")
    # println("Projection in the Sz basis of the initial MPS")
    # @show Sz₀
    # println("################################################################################")
    # println("################################################################################")
    
    # # @show Sz_sample
    # # Store data in hdf5 file
    # file = h5open("Data_Benchmark/holoQUADS_Circuit_Finite_N$(N_total)_T$(floquet_time)_AFM_Sz_Sample1.h5", "w")
    # write(file, "Initial Sz", Sz₀)
    # write(file, "Sx", Sx)
    # write(file, "Sy", Sy)
    # write(file, "Sz", Sz)
    # # write(file, "Cxx", Cxx)
    # # write(file, "Cyy", Cyy)
    # # write(file, "Czz", Czz)
    # write(file, "Samples", Samples)
    # # write(file, "Sz_Reset", Sz_Reset)
    # # write(file, "Wavefunction Overlap", ψ_overlap)
    # write(file, "Entropy", entropy)
    # close(file)

    return
end  