# 05/02/2023
# Implement the holoQUADS circuit for the SDKI model
# Skip the resetting part and avoid using a long-range two-site gate

using ITensors
using ITensorMPS
using HDF5
using Random
using TimerOutputs
# using MKL
using LinearAlgebra
BLAS.set_num_threads(8)


include("Sample.jl")
include("Entanglement.jl")
include("ObtainBond.jl")
include("holoQUADS_Time_Evolution_Gates.jl")
include("dmrs_random.jl")


const time_machine = TimerOutput()
ITensors.disable_warn_order()


let
    floquet_time = 3
    circuit_time = 2 * Int(floquet_time)
    cutoff = 1e-8
    tau = 1.0
    h = 0.2                                            # an integrability-breaking longitudinal field h 
    number_of_samples = 100
    measure_string = "Sx"
    sample_index = 0

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    N_corner = 2 * Int(floquet_time) + 2
    N_total = 32
    N_diagonal = div(N_total - N_corner, 2)     # the number of diagonal parts of the holoQUADS circuit
    s = siteinds("S=1/2", N_total; conserve_qns = false)
    # @show typeof(s) 
    # @show N_diagonal, N_corner, N_total

    # Allocation for observables
    @timeit time_machine "Allocation" begin
        Sx = zeros(ComplexF64, N_total)
        Sy = zeros(ComplexF64, N_total)
        Sz = zeros(ComplexF64, N_total)
        samples = zeros(Float64, number_of_samples, N_total)
        samples_bitstring = zeros(Float64, number_of_samples, N_total)
        Sx_rdm = zeros(Float64, number_of_samples, N_total)
        Sz_rdm = zeros(Float64, number_of_samples, N_total)
        SvN = zeros(Float64, number_of_samples, N_total * (N_total - 1))
        Bond = zeros(Float64, number_of_samples, N_total * (N_total - 1))
    end
    

    # Initialize the wavefunction as a random MPS or a product state
    states = [isodd(n) ? "Up" : "Dn" for n = 1:N_total]
    Random.seed!(123)
    ψ = randomMPS(s, states, linkdims = 2)
    # ψ = MPS(s, states)
    Sx₀ = expect(ψ, "Sx"; sites = 1:N_total)
    Sz₀ = expect(ψ, "Sz"; sites = 1:N_total)
    @show Sx₀
    @show Sz₀ 


    for measure_index = 1:number_of_samples
        println("")
        println("")
        println("############################################################################")
        println("#########   PERFORMING MEASUREMENTS LOOP #$measure_index                    ")
        println("############################################################################")
        println("")
        println("")

        # Make a copy of the original wavefunction for each sample
        ψ_copy = deepcopy(ψ)
        ψ_density = deepcopy(ψ)
        tensor_pointer = 1

        @timeit time_machine "LLC Evolution" for tmp_ind = 1:circuit_time
            # Apply a sequence of two-site gates
            tmp_parity = (tmp_ind - 1) % 2
            tmp_number_of_gates = Int(floquet_time) - floor(Int, (tmp_ind - 1) / 2)

            # APPLY ONE-SITE GATES
            if tmp_ind % 2 == 1
                tmp_kick_gate = build_kick_gates(1, 2 * tmp_number_of_gates + 1, s)
                ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                normalize!(ψ_copy)

                ψ_density = apply(tmp_kick_gate, ψ_density; cutoff)
                normalize!(ψ_density)
            end

            # APPLY TWO-SITE GATES
            tmp_two_site_gates = left_light_cone(tmp_number_of_gates, tmp_parity, h, tau, s)
            ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)

            ψ_density = apply(tmp_two_site_gates, ψ_density; cutoff)
            normalize!(ψ_density)
        end

        # Measure and timing the first two sites after applying the left light cone
        @timeit time_machine "Measure LLC unit cell" begin
            if measure_index == 1
                # Measure Sx, Sy, and Sz on each site
                Sx[1:2] = expect(ψ_copy, "Sx"; sites = 1:2)
                Sy[1:2] = expect(ψ_copy, "Sy"; sites = 1:2)
                Sz[1:2] = expect(ψ_copy, "Sz"; sites = 1:2)
                @show Sx[1:2], Sy[1:2], Sz[1:2]
            end

            # Measure von Neumann entanglement entropy before and after measurements 
            SvN[
                measure_index,
                (2*tensor_pointer-2)*(N_total-1)+1:(2*tensor_pointer-1)*(N_total-1),
            ] = entanglement_entropy(ψ_copy, N_total)
            Bond[
                measure_index,
                (2*tensor_pointer-2)*(N_total-1)+1:(2*tensor_pointer-1)*(N_total-1),
            ] = obtain_bond_dimension(ψ_copy, N_total)

            # Take measurements of a two-site unit cell
            samples[measure_index, 2*tensor_pointer-1:2*tensor_pointer] = 
                expect(ψ_copy, measure_string; sites = 2*tensor_pointer-1:2*tensor_pointer)
            samples_bitstring[measure_index, 2*tensor_pointer-1:2*tensor_pointer]=
                revised_sample(ψ_copy, 2 * tensor_pointer - 1, measure_string)
            normalize!(ψ_copy)
            # @show samples[measure_index, 2*tensor_pointer-1:2*tensor_pointer]

            @show ψ_density
            Sx_rdm[measure_index, 2*tensor_pointer-1:2*tensor_pointer], 
            Sz_rdm[measure_index, 2*tensor_pointer-1:2*tensor_pointer] = 
                sample_density_matrix(ψ_density, 2 * tensor_pointer - 1, N_total)
            normalize!(ψ_density)
            println("After sampling the first two sites")
            @show ψ_density
            # @show Sx_rdm[measure_index, 2*tensor_pointer-1:2*tensor_pointer]

            SvN[
                measure_index,
                (2*tensor_pointer-1)*(N_total-1)+1:2*tensor_pointer*(N_total-1),
            ] = entanglement_entropy(ψ_copy, N_total)
            Bond[
                measure_index,
                (2*tensor_pointer-1)*(N_total-1)+1:2*tensor_pointer*(N_total-1),
            ] = obtain_bond_dimension(ψ_copy, N_total)
        end

        @show ψ_density
        # Running the diagonal part of the circuit 
        if N_diagonal > 1E-8
            for ind₁ = 1:N_diagonal
                tensor_pointer += 1

                gate_seeds = []
                for ind₂ = 1:circuit_time
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

                @timeit time_machine "DC Evoltuion" for ind₃ = 1:circuit_time                    # Apply the kick gates at integer time
                    if mod(ind₃, 2) == 1
                        tmp_kick_gate = build_kick_gates(gate_seeds[ind₃] - 1, gate_seeds[ind₃], s)
                        ψ_copy = apply(tmp_kick_gate, ψ_copy; cutoff)
                        normalize!(ψ_copy)
                        
                        ψ_density = apply(tmp_kick_gate, ψ_density; cutoff)
                        normalize!(ψ_density)
                    end

                    # Apply the Ising interaction and longitudinal fields using a sequence of two-site gates
                    # tmp_two_site_gates = diagonal_circuit(gate_seeds[ind₃], h, tau, s)
                    tmp_two_site_gate = diagonal_right_edge(gate_seeds[ind₃], N_total, h, tau, s)
                    ψ_copy = apply(tmp_two_site_gate, ψ_copy; cutoff)
                    normalize!(ψ_copy)

                    ψ_density = apply(tmp_two_site_gate, ψ_density; cutoff)
                    normalize!(ψ_density)
                end
                                

                @timeit time_machine "Measure DC unit cell" begin
                    if measure_index == 1
                        Sx[2*tensor_pointer-1:2*tensor_pointer] = 
                            expect(ψ_copy, "Sx"; sites = 2*tensor_pointer-1:2*tensor_pointer)
                        Sy[2*tensor_pointer-1:2*tensor_pointer] =
                            expect(ψ_copy, "Sy"; sites = 2*tensor_pointer-1:2*tensor_pointer)
                        Sz[2*tensor_pointer-1:2*tensor_pointer] =
                            expect(ψ_copy, "Sz"; sites = 2*tensor_pointer-1:2*tensor_pointer)
                    end
    
                    # Compute von Neumann entanglement entropy before taking measurements
                    SvN[
                        measure_index,
                        (2*tensor_pointer-2)*(N_total-1)+1:(2*tensor_pointer-1)*(N_total-1),
                    ] = entanglement_entropy(ψ_copy, N_total)
                    Bond[
                        measure_index,
                        (2*tensor_pointer-2)*(N_total-1)+1:(2*tensor_pointer-1)*(N_total-1),
                    ] = obtain_bond_dimension(ψ_copy, N_total)
                    # @show Bond[
                    #     measure_index,
                    #     (2*tensor_pointer-2)*(N_total-1)+1:(2*tensor_pointer-1)*(N_total-1),
                    # ]

                    # Taking measurements of one two-site unit cell in the diagonal part of a circuit
                    samples[measure_index, 2*tensor_pointer-1:2*tensor_pointer] =
                        expect(ψ_copy, measure_string; sites = 2*tensor_pointer-1:2*tensor_pointer)
                    samples_bitstring[measure_index, 2*tensor_pointer-1:2*tensor_pointer] = 
                        revised_sample(ψ_copy, 2 * tensor_pointer - 1, measure_string)
                    normalize!(ψ_copy)
                    # @show samples[measure_index, 2*tensor_pointer-1:2*tensor_pointer]

                    println("")
                    println("")
                    println("Right before sampling the two-site unit cell")
                    @show ψ_density
                    Sx_rdm[measure_index, 2*tensor_pointer-1:2*tensor_pointer],
                    Sz_rdm[measure_index, 2*tensor_pointer-1:2*tensor_pointer] = 
                    sample_rdm_bulk(ψ_density, 2 * tensor_pointer - 1, N_total)
                    normalize!(ψ_density)
                    # @show ψ_density
                    # @show Sx_rdm[measure_index, 2*tensor_pointer-1:2*tensor_pointer]


                    # Compute von Neumann entanglement entropy after taking measurements
                    SvN[measure_index, (2*tensor_pointer-1)*(N_total-1)+1:2*tensor_pointer*(N_total-1)] = entanglement_entropy(ψ_copy, N_total)
                    Bond[measure_index, (2*tensor_pointer-1)*(N_total-1)+1:2*tensor_pointer*(N_total-1)] = obtain_bond_dimension(ψ_copy, N_total)
                    # @show Bond[measure_index, (2*tensor_pointer-1)*(N_total-1)+1:2*tensor_pointer*(N_total-1)] 
                end
            end
        end

        # 06/11/2023
        # Reconstruct the right light cone to simulate a chain with finite length.
        for ind = 1:div(N_corner - 2, 2)
            tensor_pointer += 1
            left_ptr = 2 * tensor_pointer - 1
            right_ptr = 2 * tensor_pointer
            # @show ind, left_ptr, right_ptr

            @timeit time_machine "RLC Evolution" for time_index = 1:circuit_time-2*(ind-1)
                if time_index == 1
                    ending_index = N_total
                    starting_index = N_total
                else
                    ending_index = N_total - time_index + 2
                    starting_index = ending_index - 1
                end

                # @show time_index, starting_index, ending_index
                # Applying a sequence of one-site gates
                if time_index % 2 == 1
                    # @show time_index, starting_index, ending_index
                    tmp_kick_gates = build_kick_gates(starting_index, ending_index, s)
                    ψ_copy = apply(tmp_kick_gates, ψ_copy; cutoff)
                    normalize!(ψ_copy)

                    ψ_density = apply(tmp_kick_gates, ψ_density; cutoff)
                    normalize!(ψ_density)
                end

                if time_index - 1 > 1E-8
                    tmp_two_site_gate = diagonal_right_edge(ending_index, N_total, h, tau, s)
                    ψ_copy = apply(tmp_two_site_gate, ψ_copy; cutoff)
                    normalize!(ψ_copy)

                    ψ_density = apply(tmp_two_site_gate, ψ_density; cutoff)
                    normalize!(ψ_density)
                end
            end

            # Measure local observables directly from the wavefunction
            @timeit time_machine "Measure RLC unit cell" begin
                if measure_index == 1
                    Sx[left_ptr:right_ptr] = expect(ψ_copy, "Sx"; sites = left_ptr:right_ptr)
                    Sy[left_ptr:right_ptr] = expect(ψ_copy, "Sy"; sites = left_ptr:right_ptr)
                    Sz[left_ptr:right_ptr] = expect(ψ_copy, "Sz"; sites = left_ptr:right_ptr)
                end

                # Compute von Neumann entanglement entropy before taking measurements
                SvN[measure_index, (left_ptr-1)*(N_total-1)+1:left_ptr*(N_total-1)] =
                entanglement_entropy(ψ_copy, N_total)
                Bond[measure_index, (left_ptr-1)*(N_total-1)+1:left_ptr*(N_total-1)] =
                    obtain_bond_dimension(ψ_copy, N_total)
                
                # Taking measurements of one two-site unit cell in the right light cone
                samples[measure_index, left_ptr:right_ptr] =
                    expect(ψ_copy, measure_string; sites = left_ptr:right_ptr)
                samples_bitstring[measure_index, left_ptr:right_ptr] = 
                    revised_sample(ψ_copy, left_ptr, measure_string)
                normalize!(ψ_copy)

                Sx_rdm[measure_index, left_ptr:right_ptr],
                Sz_rdm[measure_index, left_ptr:right_ptr] = sample_rdm_bulk(ψ_density, left_ptr, N_total)
                normalize!(ψ_density)

                # Compute von Neumann entanglement entropy after taking measurements
                SvN[measure_index, (right_ptr-1)*(N_total-1)+1:right_ptr*(N_total-1)] =
                    entanglement_entropy(ψ_copy, N_total)
                Bond[measure_index, (right_ptr-1)*(N_total-1)+1:right_ptr*(N_total-1)] =
                    obtain_bond_dimension(ψ_copy, N_total) 
            end
        end
        # @show Bond[measure_index, :]
    end

    replace!(samples_bitstring, 1.0 => 0.5, 2.0 => -0.5)
    @show time_machine
    

    # STORE DATA IN A HDF5 FILE 
    h5open("../../data/rdm_sample/ELCS_N$(N_total)_h$(h)_t$(floquet_time)_sample$(number_of_samples).h5", "w") do file
        write(file, "Sz0", Sz₀)
        write(file, "Sx", Sx)
        write(file, "Sy", Sy)
        write(file, "Sz", Sz)
        write(file, "Entropy", SvN)
        write(file, "Bond Dimension", Bond)
        write(file, "ES", samples)
        write(file, "Bitstring", samples_bitstring)
        write(file, "Sx_rdm", Sx_rdm)
        write(file, "Sz_rdm", Sz_rdm)
    end

    return
end