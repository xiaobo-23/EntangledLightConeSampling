## 02/07/2023
## Implement the holographic quantum circuit for the Heisenberg model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product, Float64
using HDF5: file
using Random
ITensors.disable_warn_order()
include("Heisenberg_src/Sample.jl")
include("Heisenberg_src/holoQUADS_Gates.jl")


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
    ##### Define parameters used in the holoQUADS circuit                                                                           #####
    ##### Given the light-cone structure of the real-time dynamics, circuit depth and number of sites are related                   #####
    #####################################################################################################################################
    floquet_time = 2.0
    tau = 0.1                                                                               # Trotter decomposition time step 
    N_time_slice = Int(floquet_time / tau) * 2
    
    unit_cell_size = N_time_slice + 2
    number_of_DC = 0
    N = unit_cell_size + 2 * number_of_DC                                                   # Number of total sites on a MPS
    
    # N_half_infinite = N + 2 * N_diagonal_circuit

    cutoff = 1E-8
    @show floquet_time, typeof(floquet_time)
    num_measurements = 1
    #####################################################################################################################################
    #####################################################################################################################################


    # Make an array of 'site' indices && quantum numbers are CONSERVED for the Heisenberg model
    # Using Neel state as the initial state
    s = siteinds("S=1/2", N; conserve_qns = false)
    states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    ψ = MPS(s, states)
    Sz₀ = expect(ψ, "Sz"; sites = 1:N)

    ## Using a random state as the initial state
    # Random.seed!(1234567)
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # Sz₀ = expect(ψ, "Sz"; sites = 1 : N)                
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
    Sx = complex(zeros(N_diagonal_circuit + 2, N))
    Sy = complex(zeros(N_diagonal_circuit + 2, N))
    Sz = complex(zeros(N_diagonal_circuit + 2, N))
    Sz_Reset = complex(zeros(N_diagonal_circuit + 2, N))

    # ## Construct the holoQUADS circuit 
    # Random.seed!(2000)
    
    for measure_ind = 1:num_measurements
        println("")
        println("")
        println("############################################################################")
        println("#########  GENERATE SAMPLE #$measure_ind ")
        println("############################################################################")
        println("")
        println("")

        # Compute the overlap between the original and time evolved wavefunctions
        ψ_copy = deepcopy(ψ)
        ψ_overlap = Complex{Float64}[]
        tensor_index=1

        tmp_overlap = abs(inner(ψ, ψ_copy))
        println("The overlap of wavefuctions @T=0 is: $tmp_overlap")
        append!(ψ_overlap, tmp_overlap)

        
        # Construct the left lightcone and measure the leftmost two sites
        construct_left_lightcone(ψ_copy, N_time_slice, tau, cutoff, s)
        normalize!(ψ_copy)
        if measure_ind == 1
            Sz[2 * tensor_index - 1, :] = expect(ψ_copy, "Sz"; sites = 1:N)
        end
        Sz_sample[measure_ind, 1:2] = sample(ψ_copy, 1)
        if measure_ind == 1
            Sz_Reset[2 * tensor_index - 1, :] = expect(ψ_copy, "Sz"; sites = 1:N)
        end
        # normalize!(ψ_copy)

        # Construct the diagonal circuit and measure the corresponding sites
        if N_diagonal_circuit > 1E-8
            @time for ind₁ = 1:N_diagonal_circuit
                gate_seeds = []
                for gate_ind = 1:N_time_slice
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

                for ind₂ = 1:N_time_slice
                    tmp_starting_index = gate_seeds[ind₂]
                    if tmp_starting_index - 1 < 1E-8
                        tmp_ending_index = N
                    else
                        tmp_ending_index = tmp_starting_index - 1
                    end
                    @show tmp_starting_index, tmp_ending_index
                    diagonal_gate = construct_diagonal_layer(
                        tmp_starting_index,
                        tmp_ending_index,
                        s,
                        tau,
                    )
                    ψ_copy = apply(diagonal_gate, ψ_copy; cutoff)
                end
                normalize!(ψ_copy)

                if measure_ind - 1 < 1E-8
                    tmp_Sz = expect(ψ_copy, "Sz"; sites = 1:N)
                    Sz[ind₁+1, :] = tmp_Sz
                    println("")
                    @show tmp_Sz
                end
                Sz_sample[measure_ind, 2*ind₁+1:2*ind₁+2] = sample(ψ_copy, 2 * ind₁ + 1)
                if measure_ind - 1 < 1E-8
                    tmp_Sz = expect(ψ_copy, "Sz"; sites = 1:N)
                    Sz_Reset[ind₁+1, :] = tmp_Sz
                    println("")
                    @show tmp_Sz
                end
            end
        end

        # Label the tensor which is measured before applying the right corner
        tensor_index = (N_diagonal_circuit + 1) % div(N, 2)
        if tensor_index < 1E-8
            tensor_index = div(N, 2)
        end

        # The right light cone structure in the holoQAUDS circuit
        @time for ind₁ = 1:div(N_time_slice, 2)
            number_of_gates = ind₁
            starting_tensor = (tensor_index - 1 + div(N, 2)) % div(N, 2)
            if starting_tensor < 1E-8
                starting_tensor = div(N, 2)
            end
            starting_point = 2 * starting_tensor

            if abs(ind₁ - div(N_time_slice, 2)) > 1E-8
                index_array = [starting_point, starting_point - 1]
            else
                index_array = [starting_point]
            end
            @show ind₁, number_of_gates, index_array

            for tmp_index in index_array
                right_light_cone_layer =
                    construct_right_light_cone_layer(tmp_index, number_of_gates, N, s, tau)
                for temporary_gate in right_light_cone_layer
                    ψ_copy = apply(temporary_gate, ψ_copy; cutoff)
                    normalize!(ψ_copy)
                end
            end
        end
        normalize!(ψ_copy)

        if measure_ind - 1 < 1E-8
            tmp_Sz = expect(ψ_copy, "Sz"; sites = 1:N)
            Sz[N_diagonal_circuit+2, :] = tmp_Sz
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
    # file = h5open(
    #     "Data_Benchmark/holoQUADS_Circuit_Heisenberg_Finite_N$(N)_T$(floquet_time)_AFM.h5",
    #     "w",
    # )
    # write(file, "Initial Sz", Sz₀)
    # # write(file, "Sx", Sx)
    # # write(file, "Sy", Sy)
    # write(file, "Sz", Sz)
    # write(file, "Sz after reset", Sz_Reset)
    # # write(file, "Cxx", Cxx)
    # # write(file, "Cyy", Cyy)
    # # write(file, "Czz", Czz)
    # write(file, "Sz samples", Sz_sample)
    # # write(file, "Wavefunction Overlap", ψ_overlap)
    # close(file)
    return
end
