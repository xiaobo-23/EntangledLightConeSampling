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


let
    #####################################################################################################################################
    ##### Define parameters used in the holoQUADS circuit                                                                           #####
    ##### Given the light-cone structure of the real-time dynamics, circuit depth and number of sites are related                   #####
    #####################################################################################################################################
    global floquet_time=2.0
    global Δτ=0.1        
    global running_cutoff=1E-8                                                                         # Trotter decomposition time step 
    global N=50
    global N_time_slice = Int(floquet_time/Δτ) * 2
    global unit_cell_size = N_time_slice + 2                                                            # Number of total sites on a MPS
    
    number_of_DC = div(N - unit_cell_size, 2)               
    num_measurements = 1
    measurement_type = "Sz"

    @show floquet_time, typeof(floquet_time)
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
    bitstring_sample = real(zeros(num_measurements, N))

    #####################################################################################################################################
    # Sample from the time-evolved wavefunction and store the measurements
    #####################################################################################################################################    
    # Sx = complex(zeros(N_diagonal_circuit + 2, N))
    # Sy = complex(zeros(N_diagonal_circuit + 2, N))
    Sz = Vector{ComplexF64}(undef, N)
    Sz_Reset = Vector{ComplexF64}(undef, N)

    # ## Construct the holoQUADS circuit 
    # Random.seed!(2000)
    
    for measure_index = 1:num_measurements
        println("")
        println("")
        println("############################################################################")
        println("#########  GENERATE SAMPLE #$measure_index ")
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
        construct_left_lightcone(ψ_copy, N_time_slice, s)
        # normalize!(ψ_copy)
        @time if measure_index == 1
            Sz[2 * tensor_index - 1 : 2 * tensor_index] = expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)
        end
        @time bitstring_sample[measure_index, 1:2] = sample(ψ_copy, 1)
        @time if measure_index == 1
            Sz_Reset[2 * tensor_index - 1 : 2 * tensor_index] = expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)
        end

        # Construct the diagonal circuit and measure the corresponding sites
        if number_of_DC > 1E-8
            @time for index_DC = 1 : number_of_DC
                construct_diagonal_part(ψ_copy, N_time_slice, index_DC, s)
                tensor_index += 1

                if measure_index == 1
                    Sz[2 * tensor_index - 1 : 2 * tensor_index] = expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)                   
                end
                bitstring_sample[measure_index, 2 * tensor_index - 1 : 2 * tensor_index] = sample(ψ_copy, 2 * tensor_index - 1)
                if measure_index == 1
                    Sz_Reset[2 * tensor_index - 1 : 2 * tensor_index] = expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)
                end
            end
        end

        # The right light cone structure in the holoQAUDS circuit
        @time for index_RL = 1:div(N_time_slice, 2)
            tensor_index += 1
            construct_right_lightcone(ψ_copy, index_RL, N_time_slice, s)
            
            if measure_index == 1
                Sz[2 * tensor_index - 1 : 2 * tensor_index] = expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)
            end
            bitstring_sample[measure_index, 2 * tensor_index - 1 : 2 * tensor_index] = sample(ψ_copy, 2 * tensor_index - 1)
            if measure_index == 1
                Sz_Reset[2 * tensor_index - 1 : 2 * tensor_index] = expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)
            end
        end        
    end
    replace!(bitstring_sample, 1.0 => 0.5, 2.0 => -0.5)

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
    # write(file, "Sz samples", bitstring_sample)
    # # write(file, "Wavefunction Overlap", ψ_overlap)
    # close(file)
    return
end