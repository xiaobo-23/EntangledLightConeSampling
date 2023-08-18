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


include("src/Heisenberg/Sample.jl")
include("src/Heisenberg/holoQUADS_Gates.jl")
include("src/Heisenberg/Entanglement.jl")
include("src/Heisenberg/ObtainBond.jl")
include("src/Heisenberg/Sample_and_Measure.jl")


let
    #####################################################################################################################################
    ##### Define parameters used in the holoQUADS circuit                                                                           #####
    ##### Given the light-cone structure of the real-time dynamics, circuit depth and number of sites are related                   #####
    #####################################################################################################################################
    global floquet_time=1.0
    global Δτ=0.1        
    global running_cutoff=1E-8                                                                         # Trotter decomposition time step 
    global N=500
    global N_time_slice = Int(floquet_time/Δτ) * 2
    global unit_cell_size = N_time_slice + 2                                                            # Number of total sites on a MPS
    
    number_of_DC = div(N - unit_cell_size, 2)               
    number_of_samples = 1
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

    ## 08/17/2023
    ## Set up the variables for measurements
    Sx = Vector{ComplexF64}(undef, N)
    Sy = Vector{ComplexF64}(undef, N)
    Sz = Vector{ComplexF64}(undef, N)
    # Sz_Reset = Vector{ComplexF64}(undef, N)
    
    SvN  = Array{Float64}(undef, number_of_samples, N * (N - 1))
    bond = Array{Float64}(undef, number_of_samples, N * (N - 1))
    samples = Array{Float64}(undef, number_of_samples, N)
    samples_bitstring = Array{Float64}(undef, number_of_samples, N)

    #####################################################################################################################################
    ## Take measurements and generate the snapshots 
    #####################################################################################################################################

    for measure_index = 1:number_of_samples
        println("############################################################################")
        println("#########  GENERATE SAMPLE #$measure_index ")
        println("############################################################################")
        println("")
        println("")

        # Make a copy of the original wavefunciton and time evolve the copy
        ψ_copy = deepcopy(ψ)
        tensor_index=1

        # Compute the overlap between the original and time evolved wavefunctions
        # ψ_overlap = Complex{Float64}[]
        # tmp_overlap = abs(inner(ψ, ψ_copy))
        # println("The overlap of wavefuctions @T=0 is: $tmp_overlap")
        # append!(ψ_overlap, tmp_overlap)
        
        # Set up the left lightcone, apply all the gates to, and take measurements of the first two sites
        LC_gates = construct_left_lightcone(ψ_copy, N_time_slice, s)
        ψ_copy = apply(LC_gates, ψ_copy; cutoff=running_cutoff)
        normalize!(ψ_copy)
        # @show inner(ψ, ψ_copy) 

        # Debug the holoQUADS when the collapse part is turned off
        if measure_index == 1
            Sz[2 * tensor_index - 1 : 2 * tensor_index] = (
                expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)
            )
        end

        sample_and_measure!(ψ_copy, 2 * tensor_index - 1, measure_index, measurement_type, 
        SvN, bond, samples, samples_bitstring)

        # # Measure the bond dimension, entanglement entropy and sample the wavefunction
        # measure_chi_and_SvN!(ψ_copy, 2 * tensor_index - 1, measure_index, SvN, bond)
        
        # samples[measure_index, 2 * tensor_index - 1 : 2 * tensor_index] = (
        #     expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index))
        # samples_bitstring[measure_index, 2 * tensor_index - 1 : 2 * tensor_index] = (
        #     expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index))
        # normalize!(ψ_copy)

        # measure_chi_and_SvN!(ψ_copy, 2 * tensor_index, measure_index, SvN, bond)

        # Construct the diagonal circuit and measure the corresponding sites
        if number_of_DC > 1E-8
            @time for index_DC = 1 : number_of_DC
                tensor_index += 1
                DC_gates = construct_diagonal_part(ψ_copy, N_time_slice, index_DC, s)
                ψ_copy = apply(DC_gates, ψ_copy; cutoff=running_cutoff)
                normalize!(ψ_copy) 
                @show inner(ψ, ψ_copy)
                
                # Debug the holoQUADS citcuit when the collapse part is turned off
                if measure_index == 1
                    Sz[2 * tensor_index - 1 : 2 * tensor_index] = expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index)                   
                end

                sample_and_measure!(ψ_copy, 2 * tensor_index - 1, measure_index, measurement_type, 
                SvN, bond, samples, samples_bitstring)
            end
        end

        # The right light cone structure in the holoQAUDS circuit
        @time for index_RL = 1:div(N_time_slice, 2)
            tensor_index += 1
            RL_gates = construct_right_lightcone(ψ_copy, index_RL, N_time_slice, s)
            ψ_copy = apply(RL_gates, ψ_copy; cutoff=running_cutoff)
            normalize!(ψ_copy) 
            @show inner(ψ, ψ_copy) 

            # Debug the holoQUADS when the measurement/collapse part is turned off
            if measure_index == 1
                Sz[2 * tensor_index - 1 : 2 * tensor_index] = 
                (expect(ψ_copy, measurement_type; sites = 2 * tensor_index - 1 : 2 * tensor_index))
            end
            
            sample_and_measure!(ψ_copy, 2 * tensor_index - 1, measure_index, measurement_type, 
            SvN, bond, samples, samples_bitstring)
        end        
    end
    replace!(samples_bitstring, 1.0 => 0.5, 2.0 => -0.5)

    println("################################################################################")
    println("################################################################################")
    println("Projection in the Sz basis of the initial MPS")
    @show Sz₀
    println("################################################################################")
    println("################################################################################")

    # Store data in a HDF5 file
    h5open("Data_Benchmark/holoQUADS_Circuit_Heisenberg_N$(N)_T$(floquet_time).h5", "w") do file
        write(file, "Initial Sz", Sz₀)
        # write(file, "Sx", Sx)
        # write(file, "Sy", Sy)
        write(file, "Sz", Sz)
        write(file, "MPS/MPO samples", samples)
        write(file, "Bistring samples", samples_bitstring)
        write(file, "SvN", SvN)
        write(file, "chi", bond)
        # write(file, "Wavefunction Overlap", ψ_overlap)
    end
    
    
    return
end