## 02/07/2023
## Implement the holographic quantum circuit for the Heisenberg model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product, Float64
using HDF5: file
using Random
using TimerOutputs

ITensors.disable_warn_order()


using MKL 
using LinearAlgebra
BLAS.set_num_threads(8)

include("src/Heisenberg/Sample.jl")
include("src/Heisenberg/holoQUADS_Gates.jl")
include("src/Heisenberg/Entanglement.jl")
include("src/Heisenberg/ObtainBond.jl")
include("src/Heisenberg/Sample_and_Measure.jl")

const time_machine = TimerOutput()

let
    #####################################################################################################################################
    ##### Define parameters used in the holoQUADS circuit                                                                           #####
    ##### Given the light-cone structure of the real-time dynamics, circuit depth and number of sites are related                   #####
    #####################################################################################################################################
    global floquet_time=3.0
    global Δτ=0.1        
    global running_cutoff=1E-8                                                                         # Trotter decomposition time step 
    global N=100
    global N_time_slice = Int(floquet_time/Δτ) * 2
    global unit_cell_size = N_time_slice + 2                                                            # Number of total sites on a MPS
    
    number_of_DC = div(N - unit_cell_size, 2)               
    number_of_samples = 1
    measurement_type = "Sz"
    sample_index=0

    @show floquet_time, typeof(floquet_time)
    #####################################################################################################################################
    #####################################################################################################################################

    # ## Use a product state e.g. Neel state as the initial state
    # s = siteinds("S=1/2", N; conserve_qns = false)
    # states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    # ψ = MPS(s, states)
    # Sz₀ = expect(ψ, "Sz"; sites = 1:N)
    # Random.seed!(123456)
 
    # Use a random state as the initial state for debug purpose
    Random.seed!(1234567)
    s = siteinds("S=1/2", N; conserve_qns = false)
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = randomMPS(s, states, linkdims = 2)
    Sz₀ = expect(ψ, "Sz"; sites = 1 : N)                


    ## 08/17/2023
    ## Set up the variables for measurements
    @timeit time_machine "Allocation" begin
        Sx = Vector{ComplexF64}(undef, N)
        Sy = Vector{ComplexF64}(undef, N)
        Sz = Vector{ComplexF64}(undef, N)
        # Sz_Reset = Vector{ComplexF64}(undef, N)
        
        SvN  = Array{Float64}(undef, number_of_samples, N * (N - 1))
        bond = Array{Float64}(undef, number_of_samples, N * (N - 1))
        samples = Array{Float64}(undef, number_of_samples, N)
        samples_bitstring = Array{Float64}(undef, number_of_samples, N)
    end

    #####################################################################################################################################
    ## Take measurements and generate the snapshots 
    #####################################################################################################################################

    for measure_index = 1:number_of_samples
        println("############################################################################")
        println("#########  GENERATE SAMPLE #$measure_index ")
        println("############################################################################")
        println("")
        println("")
        # Random.seed!(123456)

        # Make a copy of the original wavefunciton and time evolve the copy
        ψ_copy = deepcopy(ψ)
        tensor_index=1

        # Compute the overlap between the original and time evolved wavefunctions
        # ψ_overlap = Complex{Float64}[]
        # tmp_overlap = abs(inner(ψ, ψ_copy))
        # println("The overlap of wavefuctions @T=0 is: $tmp_overlap")
        # append!(ψ_overlap, tmp_overlap)
        
        # Construct the left lightcone, apply all the gates, and measure the first two sites
        LC_gates = construct_left_lightcone(ψ_copy, N_time_slice, s)
        @timeit time_machine "LLC Evolution" begin
            ψ_copy = apply(LC_gates, ψ_copy; cutoff=running_cutoff)
        end
        normalize!(ψ_copy)
        # @show inner(ψ, ψ_copy) 

        # Debug the holoQUADS when the collapse part is turned off
        if measure_index == 1
            Sx[2 * tensor_index - 1 : 2 * tensor_index] = (
                expect(ψ_copy, "Sx"; sites = 2 * tensor_index - 1 : 2 * tensor_index)
            )
            Sz[2 * tensor_index - 1 : 2 * tensor_index] = (
                expect(ψ_copy, "Sz"; sites = 2 * tensor_index - 1 : 2 * tensor_index)
            )
        end

        @timeit time_machine "Measure & Sample LLC" begin
            sample_and_measure!(ψ_copy, 2 * tensor_index - 1, measure_index, measurement_type, 
            SvN, bond, samples, samples_bitstring)
        end
        
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
            for index_DC = 1 : number_of_DC
                tensor_index += 1
                DC_gates = construct_diagonal_part(ψ_copy, N_time_slice, index_DC, s)
                @timeit time_machine "DC Evolution" begin
                    ψ_copy = apply(DC_gates, ψ_copy; cutoff=running_cutoff)
                end
                normalize!(ψ_copy) 
                @show inner(ψ, ψ_copy)
                
                # Debug the holoQUADS citcuit when the collapse part is turned off
                if measure_index == 1
                    Sx[2 * tensor_index - 1 : 2 * tensor_index] = (
                        expect(ψ_copy, "Sx"; sites = 2 * tensor_index - 1 : 2 * tensor_index)
                    )
                    Sz[2 * tensor_index - 1 : 2 * tensor_index] = (
                        expect(ψ_copy, "Sz"; sites = 2 * tensor_index - 1 : 2 * tensor_index)
                    )                   
                end

                @timeit time_machine "Measure & Sample DC" begin
                    sample_and_measure!(ψ_copy, 2 * tensor_index - 1, measure_index, measurement_type, 
                    SvN, bond, samples, samples_bitstring)
                end
            end
        end

        # The right light cone structure in the holoQAUDS circuit
        for index_RL = 1:div(N_time_slice, 2)
            tensor_index += 1
            RL_gates = construct_right_lightcone(ψ_copy, index_RL, N_time_slice, s)
            @timeit time_machine "RLC Evolution" begin
                ψ_copy = apply(RL_gates, ψ_copy; cutoff=running_cutoff)
            end
            normalize!(ψ_copy) 
            @show inner(ψ, ψ_copy) 

            # Debug the holoQUADS when the measurement/collapse part is turned off
            if measure_index == 1
                Sx[2 * tensor_index - 1 : 2 * tensor_index] = (
                    expect(ψ_copy, "Sx"; sites = 2 * tensor_index - 1 : 2 * tensor_index)
                )
                Sz[2 * tensor_index - 1 : 2 * tensor_index] = (
                    expect(ψ_copy, "Sz"; sites = 2 * tensor_index - 1 : 2 * tensor_index)
                )      
            end

            @timeit time_machine "Measure & Sample RLC" begin
                sample_and_measure!(ψ_copy, 2 * tensor_index - 1, measure_index, measurement_type, 
                SvN, bond, samples, samples_bitstring)
            end
        end        
    end
    replace!(samples_bitstring, 1 => 0.5, 2 => -0.5)
    @show time_machine

    # println("################################################################################")
    # println("################################################################################")
    # println("Projection in the Sz basis of the initial MPS")
    # @show Sz₀
    # println("################################################################################")
    # println("################################################################################")

    # Store data in a HDF5 file
    h5open("Data_Test/holoQUADS_Heisenberg_N$(N)_T$(floquet_time)_Sample$(sample_index).h5", "w") do file
        write(file, "Initial Sz", Sz₀)
        write(file, "Sx", Sx)
        # write(file, "Sy", Sy)
        write(file, "Sz", Sz)
        write(file, "MPS/MPO samples", samples)
        write(file, "Bitstring samples", samples_bitstring)
        write(file, "SvN", SvN)
        write(file, "chi", bond)
        # write(file, "Wavefunction Overlap", ψ_overlap)
    end
    return
end