## Implement time evolution block decimation (TEBD) for the one-dimensional Heisenberg model
using ITensors
using ITensors.HDF5
using Random
using TimerOutputs

using MKL
using LinearAlgebra
BLAS.set_num_threads(8)

const time_machine = TimerOutput()
ITensors.disable_warn_order()

include("src/Heisenberg/Entanglement.jl")
include("src/Heisenberg/ObtainBond.jl")


function generate_gates_in_brickwall_pattern!(starting_index :: Int, ending_index :: Int, input_gates, tmp_sites)
    counting_index = 0
    for tmp_index = starting_index : 2 : ending_index
        s1 = tmp_sites[tmp_index]
        s2 = tmp_sites[tmp_index + 1]
        hj =
            op("Sz", s1) * op("Sz", s2) +
            1 / 2 * op("S+", s1) * op("S-", s2) +
            1 / 2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * Δτ * hj)
        push!(input_gates, Gj)
        counting_index += 1
    end
    @show counting_index
end


function generate_gates_in_staircase_pattern!(length_of_chain :: Int, input_gates, tmp_sites)
    count_index = 0
    
    # Make gates (1, 2), (2, 3), (3, 4) ...
    for ind = 1 : length_of_chain - 1
        s1 = tmp_sites[ind]
        s2 = tmp_sites[ind+1]
        hj =
            op("Sz", s1) * op("Sz", s2) +
            1 / 2 * op("S+", s1) * op("S-", s2) +
            1 / 2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * Δτ / 2 * hj)
        push!(input_gates, Gj)
        count_index += 1
    end

    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(input_gates, reverse(input_gates))
    @show 2 * counting_index
end

let
    N = 100
    running_cutoff = 1E-8
    ttotal = 5.2
    global Δτ = 0.1

    # Make an array of 'site' indices
    s = siteinds("S=1/2", N; conserve_qns = false)
    
    # ## 08/16/2023
    # ## Generate the time evolution gates using the staircase pattern for one time slice/step
    # gates = ITensor[]
    # generate_gates_in_staircase_pattern!(N, gates, s)

    
    ## 08/16/2023
    ## Geenrate the time evolution gates using brickwall pattern for one time slice/step
    @timeit time_machine "Generate gates in the brickwall pattern" begin
        gates = ITensor[]
        generate_gates_in_brickwall_pattern!(2, N - 2, gates, s)
        generate_gates_in_brickwall_pattern!(1, N - 1, gates, s)
    end
    
    
    # ## Use a product state e.g. Neel state as the initial state
    # ψ₀ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # ψ = deepcopy(ψ₀)


    ## Use a random state as the initial state
    Random.seed!(1234567)
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ₀ = randomMPS(s, states, linkdims = 2)
    ψ = deepcopy(ψ₀)
    Sz₀ = expect(ψ₀, "Sz"; sites = 1 : N)              


    ## Initialize observables used in the time evolution process
    number_of_measurements = Int(ttotal / Δτ) + 1
    @timeit time_machine "Memory Allocation" begin
        Sx = Array{ComplexF64}(undef, number_of_measurements, N)
        Sy = Array{ComplexF64}(undef, number_of_measurements, N)
        Sz = Array{ComplexF64}(undef, number_of_measurements, N)
        SvN = Array{Float64}(undef, number_of_measurements, N - 1)
        Bond = Array{Float64}(undef, number_of_measurements, N - 1)
        Overlap = Vector{Float64}(undef, number_of_measurements)    
    end
    

    # Use TEBD to evolve the wavefunction in real time && taking measurements of local observables
    index = 1
    @time for time = 0.0 : Δτ : ttotal
        ## Measure and time one-point functions
        @timeit time_machine "Compute one-point function" begin
            # tmp_Sx = expect(ψ, "Sx", sites = 1 : N)
            # Sx[index, :] = tmp_Sx

            ## 08/18/2023
            ## Fix the Sy measurements in the ITensor code
            # tmp_Sy = epxect(ψ, "Sy", sites = 1 : N)
            # Sy[index, :] = tmp_Sy    

            tmp_Sz = expect(ψ, "Sz", sites = 1 : N)
            Sz[index, :] = tmp_Sz
        end
        


        ## Measure the overlap between time-evolved wavefunction and the original wavefunction
        @timeit time_machine "Compute overlap of wavefunctions" begin
            tmp_overlap = abs(inner(ψ, ψ₀))
            Overlap[index] = tmp_overlap
        end 
        
        ## Measure bond dimension and von Neumann entanglement entropy
        @timeit time_machine "Compute von Neumann entanglement entropy and bond dimension" begin
            SvN[index, :]  = entanglement_entropy(ψ, N)
            Bond[index, :] = obtain_bond_dimension(ψ, N)
        end

        ## Print the fidelity of wavefunction and Sz to monitor the time evolution
        println("")
        println("At time step $time, Sz is $tmp_Sz, fidelity is $tmp_overlap")
        println("")

        time ≈ ttotal && break
        @timeit time_machine "Apply time evolution gates to the wavefunction" begin
            ψ = apply(gates, ψ; cutoff = running_cutoff)    
        end
        normalize!(ψ)
        
        index += 1
        
        ## Store results in a output file
        h5open("Data_Benchmark/TEBD_Heisenberg_N$(N)_T$(ttotal)_tau$(Δτ)_Brickwall.h5", "w") do file
            # write(file, "Sx", Sx)
            # write(file, "Sy", Sy)
            write(file, "Sz", Sz)
            write(file, "Overlap", Overlap)
            write(file, "Entropy", SvN)
            write(file, "Bond", Bond)
        end
    end
    
    @show time_machine
    return


end