## 03/29/2023
## Implement time evolution block decimation (TEBD) for the SDKI model
## Using a brick-wall pattern

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, site, timer
using Base: Float64, Real, Integer
using Random
using TimerOutputs

# using AppleAccelerate
# using AppleAccelerateLinAlgWrapper
using MKL
using LinearAlgebra
BLAS.set_num_threads(8)

const time_machine = TimerOutput()
ITensors.disable_warn_order()

include("Entanglement.jl")
include("TEBD_Time_Evolution_Gates.jl")
include("ObtainBond.jl")

let 
    N = 100
    cutoff=1E-8
    Δτ = 1.0 
    ttotal = 20
    h = 0.2                                            # an integrability-breaking longitudinal field h 

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false)

    # Construct layers of gates to perform the real-time evolution
    @timeit time_machine "Generating sequences of gates" begin
        gates = Vector{ITensor}()
        even_layer = build_a_layer_of_gates!(2, N-2, N, h, Δτ, s, gates)
        odd_layer = build_a_layer_of_gates!(1, N-1, N, h, Δτ, s, gates)
        kick_gates = build_kick_gates_TEBD(s, 1, N)
    end

    # Initialize the wavefunction using a Neel state
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = MPS(s, states)
    ψ_copy = deepcopy(ψ)
    ψ_overlap = Complex{Float64}[]

    # # Intialize the wvaefunction as a random MPS   
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # Random.seed!(87900)
    # ψ = randomMPS(s, states, linkdims = 2)
    # # @show eltype(ψ), eltype(ψ[1])
    # # @show maxlinkdim(ψ)
    # ψ_copy = deepcopy(ψ)
    # ψ_overlap = Complex{Float64}[]

    timeSlices = Int(ttotal / Δτ) + 1
    println("Total number of time slices that need to be saved is : $(timeSlices)")
    
    # Allocate memory for physical observables
    @timeit time_machine "Allocate memory" begin
        # Local observables including various one-point functions
        Sx = Array{ComplexF64}(undef, timeSlices, N)
        Sy = Array{ComplexF64}(undef, timeSlices, N)
        Sz = Array{ComplexF64}(undef, timeSlices, N)
        
        # Equal-time observables including various two-point functions
        Cxx = Array{ComplexF64}(undef, timeSlices, N * N)
        Cyy = Array{ComplexF64}(undef, timeSlices, N * N)
        Czz = Array{ComplexF64}(undef, timeSlices, N * N)

        # Entanglement etc. for the whole chain
        SvN = Array{Float64}(undef, timeSlices, N - 1)
        Bond = Array{Float64}(undef, timeSlices, N - 1)
    end
    
    # Measure local observables, bond dimension and von Neumann entanglement entropy of the intiial wave function
    @timeit time_machine "Measure the initial wavefunction" begin
        Sx[1, :] = expect(ψ_copy, "Sx"; sites = 1 : N)
        # Sy[1, :] = expect(ψ_copy, "Sy"; sites = 1 : N)
        Sz[1, :] = expect(ψ_copy, "Sz"; sites = 1 : N)

        # Cxx[1, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N)
        # # Cyy[1, :] = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N)
        # Czz[1, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)
        
        SvN[1, :] = entanglement_entropy(ψ_copy, N)
        Bond[1, :] = obtain_bond_dimension(ψ_copy, N)
    end
    
    distance = Int(1.0 / Δτ)

    # Real time dynamics of the SDKI model
    for time in 0 : Δτ : ttotal
        time ≈ ttotal && break
        index = Int(time/Δτ) + 1
        
        # Apply the kicked gates at integer time
        @timeit time_machine "Applying one-site gates" if (abs((time / Δτ) % distance) < 1E-8)
            ψ_copy = apply(kick_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
        end

        # Apply the two-site gates which include the Ising interaction and longitudinal fields
        @timeit time_machine "Applying two-site gates" begin
            ψ_copy = apply(gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
        end
        
        # Local observables e.g. Sx, Sz
        @timeit time_machine "Compute one-point function" begin 
            Sx[index, :] = expect(ψ_copy, "Sx"; sites = 1 : N) 
            # Sy[index, :] = expect(ψ_copy, "Sy"; sites = 1 : N)
            Sz[index, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
        end

        # @timeit time_machine "Compute two-point function" begin
        # # Correlation functions e.g. Cxx, Czz
        #     Cxx[index, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N)
        #     # Cyy[index, :] = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N)
        #     Czz[index, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)
        # end

        @timeit time_machine "Compute von Neumann entanglement entropy and bond dimension" begin
            SvN[index, :] = entanglement_entropy(ψ_copy, N)
            Bond[index, :] = obtain_bond_dimension(ψ_copy, N)
        end

        @show time_machine
        # @show SvN[index, :]
        # @show Sz[index, :]
        @show Bond[index, :]

        # Store output data in a HDF5 file
        h5open("../Scalable_Data/TEBD_N$(N)_h$(h)_tau$(Δτ)_T$(ttotal)_cuttoff$(cutoff).h5", "w") do file
            write(file, "Sx", Sx)
            write(file, "Sy", Sy)
            write(file, "Sz", Sz)
            write(file, "Cxx", Cxx)
            write(file, "Cyy", Cyy)
            write(file, "Czz", Czz)
            write(file, "Entropy", SvN)
            write(file, "Bond", Bond)
            write(file, "Initial Sx", Sx[1, :])
            write(file, "Initial Sy", Sy[1, :])
            write(file, "Initial Sz", Sz[1, :])
        end
    end
    return
end  