## 03/29/2023
## Implement time evolution block decimation (TEBD) using a brick wall pattern

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, site, timer
using Base: Float64, Real, Integer
using Random
using Dates
using TimerOutputs
using MKL
using AppleAccelerate
using AppleAccelerateLinAlgWrapper

const time_machine = TimerOutput()
ITensors.disable_warn_order()

include("Entanglement.jl")
include("TEBD_Time_Evolution_Gates.jl")

let
    N = 100
    cutoff = 1E-8
    Δτ = 1.0
    ttotal = 20
    h = 0.2                                            # an integrability-breaking longitudinal field h 

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false)

    # Construct a layer (odd & even) of gates for the SDKI model
    gates = ITensor[]
    even_layer = build_a_layer_of_gates(2, N - 2, N, h, Δτ, s)
    for tmp₁ in even_layer
        push!(gates, tmp₁)
    end

    odd_layer = build_a_layer_of_gates(1, N - 1, N, h, Δτ, s)
    for tmp₂ in odd_layer
        push!(gates, tmp₂)
    end

    # Construct a layer of kicked gates
    kick_gates = build_kick_gates(s, 1, N)

    # Initialize the wavefunction using a Neel state
    states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
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


    # Local observables including various one-point functions
    Sx = Array{ComplexF64}(undef, timeSlices, N)
    Sy = Array{ComplexF64}(undef, timeSlices, N)
    Sz = Array{ComplexF64}(undef, timeSlices, N)

    # Equal-time observables including various two-point functions
    Cxx = Array{ComplexF64}(undef, timeSlices, N * N)
    Cyy = Array{ComplexF64}(undef, timeSlices, N * N)
    Czz = Array{ComplexF64}(undef, timeSlices, N * N)
    SvN = Array{Float64}(undef, timeSlices, N - 1)

    # Take measurements of the initial wavefunction
    Sx[1, :] = expect(ψ_copy, "Sx"; sites = 1:N)
    Sy[1, :] = expect(ψ_copy, "Sy"; sites = 1:N)
    Sz[1, :] = expect(ψ_copy, "Sz"; sites = 1:N)

    Cxx[1, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1:N)
    Cyy[1, :] = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1:N)
    Czz[1, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1:N)
    # append!(ψ_overlap, abs(inner(ψ, ψ_copy)))

    distance = Int(1.0 / Δτ)
    index = 2
    for time = 0.0:Δτ:ttotal
        time ≈ ttotal && break
        SvN[index-1, :] = entanglement_entropy(ψ_copy, N)

        # Apply the kicked gates at integer time
        @timeit time_machine "kick gates" if (abs((time / Δτ) % distance) < 1E-8)
            ψ_copy = apply(kick_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
            # append!(ψ_overlap, abs(inner(ψ, ψ_copy)))
        end

        # Apply the two-site gates which include the Ising interaction and longitudinal fields
        @timeit time_machine "two-site gates" begin
            ψ_copy = apply(gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
            # append!(ψ_overlap, abs(inner(ψ, ψ_copy)))
        end

        # Local observables e.g. Sx, Sz
        Sx[index, :] = expect(ψ_copy, "Sx"; sites = 1:N)
        Sy[index, :] = expect(ψ_copy, "Sy"; sites = 1:N)
        Sz[index, :] = expect(ψ_copy, "Sz"; sites = 1:N)

        # Correlation functions e.g. Cxx, Czz
        Cxx[index, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1:N)
        Cyy[index, :] = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1:N)
        Czz[index, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1:N)
        index += 1

        # append!(ψ_overlap, abs(inner(ψ, ψ_copy)))
        @show time_machine
    end

    # Store data into a hdf5 file
    file = h5open("TEBD_N$(N)_h$(h)_tau$(Δτ)_T$(ttotal).h5", "w")
    write(file, "Sx", Sx)
    write(file, "Sy", Sy)
    write(file, "Sz", Sz)
    write(file, "Cxx", Cxx)
    write(file, "Cyy", Cyy)
    write(file, "Czz", Czz)
    # write(file, "Wavefunction Overlap", ψ_overlap)
    write(file, "Entropy", SvN)
    write(file, "Initial Sx", Sx[1, :])
    write(file, "Initial Sy", Sy[1, :])
    write(file, "Initial Sz", Sz[1, :])
    close(file)

    return
end