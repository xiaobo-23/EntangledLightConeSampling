## Implement time evolution block decimation (TEBD) for the kicked Ising model
## Using a brick-wall patternn 
## Compute non-equal time correlation functions 

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, site, timer
using Base: Float64, Real, Integer
using Random
using Dates
using TimerOutputs


# using AppleAccelerate
# using AppleAccelerateLinAlgWrapper

using MKL
using LinearAlgebra
BLAS.set_num_threads(8)

const time_machine = TimerOutput()
ITensors.disable_warn_order()

include("src/SDKI/Entanglement.jl")
include("src/SDKI/TEBD_Time_Evolution_Gates.jl")
include("src/SDKI/ObtainBond.jl")

let 
    N=100
    cutoff=1E-8
    Δτ=1
    ttotal=5
    h=0.2                                              # an integrability-breaking longitudinal field h 
    time_dependent=true
    # bond_dimension_upper_bound=10

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
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # ψ = MPS(s, states)
    ψ = randomMPS(s, linkdims = 16)
    ψ_copy = deepcopy(ψ)
    ψ_overlap = Complex{Float64}[]

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

        # Time-dependent spin correlation funciton
        if time_dependent
            Cxx_time = Array{ComplexF64}(undef, timeSlices, N)
            # Cyy_time = Array{ComplexF64}(undef, timeSlices, N)
            # Czz_time = Array{ComplexF64}(undef, timeSlices, N)
        end

        # Entanglement etc. for the whole chain
        SvN = Array{Float64}(undef, timeSlices, N - 1)
        Bond = Array{Float64}(undef, timeSlices, N - 1)
    end
    
    # Measure local observables, bond dimension and von Neumann entanglement entropy of the intiial wave function
    @timeit time_machine "Measure the initial wavefunction" begin
        Sx[1, :] = expect(ψ_copy, "Sx"; sites = 1 : N)
        Sy[1, :] = expect(ψ_copy, "iSy"; sites = 1 : N)
        Sz[1, :] = expect(ψ_copy, "Sz"; sites = 1 : N)

        Cxx[1, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N)
        Cyy[1, :] = correlation_matrix(ψ_copy, "iSy", "iSy"; sites = 1 : N)
        Czz[1, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)
        
        SvN[1, :] = entanglement_entropy(ψ_copy, N)
        Bond[1, :] = obtain_bond_dimension(ψ_copy, N)
    end
    
    distance = Int(1.0 / Δτ)
    index = 2

    # Choose a reference site and act on the initial wavefunction with 
    # the corresponding operator before doing any time evolution
    if time_dependent
        reference_position = 50
        reference_operator = op("Sz", s[reference_position])
        ψ_right = apply(reference_operator, ψ_copy; cutoff)
        normalize!(ψ_right)
        println("After applying the local spin operator, normalize the wavefunction leads to :")
        @show inner(ψ_right', ψ_right)
    end
    

    # Time evolve the wavefunction using TEBD
    for time in 0.0 : Δτ : ttotal
        time ≈ ttotal && break

        # Apply the one-site gates which include the kicked transverse field at integer time
        @timeit time_machine "Applying one-site gates" if (abs((time / Δτ) % distance) < 1E-8)
            @show time/Δτ, abs((time / Δτ) % distance), distance
            ψ_copy = apply(kick_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)

            if time_dependent
                ψ_right = apply(kick_gates, ψ_right; cutoff)
                normalize!(ψ_right)
            end

            # Setting a maximum bond dimension during the time evolution
            # ψ_copy = apply(kick_gates, ψ_copy; cutoff, maxdim=bond_dimension_upper_bound)
        end

        # Apply the two-site gates which include the Ising interaction and longitudinal fields
        @timeit time_machine "Applying two-site gates" begin
            ψ_copy = apply(gates, ψ_copy; cutoff)
            normalize!(ψ_copy)

            if time_dependent
                ψ_right = apply(gates, ψ_right; cutoff)
                normalize!(ψ_right)
            end

            # Setting a maximum bond dimension during the time evolution
            # ψ_copy = apply(gates, ψ_copy; cutoff, maxdim=bond_dimension_upper_bound)
        end
        
        # Local observables e.g. Sx, Sz
        @timeit time_machine "Compute one-point function" begin 
            Sx[index, :] = expect(ψ_copy, "Sx"; sites = 1 : N) 
            Sy[index, :] = expect(ψ_copy, "iSy"; sites = 1 : N)
            Sz[index, :] = expect(ψ_copy, "Sz"; sites = 1 : N)
        end

        @timeit time_machine "Compute two-point function" begin
        # Correlation functions e.g. Cxx, Czz
            Cxx[index, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N)
            Cyy[index, :] = correlation_matrix(ψ_copy, "iSy", "iSy"; sites = 1 : N)
            Czz[index, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)
        end

        @timeit time_machine "Compute von Neumann entanglement entropy and bond dimension" begin
            SvN[index, :] = entanglement_entropy(ψ_copy, N)
            Bond[index, :] = obtain_bond_dimension(ψ_copy, N)
        end

        
        if time_dependent
            for site_ind in eachindex(collect(1:N))
                # @show site_ind
                tmp_ψ = deepcopy(ψ_right)
                time_operator = op("Sz", s[site_ind])
                tmp_ψ = apply(time_operator, tmp_ψ; cutoff)
                Cxx_time[index, site_ind] = inner(ψ_copy, tmp_ψ)
            end
        end

        # @show Bond[index, :]
        # @show SvN[index, :]
        # @show Sz[index, :]
        
        index += 1
        
        # Store output data in a HDF5 file
        h5open("data/kicked_ising/TEBD_N$(N)_h$(h)_tau$(Δτ)_T$(ttotal)_random_ref$(reference_position).h5", "w") do file
            write(file, "Sx", Sx)
            write(file, "Sy", Sy)
            write(file, "Sz", Sz)
            write(file, "Cxx", Cxx)
            write(file, "CxxTime", Cxx_time)
            write(file, "Cyy", Cyy)
            write(file, "Czz", Czz)
            write(file, "SvN", SvN)
            write(file, "chi", Bond)
            write(file, "Sx0", Sx[1, :])
            write(file, "Sy0", Sy[1, :])
            write(file, "Sz0", Sz[1, :])
        end
    end
   
    # Show information (time, memorty, etc.) about the time evolution
    @show time_machine
    return
end  