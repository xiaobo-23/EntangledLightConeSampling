## 03/29/2023
## Implement time evolution block decimation (TEBD) using a brick wall pattern

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, site
using Base: Float64, Real
using Random
using Dates
include("Entanglement.jl")
ITensors.disable_warn_order()



function build_a_layer_of_gates(starting_index :: Int, ending_index :: Int, upper_bound :: Int, 
    amplitude :: Real, delta_tau :: Real, tmp_sites)
    tmp_gates = []
    for ind in starting_index : 2 : ending_index
        s1 = tmp_sites[ind]
        s2 = tmp_sites[ind + 1]

        if (ind - 1 < 1E-8)
            tmp1 = 2 
            tmp2 = 1
        elseif (abs(ind - (upper_bound - 1)) < 1E-8)
            tmp1 = 1
            tmp2 = 2
        else
            tmp1 = 1
            tmp2 = 1
        end

        # hj = tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2)   
        # hj = π * op("Sz", s1) * op("Sz", s2) + tmp1 * amplitude * op("Sz", s1) * op("Id", s2) + tmp2 * amplitude * op("Id", s1) * op("Sz", s2) 
        hj = π/2 * op("Sz", s1) * op("Sz", s2) + tmp1 * amplitude * op("Sz", s1) * op("Id", s2) + tmp2 * amplitude * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * delta_tau * hj)
        push!(tmp_gates, Gj)
    end
    return tmp_gates
end

# Build a sequence of one-site kick gates
function build_kick_gates(tmp_site, starting_index :: Int, ending_index :: Int)
    tmp_gates = ITensor[]
    for index in starting_index : ending_index
        tmpS = tmp_site[index]
        tmpHamiltonian = π/2 * op("Sx", tmpS)
        tmpGate = exp(-1.0im * tmpHamiltonian)
        push!(tmp_gates, tmpGate)
    end
    return tmp_gates
end

let 
    N = 50
    cutoff = 1E-8
    Δτ = 1.0; ttotal = 10
    h = 0.2                                            # an integrability-breaking longitudinal field h 

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);     # s = siteinds("S=1/2", N; conserve_qns = true)

    # Construct a layer (odd & even) of gates for the SDKI model
    gates = ITensor[]
    even_layer = build_a_layer_of_gates(2, N-2, N, h, Δτ, s)
    for tmp₁ in even_layer
        push!(gates, tmp₁)
    end

    odd_layer = build_a_layer_of_gates(1, N-1, N, h, Δτ, s)
    for tmp₂ in odd_layer
        push!(gates, tmp₂)
    end

    # Construct a layer of kicked gates
    kick_gates = build_kick_gates(s, 1, N)

    
    # Initialize the wavefunction using a Neel state
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = MPS(s, states)
    ψ_copy = deepcopy(ψ)
    # ψ_overlap = Complex{Float64}[]

    # # Intialize the wvaefunction as a random MPS   
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # Random.seed!(87900)
    # ψ = randomMPS(s, states, linkdims = 2)
    # # @show eltype(ψ), eltype(ψ[1])
    # # @show maxlinkdim(ψ)
    # ψ_copy = deepcopy(ψ)
    # ψ_overlap = Complex{Float64}[]

    timeSlices = Int(ttotal / Δτ) + 1; println("Total number of time slices that need to be saved is : $(timeSlices)")
    # Local observables including various one-point functions
    Sx = Array{ComplexF64}(undef, timeSlices, N)
    Sy = Array{ComplexF64}(undef, timeSlices, N)
    Sz = Array{ComplexF64}(undef, timeSlices, N)
    
    # Equal-time observables including various two-point functions
    Cxx = Array{ComplexF64}(undef, timeSlices, N * N)
    Cyy = Array{ComplexF64}(undef, timeSlices, N * N)
    Czz = Array{ComplexF64}(undef, timeSlices, N * N)
    SvN = Array{Float64}(undef, timeSlices, N - 1)

    timing = Float64[]

    # Take measurements of the initial wavefunction
    Sx[1, :] = expect(ψ_copy, "Sx"; sites = 1 : N)
    Sy[1, :] = expect(ψ_copy, "Sy"; sites = 1 : N)
    Sz[1, :] = expect(ψ_copy, "Sz"; sites = 1 : N)

    Cxx[1, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N)
    Cyy[1, :] = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N)
    Czz[1, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)
    append!(ψ_overlap, abs(inner(ψ, ψ_copy)))

    
    distance = Int(1.0 / Δτ); index = 2
    @time for time in 0.0 : Δτ : ttotal
        time ≈ ttotal && break
        SvN[index - 1, :] = entanglement_entropy(ψ_copy, N)
        
        
        #  
        tmp_t1 = Dates.now()
        
        # Apply the kicked gates at integer time
        if (abs((time / Δτ) % distance) < 1E-8)
            # println("")
            # println("Apply the kicked gates at integer time $time")
            # println("")
            ψ_copy = apply(kick_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
            # append!(ψ_overlap, abs(inner(ψ, ψ_copy)))
        end

        # Apply the Ising interaction and longitudinal gates
        ψ_copy = apply(gates, ψ_copy; cutoff)
        normalize!(ψ_copy)

        tmp_t2 = Dates.now()
        Δt = Dates.value(tmp_t2 - tmp_t1)
        push!(timing, Δt)
        @show timing
        
        # Local observables e.g. Sx, Sz
        Sx[index, :] = expect(ψ_copy, "Sx"; sites = 1 : N) 
        Sy[index, :] = expect(ψ_copy, "Sy"; sites = 1 : N)
        Sz[index, :] = expect(ψ_copy, "Sz"; sites = 1 : N)

        # Correlation functions e.g. Cxx, Czz
        Cxx[index, :] = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N)
        Cyy[index, :] = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N)
        Czz[index, :] = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N)
        index += 1

        # tmp_overlap = abs(inner(ψ, ψ_copy))
        # println("The inner product is: $tmp_overlap")
        # append!(ψ_overlap, tmp_overlap)
    end

    println("################################################################################")
    println("################################################################################")
    println("Projective measurements of the initial MPS in the Sz basis")
    @show Sz[1, :]
    println("################################################################################")
    println("################################################################################")

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
    write(file, "Time Sequence", timing)
    write(file, "Initial Sx", Sx[1, :])
    write(file, "Initial Sy", Sy[1, :])
    write(file, "Initial Sz", Sz[1, :])
    close(file)
    
    return
end  