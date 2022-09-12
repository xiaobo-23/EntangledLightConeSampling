## Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites

ITensors.disable_warn_order()
let 
    N = 8
    cutoff = 1E-8
    τ = 1
    iterationLimit = 10
    h = 5                                                   # an integrability-breaking longitudinal field h 
    
    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);         # s = siteinds("S=1/2", N; conserve_qns = true)

    # Define the Ising model Hamiltonian with longitudinal field 
    ampoA = OpSum()
    for ind in 1 : (N - 1)
        ampoA += π, "Sz", ind, "Sz", ind + 1
        ampoA += 2 * h, "Sz", ind
    end
    ampoA += 2 * h, "Sz", N
    H₁ = MPO(ampoA, s)

    # Define the Hamiltonian with transverse field
    ampoB = OpSum()
    for ind in 1 : N
        ampoB += π/2, "Sx", ind
    end
    H₂ = MPO(ampoB, s)

    # Make a single ITensor out of the MPO
    Hamiltonian₁ = H₁[1]
    Hamiltonian₂ = H₂[1]
    for i in 2 : N
        Hamiltonian₁ *= H₁[i]
        Hamiltonian₂ *= H₂[i]
    end

    # Exponentiate the Ising plus longitudinal field Hamiltonian
    expHamiltonian₁ = exp(-1.0im * τ * Hamiltonian₁)
    expHamiltonian₂ = exp(-1.0im * Hamiltonian₂)
    
    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # @show maxlinkdim(ψ)

    # Locate the central site
    centralSite = div(N, 2)

    # Time evovle the initial random wavefunction to obtain the ground state
    # Need to be updated for the circuit setup
    ψ_copy = copy(ψ)
    Sz = Complex{Float64}[]
    ψ_overlap = Complex{Float64}[]
    Czz = zeros(51,  N); Czz = complex(Czz)
    index = 1

    @time for ind in 1:iterationLimit
        tmpOverlap = abs(inner(ψ, ψ_copy))
        println("At projection step $(ind - 1), overlap of wavefunciton is $tmpOverlap")
        append!(ψ_overlap, tmpOverlap)

        # ψ_copy = apply(expHamiltonian₂, ψ_copy; cutoff)
        # normalize!(ψ_copy)
        ψ_copy = apply(expHamiltonian₁, ψ_copy; cutoff)
        normalize!(ψ_copy)
        tmpSz = expect(ψ_copy, "Sz")
        @show size(tmpSz)
        println("At projection step $ind, Sz is $tmpSz")
        append!(Sz, tmpSz)

        tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz", sites = 1 : N)
        @show size(tmpCzz)
        println("At projection step $ind, Czz is $(tmpCzz[1, :])")
        Czz[index, :] = tmpCzz[1, :]
        index += 1

        if ind == iterationLimit
            tmpOverlap = abs(inner(ψ, ψ_copy))
            println("At projection step $ind, overlap of wavefunciton is $tmpOverlap")
            append!(ψ_overlap, tmpOverlap)
        end
    end

     # Store data into a hdf5 file
     file = h5open("ED_N$(N)_Info.h5", "w")
     write(file, "Sz", Sz)
     write(file, "Czz", Czz)
     write(file, "Wavefunction Overlap", ψ_overlap)
     close(file)

    # Sz = expect(ψ, "Sz"; sites = centralSite)
    # Sz_prime = expect(ψ_copy, "Sz"; sites = centralSite)
    # println("Using initial random MPS, Sz is $Sz")
    # println("Using the projected MPS, Sz is $Sz_prime")
    
    return
end 