## Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites

ITensors.disable_warn_order()
let 
    N = 8
    cutoff = 1E-8
    τ = 0.1; timeSlice = Int(1 / τ)
    iterationLimit = 10
    h = 0.2                                                 # an integrability-breaking longitudinal field h 
    
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
    ψ_copy = copy(ψ)
    # Sz = Complex{Float64}[]
    ψ_overlap = Complex{Float64}[]
    Sz = zeros(iterationLimit, N); Czz = complex(Sz)
    Czz = zeros(iterationLimit, N); Czz = complex(Czz)
    index = 1

    @time for ind in 1:iterationLimit
        # Compute overlap of wavefunction < Psi(t) | Psi(0) > 
        if (ind - 1) < 1E-8
            tmpOverlap = abs(inner(ψ, ψ_copy))
            append!(ψ_overlap, tmpOverlap)
            println("At projection step $(ind - 1), overlap of wavefunciton is $tmpOverlap") 
        end
    
        # Compute local observables e.g. Sz
        tmpSz = expect(ψ_copy, "Sz")
        @show size(tmpSz)
        # println("At projection step $ind, Sz is $tmpSz")
        # append!(Sz, tmpSz)
        Sz[index, :] = tmpSz

        # Compute spin correlation function Czz
        tmpCzz = correlation_matrix(ψ_copy, "Sx", "Sx", sites = 1 : N)
        @show size(tmpCzz')
        # println("At projection step $ind, Czz is $(tmpCzz)")
        Czz[index, :] = tmpCzz[1, :]
        # Czz[index, :] = vec(tmpCzz')
        index += 1

        # Apply the kicked transverse field gate
        ψ_copy = apply(expHamiltonian₂, ψ_copy; cutoff)
        normalize!(ψ_copy)
       
        # Apply the Ising + longitudinal field gate
        for tmpInd in 1 : timeSlice
            # Compute the overlap of wavefunctions < Psi(t) | Psi(0) >
            tmpOverlap = abs(inner(ψ, ψ_copy))
            append!(ψ_overlap, tmpOverlap)
            @show size(ψ_overlap)

            ψ_copy = apply(expHamiltonian₁, ψ_copy; cutoff)
            normalize!(ψ_copy)
        end

        # Compute overlap of wavefunction < Psi(t) | Psi(0) >
        # if ind == iterationLimit
        #     tmpOverlap = abs(inner(ψ, ψ_copy))
        #     println("At projection step $ind, overlap of wavefunciton is $tmpOverlap")
        #     append!(ψ_overlap, tmpOverlap)
        # end
    end

    # Store data into a hdf5 file
    file = h5open("ED_N$(N)_h$(h)_Info.h5", "w")
    write(file, "Sz", Sz)
    write(file, "Czz", Czz)
    @show size(ψ_overlap)
    write(file, "Wavefunction Overlap", ψ_overlap)
    close(file)
    
    return
end 