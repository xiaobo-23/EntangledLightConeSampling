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
    h = 0.5                                            # an integrability-breaking longitudinal field h 
    
    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);    # s = siteinds("S=1/2", N; conserve_qns = true)

    # Define the Ising model Hamiltonian with longitudinal field 
    ampoA = OpSum()
    for ind in 1 : (N - 1)
        ampoA += π, "Sz", ind, "Sz", ind + 1
        # ampoA += 2 * h, "Sz", ind
    end
    # ampoA += 2 * h, "Sz", N
    for ind in 1 : N
        ampoA += 2 * h, "Sz", ind
    end
    H₁ = MPO(ampoA, s)

    # ampoA = OpSum()
    # for ind in 1 : (N - 1)
    #     if (ind - 1 < 1E-8)
    #         tmp1 = 2
    #         tmp2 = 1
    #     elseif (abs(ind - (N - 1)) < 1E-8)
    #         tmp1 = 1
    #         tmp2 = 2
    #     else
    #         tmp1 = 1
    #         tmp2 = 1
    #     end
    #     ampoA += π, "Sz", ind, "Sz", ind + 1
    #     ampoA += tmp1 * h, "Sz", ind
    #     ampoA += tmp2 * h, "Sz", ind + 1
    # end
    # H₁ = MPO(ampoA, s)

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
    # centralSite = div(N, 2)
    ψ_copy = copy(ψ)

    # Compute overlaps between the original and time evolved wavefunctions
    ψ_overlap = Complex{Float64}[]

    # Compute local observables
    Sx = zeros(iterationLimit, N);  Sx = complex(Sx)
    Sy = zeros(iterationLimit, N);  Sy = complex(Sy)
    Sz = zeros(iterationLimit, N);  Sz = complex(Sz)

    # Compute spin correlation functions
    Cxx = zeros(iterationLimit, N); Cxx = complex(Cxx)
    Cyy = zeros(iterationLimit, N); Cyy = complex(Cyy)
    Czz = zeros(iterationLimit, N); Czz = complex(Czz)
    index = 1

    @time for ind in 1:iterationLimit
        # Compute overlap of wavefunction < Psi(t) | Psi(0) > 
        # tmpOverlap = abs(inner(ψ, ψ_copy))
        # append!(ψ_overlap, tmpOverlap)
        # println("At projection step $(ind - 1), overlap of wavefunciton is $tmpOverlap") 

        if (ind - 1) < 1E-8
            tmpOverlap = abs(inner(ψ, ψ_copy))
            append!(ψ_overlap, tmpOverlap)
            println("At projection step $(ind - 1), overlap of wavefunciton is $tmpOverlap") 
        end
    
        # Compute local observables e.g. Sz
        tmpSx = expect(ψ_copy, "Sx"); Sx[index, :] = tmpSx; @show size(tmpSx)
        tmpSy = expect(ψ_copy, "Sy"); Sy[index, :] = tmpSy; @show size(tmpSy)
        tmpSz = expect(ψ_copy, "Sz"); Sz[index, :] = tmpSz; @show size(tmpSz)

        # Compute spin correlation functions e.g. Czz
        tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx", sites = 1 : N); Cxx[index, :] = tmpCxx[4, :]; @show size(tmpCxx')
        tmpCyy = correlation_matrix(ψ_copy, "Sy", "Sy", sites = 1 : N); Cyy[index, :] = tmpCyy[4, :]; @show size(tmpCyy')
        tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz", sites = 1 : N); Czz[index, :] = tmpCzz[4, :]; @show size(tmpCzz')
        # Vectorize the correlation matrix to store all information
        # Czz[index, :] = vec(tmpCzz')
        index += 1

        # Apply the kicked transverse field gate
        ψ_copy = apply(expHamiltonian₂, ψ_copy; cutoff)
        normalize!(ψ_copy)
        
        # tmpOverlap = abs(inner(ψ, ψ_copy))
        # println("")
        # println("Apply the kicked gates at integer time $ind")
        # println("Overlap of wavefunctions are: $tmpOverlap")
        # println("")
       
        # Apply the Ising interaction plus longitudinal field gate using a smaller time step
        for tmpInd in 1 : timeSlice
            # Compute the overlap of wavefunctions < Psi(t) | Psi(0) >
            tmpOverlap = abs(inner(ψ, ψ_copy))
            append!(ψ_overlap, tmpOverlap)
            # @show size(ψ_overlap)

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

    @show size(ψ_overlap)
    @show size(Czz)
    # Save measurements into a hdf5 file
    file = h5open("RawData/ED_N$(N)_h$(h)_Info.h5", "w")
    write(file, "Sx", Sx)       # Sx
    write(file, "Sy", Sy)       # Sy
    write(file, "Sz", Sz)       # Sz
    write(file, "Cxx", Cxx)     # Cxx
    write(file, "Cyy", Cyy)     # Cyy   
    write(file, "Czz", Czz)     # Czz
    write(file, "Wavefunction Overlap", ψ_overlap);  @show size(ψ_overlap)
    close(file)
    
    return
end 