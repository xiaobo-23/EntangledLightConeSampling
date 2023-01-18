## Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites
using Random
ITensors.disable_warn_order()
let 
    N = 8
    cutoff = 1E-8
    τ = 0.1; timeSlice = Int(1 / τ)
    iterationLimit = 12
    h = 0.2                                           # an integrability-breaking longitudinal field h 
    
    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);     # s = siteinds("S=1/2", N; conserve_qns = true)

    # Define the Ising model Hamiltonian with longitudinal field 
    ampoA = OpSum()
    # for ind in 1 : (N - 1)
    #     ampoA += π, "Sz", ind, "Sz", ind + 1
    #     # ampoA += 2 * h, "Sz", ind
    # end
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
    
    # # Initialize the wavefunction as a Neel state
    # ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # ψ = MPS(s, states)
    # ψ_copy = deepcopy(ψ)
    # ψ_overlap = Complex{Float64}[]
    
    # Initializa the wavefunction as a random MPS
    Random.seed!(200)
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = randomMPS(s, states, linkdims = 2); # show maxlinkdim(ψ)
    # ψ = randomMPS(s, linkdims = 2)
    ψ_copy = deepcopy(ψ)
    ψ_overlap = Complex{Float64}[]
    
    # Store the initial wvaefunction in a HDF5 file
    # wavefunction_file = h5open("random_MPS.h5", "w")
    # write(wavefunction_file, "Psi", ψ)
    # close(wavefunction_file)

    # Take a measurement of the initial random MPS to make sure the same random MPS is used through all codes
    Sz₀ = expect(ψ_copy, "Sz"; sites = 1: N)

    # Compute local observables
    Sx = complex(zeros(iterationLimit, N))
    Sy = complex(zeros(iterationLimit, N))
    Sz = complex(zeros(iterationLimit, N))

    # Compute spin correlation functions
    Cxx = complex(zeros(iterationLimit, N * N))
    Cyy = complex(zeros(iterationLimit, N * N))
    Czz = complex(zeros(iterationLimit, N * N))

    # Compute the overlap of wavefunctions before starting real-time evolution
    append!(ψ_overlap, abs(inner(ψ, ψ_copy)))
    println("")
    println("At time T=0, the overlap of wavefunctions is $(ψ_overlap[1])")
    println("")
    
    @time for ind in 1:iterationLimit
        # Compute local observables e.g. Sz
        tmpSx = expect(ψ_copy, "Sx"); Sx[ind, :] = tmpSx; # @show (size(tmpSx), tmpSx)
        tmpSy = expect(ψ_copy, "Sy"); Sy[ind, :] = tmpSy; # @show (size(tmpSy), tmpSy)
        tmpSz = expect(ψ_copy, "Sz"); Sz[ind, :] = tmpSz; # @show (size(tmpSz), tmpSz)

        # Compute spin correlation functions e.g. Czz
        tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx", sites = 1 : N); Cxx[ind, :] = tmpCxx[:]; # @show size(tmpCxx')
        tmpCyy = correlation_matrix(ψ_copy, "Sy", "Sy", sites = 1 : N); Cyy[ind, :] = tmpCyy[:]; # @show size(tmpCyy')
        tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz", sites = 1 : N); Czz[ind, :] = tmpCzz[:]; # @show size(tmpCzz')
        
        # Vectorize the correlation matrix to store all information
        # Czz[index, :] = vec(tmpCzz')

        # Apply the kicked transverse field gate
        ψ_copy = apply(expHamiltonian₂, ψ_copy; cutoff)
        normalize!(ψ_copy)
        append!(ψ_overlap, abs(inner(ψ, ψ_copy)))      
        # tmpSx = expect(ψ_copy, "Sx"); @show tmpSx
        # tmpSy = expect(ψ_copy, "Sy"); @show tmpSy
        # tmpSz = expect(ψ_copy, "Sz"); @show tmpSz
       
        # Apply the Ising interaction plus longitudinal field gate using a smaller time step
        for tmpInd in 1 : timeSlice
            ψ_copy = apply(expHamiltonian₁, ψ_copy; cutoff)
            normalize!(ψ_copy)
            
            # Compute the overlap of wavefunctions < Psi(t) | Psi(0) >
            append!(ψ_overlap, abs(inner(ψ, ψ_copy)))
        end
        
        # println("")
        # println("")
        # tmpSx = expect(ψ_copy, "Sx"); @show tmpSx
        # tmpSy = expect(ψ_copy, "Sy"); @show tmpSy
        # tmpSz = expect(ψ_copy, "Sz"); @show tmpSz
        # println("")
        # println("")
    end

    # @show size(ψ_overlap)
    # @show size(Czz)

    println("################################################################################")
    println("################################################################################")
    println("Information of the initial random MPS")
    @show Sz₀
    println("################################################################################")
    println("################################################################################")
    
    # Save measurements into a hdf5 file
    file = h5open("Data/ED_N$(N)_h$(h)_Iteration$(iterationLimit)_Rotations_Only_Random_TESTING.h5", "w")
    write(file, "Sx", Sx)       # Sx
    write(file, "Sy", Sy)       # Sy
    write(file, "Sz", Sz)       # Sz
    write(file, "Cxx", Cxx)     # Cxx
    write(file, "Cyy", Cyy)     # Cyy   
    write(file, "Czz", Czz)     # Czz
    write(file, "Wavefunction Overlap", ψ_overlap);  @show size(ψ_overlap)
    write(file, "Initial Sz", Sz₀)
    close(file)
    
    return
end 