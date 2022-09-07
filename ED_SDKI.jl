## Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model
using ITensors: orthocenter, sites
using ITensors

ITensors.disable_warn_order()
let 
    N = 8
    cutoff = 1E-8
    τ = 5
    iterationLimit = 10
    h = 5                                            # an integrability-breaking longitudinal field h 
    
    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);         # s = siteinds("S=1/2", N; conserve_qns = true)

    # Define the Ising model Hamiltonian with longitudinal field 
    ampoA = OpSum()
    for ind in 1 : (N - 1)
        ampoA += π/4, "Sz", ind, "Sz", ind + 1
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
    expHamiltonian₁ = exp(-τ * Hamiltonian₁)
    expHamiltonian₂ = exp(-τ * Hamiltonian₂)
    
    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

    # Locate the central site
    centralSite = div(N, 2)

    # Time evovle the initial random wavefunction to obtain the ground state
    # Need to be updated for the circuit setup
    tmpPsi = ψ
    @time for ind in 1:iterationLimit
        tmpPsi = apply(expHamiltonian₂, tmpPsi; cutoff)
        normalize!(tmpPsi)
        tmpSz = expect(tmpPsi, "Sz")
        println("At each projection step, Sz is $tmpSz")
    end

    Sz = expect(ψ, "Sz"; sites = centralSite)
    Sz_prime = expect(tmpPsi, "Sz"; sites = centralSite)
    println("Using initial random MPS, Sz is $Sz")
    println("Using the projected MPS, Sz is $Sz_prime")

    # Compute <Sz> at rach time step and apply gates to go to the next step
    # @time for time in 0.0:tau:ttotal
    #     Sz = expect(ψ, "Sz"; sites = centralSite)
    #     Czz = correlation_matrix(ψ, "Sz", "Sz"; sites = centralSite : centralSite + 1)
    #     println("At time step $time, Sz is $Sz")
    #     println("At time step $time, Czz is $Czz")

    #     time ≈ ttotal && break
    #     if (abs(time / tau % 10) < 1E-8 || abs((time + tau)/tau % 10) < 1E-8)
    #         println("At time $(time/tau), applying the kicked fields")
    #         ψ = apply(kickGates, ψ; cutoff)
    #     else
    #         ψ = apply(gates, ψ; cutoff)
    #     end
    #     normalize!(ψ)
    # end
    
    return
end 