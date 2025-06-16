# 10/21/2024
# Obtain the ground state of paradigmatic models such as the transverse field Ising model and the Heisenberg model 
# Prepare the wavefunction to test the sampling methods 

using ITensors, ITensorMPS

let 
    # Initialize the random MPS
    N = 64                  # Number of physical sites
    sites = siteinds("S=1/2", N; conserve_qns = false) 
    state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]


    # # Set up the Heisenberg model on a one-dimensional (1D) chain
    # os = OpSum()
    # for j = 1 : N  - 1
    #     os += "Sz",j,"Sz",j+1
    #     os += 1/2,"S+",j,"S-",j+1
    #     os += 1/2,"S-",j,"S+",j+1
    # end
    # H = MPO(os, sites)
    # ψ₀ = randomMPS(sites, state, linkdims = 2)

    # Set up the transverse field Ising model on a 1D chain
    h = 1.1
    os = OpSum()
    for j = 1 : N - 1
        os += "Sz", j, "Sz", j + 1
        os += h, "Sx", j
    end
    os += h, "Sx", N 
    H = MPO(os, sites)
    ψ₀ = randomMPS(sites, state, linkdims = 2)

    
    # Set up the parameters for the DMRG simulation
    cutoff = [1E-10]
    nsweeps = 10
    maxdim = [10,20,100,100,200]
    
    energy, ψ = dmrg(H, ψ₀; nsweeps,maxdim,cutoff)
    Sz = expect(ψ, "Sz"; sites = 1 : N)
    Sx = expect(ψ, "Sx"; sites = 1 : N)
    Czz = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    
    @show Sx
    @show Sz
    @show linkdims(ψ)
    # @show Czz

    # @show typeof(ψ)
    # # Save results to a file
    h5open("data/Transverse_Ising_N$(N)_h$(h)_Psi.h5", "w") do file
        write(file, "Psi", ψ)
        write(file, "Sx", Sx)
        write(file, "Sz", Sz)
        write(file, "Czz", Czz)
    end

    return
end