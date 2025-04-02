# 10/21/2024
# Implement deterministic sampling for the ground-state of model Hamiltonians

using ITensors
using ITensorMPS
using Random
using Statistics
using HDF5
# using TimerOutput

let 
    
    # Initialize the random MPS
    N = 20                  # Number of physical sites
    sites = siteinds("S=1/2", N; conserve_qns = false) 
    state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]

    # # Set up the Heisenberg model Hamiltonian on a one-dimensional chain
    # os = OpSum()
    # for j = 1 : N  - 1
    #     os += "Sz",j,"Sz",j+1
    #     os += 1/2,"S+",j,"S-",j+1
    #     os += 1/2,"S-",j,"S+",j+1
    # end
    # H = MPO(os, sites)
    # ψ₀ = randomMPS(sites, state, linkdims = 2)

    # Set up the Ising model with longitudinal fields on a one-dimensional chain
    h = 1.2
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
    @show Czz

    @show typeof(ψ)
    # # Save results to a file
    # h5open("data/Transverse_Ising_N$(N)_h$(h)_Wavefunction.h5", "w") do file
    #     write(file, "Psi", ψ)
    #     write(file, "Sx", Sx)
    #     write(file, "Sz", Sz)
    #     write(file, "Czz", Czz)
    # end

end