using Base: Integer
## 07/03/2023
## Initialize the wavefunction @t=0

using ITensors
using ITensors.HDF5
using Random



# Use a random MPS as the initial wavefunction
# For the SDKI model, the total quantum number is not conserved

function random_initialization(site_type :: String, site_length :: Integer)
    tmp_sites = siteinds(site_type, site_length; conserve_qns = false)
    tmp_states = [isodd(n) ? "Up" : "Dn" for n = 1 : site_length]
    Random.seed!(87900) 
    ψ₀ = randomMPS(tmp_sites, tmp_states, linkdims = 2)
    # @show maxlinkdim(ψ)
    return ψ₀
end