# 5/15/2025
# Implement the idea of density matrix rotation sampling

using ITensors
using ITensorMPS
using Random
using Statistics
using LinearAlgebra
# using TimerOutput

# include("Sample.jl")
include("Projection.jl")


# Sample a two-site MPS to compute Sx, Sy or Sz
function sample(m :: MPS, j :: Int, observable :: AbstractString)
    mpsLength = length(m)

    # Move the orthogonality center of the MPS to site j
    orthogonalize!(m, j)
    if orthocenter(m) != j
        error("sample: MPS m must have orthocenter(m) == j")
    end
    
    # Check the normalization of the MPS
    if abs(1.0 - norm(m[j])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[j]))")
    end

    if observable == "Sx"
        tmp_projn = Sx_projn
        projn_up = Sx_projn_plus
        projn_dn = Sx_projn_minus
    elseif observable == "Sy"
        tmp_projn = Sy_projn
        projn_up = Sy_projn_plus
        projn_dn = Sy_projn_minus
    elseif observable == "Sz"
        tmp_projn = Sz_projn
        projn_up = Sz_projn_up
        projn_dn = Sz_projn_dn
    else
        error("sample: Measurement type doesn't exist")
    end

    # Sample the target observables
    result = zeros(Int, 2)
    A = m[j]
    
    for ind in j:j+1
        tmpS = siteind(m, ind)
        d = dim(tmpS)
        pdisc = 0.0
        r = rand()

        n = 1 
        An = ITensor()
        pn = 0.0

        while n <= d
            projn = ITensor(tmpS)
            # projn[tmpS => n] = 1.0
            projn[tmpS => 1] = tmp_projn[n][1]
            projn[tmpS => 2] = tmp_projn[n][2]
        
            # @show typeof(projn), typeof(An), typeof(A)
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind - j + 1] = n

        if ind < mpsLength
            A = m[ind + 1] * An
            A *= (1. / sqrt(pn))
        end

        # Collapse the site based on the measurements 
        if n == 1
            tmp_reset = ITensor(projn_up, tmpS', tmpS)
        else
            tmp_reset = ITensor(projn_dn, tmpS', tmpS)
        end

        m[ind] *= tmp_reset
        noprime!(m[ind])
    end
    return result
end


function compute_probability(input_vector::Vector{<:Dict{String, Any}}, keyword::AbstractString)::Float64
    return sum(state[keyword] for state in input_vector)
end


let
    # Initialize the system and set up parameters
    N = 128                
    J = 1                  
    h = 0.45                                  
    Nₛ_density = 16                            # Number of states sampled from density matrix 
    Nₛ = 100                                   # Number of bitstrings sampled according to the Born rule                  
   
    projected_states = Vector{Dict{String, Any}}()
    density_Sx = zeros(Float64, N)
    density_Sz = zeros(Float64, N)
    Prob = zeros(Float64, N)
    eigenvalues = zeros(Float64, N, 2 * Nₛ_density)
    
    
    # Set up the local observables in the Sz basis  
    Sx_matrix = 0.5 * [0 1; 1 0]
    Sy_matrix = 0.5 * [0.0 -1.0im; 1.0im 0.0]
    Sz_matrix = 0.5 * Diagonal([1, -1])

    
    # Read in the wavefunction from the file and start the sampling process
    println(" ")
    println("*************************************************************************************")
    println("Read in the wavefunction from the file and start the sampling process.")
    println("*************************************************************************************")
    println(" ")

    Δ = 1.5 * J
    file = h5open("data/XXZ_N128_Delta1.5_Psi.h5", "r")
    ψ = read(file, "Psi", MPS)
    Sz = expect(ψ, "Sz"; sites = 1 : N)
    Sx = expect(ψ, "Sx"; sites = 1 : N)
    Czz = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    @show linkdims(ψ)

    # # Set up the initial wavefunction as a product state or a random MPS
    # sites = siteinds("S=1/2", N; conserve_qns = false) 
    # # # Initialize the state of the system as a Neel state
    # state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # # Initialize the state of the system in the + state
    # # state = fill("+", N)
    # # @show state
    # # ψ = MPS(sites, state)

    # # Alternatively, initialize the state of the system as a random MPS
    # # ψ = random_mps(sites, state; linkdims = 8)
    
    # # ************************************************************************************
    # # Set up the transverse field Ising Hamiltonian
    # # ************************************************************************************
    # os = OpSum()
    # for j = 1 : N - 1
    #     os += J, "Sz", j, "Sz", j + 1
    #     os += -h, "Sx", j
    # end
    # os += -h, "Sx", N
    # H = MPO(os, sites)
    # ψ₀ = randomMPS(sites, state, linkdims = 2)
    # # ************************************************************************************
    # # ************************************************************************************


    # # #************************************************************************************
    # # # Set up the XXZ Heisenberg Hamiltonian
    # # #************************************************************************************
    # # Δ = 2 * J
    # # os = OpSum()
    # # for j = 1 : N - 1
    # #     os += 0.5J, "S+", j, "S-", j + 1
    # #     os += 0.5J, "S-", j, "S+", j + 1
    # #     os += Δ, "Sz", j, "Sz", j + 1
    # # end
    # # H = MPO(os, sites)
    # # ψ₀ = randomMPS(sites, state, linkdims = 2)
    # # #************************************************************************************
    # # #************************************************************************************

    
    # # Obtain the ground-state wavefunction using DMRG
    # cutoff = [1E-10]
    # nsweeps = 15
    # maxdim = [10,20,100,100,200]
    # energy, ψ = dmrg(H, ψ₀; nsweeps,maxdim,cutoff)
    
    # Sx = expect(ψ, "Sx"; sites = 1 : N)
    # Sz = expect(ψ, "Sz"; sites = 1 : N)
    # @show Sx, Sz
    # @show linkdims(ψ)
    
    # #***********************************************************************************
    # # Generate bistrings in the Sx basis according to the Born rule
    # #***********************************************************************************
    # bitstring_Sx = zeros(Float64, Nₛ, N)
    # bitstring_Sz = zeros(Float64, Nₛ, N)

    # for i = 1 : Nₛ
    #     ψ_copy = deepcopy(ψ)
    #     println("Generate bitstring #$(i)")
    #     println("")
    #     for j = 1 : 2 : N
    #         tmp = sample(ψ_copy, j, "Sx")
    #         normalize!(ψ_copy)
    #         bitstring_Sx[i, j: j + 1] = tmp
    #     end

    #     ψ_copy = deepcopy(ψ)
    #     println("Generate bitstring #$(i)")
    #     println("")
    #     for j = 1 : 2 : N
    #         tmp = sample(ψ_copy, j, "Sz")
    #         normalize!(ψ_copy)
    #         bitstring_Sz[i, j: j + 1] = tmp
    #     end
    # end
    # bitstring_Sx .= ifelse.(bitstring_Sx .== 1, 0.5, -0.5)
    # bitstring_Sz .= ifelse.(bitstring_Sz .== 1, 0.5, -0.5)
    # # @show bitstring_Sx, bitstring_Sz

    
    # # # Compute the expectation value and standard deviation of one-point functions based on the samples 
    # # sample_ave_Sx = mean(bitstring_Sx, dims = 1)
    # # sample_std_Sx = std(bitstring_Sx, corrected = true, dims = 1) / sqrt(Nₛ)
    # # sample_ave_Sz = mean(bitstring_Sz, dims = 1)
    # # sample_std_Sz = std(bitstring_Sz, corrected = true, dims = 1) / sqrt(Nₛ)
    # # @show sample_ave_Sx, sample_ave_Sz

    #************************************************************************************
    # Sample the wavefunction using the density matrix rotation algorithm
    #************************************************************************************
    
    # Put the orthogonaality center of the MPS at site 1
    orthogonalize!(ψ, 1)
    if orthocenter(ψ) != 1
        error("sample: MPS must have orthocenter(psi) == 1")
    end
    # @show expect(ψ, "Sz"; sites = 1 : 1)

    # Compute the 1-body reduced density Matrix
    psidag = dag(ψ)
    prime!(psidag[1])

    bond_index = commonind(ψ[1], ψ[2])
    prime!(ψ[1], bond_index)
    
    # Compute the 1-body reduced density matrix
    rho = ψ[1] * psidag[1]
    # @show rho
    # @show typeof(rho), size(rho)

    # Diagonalize the reduced density matrix and obtain the eigenvalues and eigenvectors    
    # @show inds(rho)[1], inds(rho)[2]
    matrix = Matrix(rho, inds(rho)[1], inds(rho)[2])
    vals, vecs = eigen(matrix)
    # @show matrix
    @show vals, vecs
    
    # Compute the expectation values of Sx and Sz based on the density matrix
    density_Sx[1] = tr(matrix * Sx_matrix)
    density_Sz[1] = tr(matrix * Sz_matrix)
    Prob[1] = vals[1] + vals[2]
    if abs(Sz[1]) > 1e-8 && !(density_Sz[1] ≈ Sz[1])
        error("density_Sz[1] ≉ Sz[1]: $(density_Sz[1]) vs $(Sz[1])")
    end
    if !(matrix * vecs[:, 1] ≈ vals[1] * vecs[:, 1]) || !(matrix * vecs[:, 2] ≈ vals[2] * vecs[:, 2])
        error("errors in computing the eigenvalues and eigenvectors of the density matrix")
    end
    
    # Remove prime indices from the first tensor of the MPS
    noprime!(ψ[1])
    site_index = siteind(ψ, 1)


    for j in 1 : 2
        # Construct the projector onto eigenvector of the density matrix
        projection_matrix = vecs[:, j] * vecs[:, j]'; @show projection_matrix
        projection = ITensor(projection_matrix, site_index', site_index)
        tmp = deepcopy(ψ)
        tmp[1] *= projection
        noprime!(tmp[1])
        tmp[1] /= norm(tmp[1])
        @show norm(tmp)
        # @show tmp 

        orthogonalize!(tmp, 2)
        if orthocenter(tmp) != 2
            error("sample: MPS must have orthocenter(psi) == 2")
        end

        # Store the eigenvalue and projected state in a dictionary
        push!(projected_states, Dict(
            "eigenvalue" => vals[j],
            "wavefunction" => tmp
        ))
    end
    
    
    # Sort the samples based on its weights
    projected_states = sort(projected_states, by = x -> x["eigenvalue"], rev = true)

    for (eigen_index, state) in enumerate(projected_states)
        eigenvalues[1, eigen_index] = state["eigenvalue"]
    end
    @show eigenvalues[1, :]

    
    # Sample the wavefunction from site 2 to N based on the density matrix
    for idx in 2 : N
        N_states = min(length(projected_states), Nₛ_density)
        
        for i in 1 : N_states
            tmp = popfirst!(projected_states)
            psi = tmp["wavefunction"]
            
            psidag = dag(psi)
            prime!(psidag[idx])
            prime!(psi[idx], commonind(psi[idx - 1], psi[idx]))
            if idx < N
                prime!(psi[idx], commonind(psi[idx], psi[idx + 1]))
            end

            # Compute the 1-body reduced density matrix
            rho = psi[idx] * psidag[idx]
            # @show rho
            matrix = Matrix(rho, inds(rho)[1], inds(rho)[2])
            vals, vecs = eigen(matrix)

            density_Sx[idx] += tr(matrix * Sx_matrix) * tmp["eigenvalue"]
            density_Sz[idx] += tr(matrix * Sz_matrix) * tmp["eigenvalue"]
            Prob[idx] += (vals[1] + vals[2]) * tmp["eigenvalue"]

            noprime!(psi[idx])
            site_index = siteind(psi, idx)
            
            for j in 1 : 2
                # Construct the projector onto the eigenvector of the density matrix
                projection_matrix = vecs[:, j] * vecs[:, j]' 
                projection = ITensor(projection_matrix, site_index', site_index)
                psi_copy = deepcopy(psi)
                psi_copy[idx] *= projection
                noprime!(psi_copy[idx])
                psi_copy[idx] /= norm(psi_copy[idx])
                @show norm(psi_copy)
                # @show psi_copy
                
                if idx < N
                    orthogonalize!(psi_copy, idx + 1)
                    if orthocenter(psi_copy) != idx + 1
                        error("error in indexing the orthocenter of the MPS")
                    end
                end

                # Store the eigenvalue and projected state in a dictionary
                push!(projected_states, Dict(
                    "eigenvalue" => vals[j] * tmp["eigenvalue"],
                    "wavefunction" => psi_copy
                ))
            end
        end

        # Normalize the expectation values of physical observables
        density_Sx[idx] /= Prob[idx]
        density_Sz[idx] /= Prob[idx]

        # Sort the samples based on its weights
        projected_states = sort(projected_states, by = x -> x["eigenvalue"], rev = true)
        for (eigen_index, state) in enumerate(projected_states)
            eigenvalues[idx, eigen_index] = state["eigenvalue"]
        end
        
        if length(projected_states) > Nₛ_density
            @info "Truncating projected_states from $(length(projected_states)) to $Nₛ_density"
            projected_states = projected_states[1:Nₛ_density]
        else
            @info "Current projected_states length: $(length(projected_states))"
        end

        # println(" ")
        # @show "Number of projected states: ", length(projected_states)
        # for state in projected_states
        #     @show state["eigenvalue"]
        # end
        # println(" ")
    end


    @show linkdims(ψ)
    @show Prob 
    @show density_Sz[1 : 20]
    @show Sz[1 : 20]
    

    # #*********************************************************************************************************
    # # Save results into a HDF5 file
    # #*********************************************************************************************************
    # h5open("data/Transverse_Field_Ising_N$(N)_h$(h)_Sample$(Nₛ_density)_update.h5", "w") do file
    #     write(file, "Sx", Sx)
    #     write(file, "Sz", Sz)
    #     # write(file, "Bitstring Sx", bitstring_Sx)
    #     # write(file, "Bitstring Sz", bitstring_Sz)
    #     write(file, "Density Sx", density_Sx)
    #     write(file, "Density Sz", density_Sz)
    #     write(file, "Probability", Prob)
    #     write(file, "Spectrum", eigenvalues)
    # end

    #*********************************************************************************************************
    # Save results into a HDF5 file
    #*********************************************************************************************************
    h5open("data/XXZ_N$(N)_Delta$(Δ)_Sample$(Nₛ_density)_update.h5", "w") do file
        write(file, "Sx", Sx)
        write(file, "Sz", Sz)
        # write(file, "Bitstring Sx", bitstring_Sx)
        # write(file, "Bitstring Sz", bitstring_Sz)
        write(file, "Density Sx", density_Sx)
        write(file, "Density Sz", density_Sz)
        write(file, "Probability", Prob)
        write(file, "Spectrum", eigenvalues)
    end

    return
end