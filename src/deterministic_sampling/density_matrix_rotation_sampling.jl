# 5/15/2025
# Implement the idea of density matrix rotation sampling

using ITensors
using ITensorMPS
using Random
using Statistics
using LinearAlgebra
# using HDF5
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
    N = 64                                    # Number of physical sites
    h = 1.2                                   # Strength of the transverse field
    Nₛ_density = 500                           # Number of states to be generated in the density matrix rotation sampling 
    Nₛ = 100                                   # Number of bitstrings to be generated according to the Born rule
    # prob_epsilon = 0.0002                   # Probability threshold for truncating the number of samples
   
    projected_states = Vector{Dict{String, Any}}()
    density_Sx = zeros(Float64, N)
    density_Sz = zeros(Float64, N)
    Prob = zeros(Float64, N)
    Prob_density = Vector{Vector{Float64}}()
    
    
    # Set up the local observables in the Sz basis  
    Sx_matrix = 0.5 * [0 1; 1 0]
    Sy_matrix = 0.5 * [0.0 -1.0im; 1.0im 0.0]
    Sz_matrix = 0.5 * Diagonal([1, -1])

    
    # # Read in the wavefunction from the file and start the sampling process
    # println(" ")
    # println("*************************************************************************************")
    # println("Read in the wavefunction from the file and start the sampling process.")
    # println("*************************************************************************************")
    # println(" ")
    
    # file = h5open("data/SDKI_TEBD_N20_h0.2_t5.0.h5", "r")
    # ψ = read(file, "Psi", MPS)
    # Sz = expect(ψ, "Sz"; sites = 1 : N)
    # Sx = expect(ψ, "Sx"; sites = 1 : N)
    # Czz = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)


    # Set up the initial wavefunction as a product state or a random MPS
    sites = siteinds("S=1/2", N; conserve_qns = false) 
    # # Initialize the state of the system as a Neel state
    state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # Initialize the state of the system in the + state
    # state = fill("+", N)
    # @show state
    # ψ = MPS(sites, state)

    # Alternatively, initialize the state of the system as a random MPS
    # ψ = random_mps(sites, state; linkdims = 8)
    
    # Set up the Hamiltonian as a MPO 
    os = OpSum()
    for j = 1 : N - 1
        os += "Sz", j, "Sz", j + 1
        os += h, "Sx", j
    end
    os += h, "Sx", N
    H = MPO(os, sites)
    ψ₀ = randomMPS(sites, state, linkdims = 2)

    cutoff = [1E-10]
    nsweeps = 10
    maxdim = [10,20,100,100,200]
    energy, ψ = dmrg(H, ψ₀; nsweeps,maxdim,cutoff)
    
    Sx = expect(ψ, "Sx"; sites = 1 : N)
    Sz = expect(ψ, "Sz"; sites = 1 : N)
    @show Sx, Sz
    @show linkdims(ψ)
    
    #***********************************************************************************
    # Generate bistrings in the Sx basis according to the Born rule
    #***********************************************************************************
    bitstring_Sx = zeros(Float64, Nₛ, N)
    bitstring_Sz = zeros(Float64, Nₛ, N)

    for i = 1 : Nₛ
        ψ_copy = deepcopy(ψ)
        println("Generate bitstring #$(i)")
        println("")
        for j = 1 : 2 : N
            tmp = sample(ψ_copy, j, "Sx")
            normalize!(ψ_copy)
            bitstring_Sx[i, j: j + 1] = tmp
        end

        ψ_copy = deepcopy(ψ)
        println("Generate bitstring #$(i)")
        println("")
        for j = 1 : 2 : N
            tmp = sample(ψ_copy, j, "Sz")
            normalize!(ψ_copy)
            bitstring_Sz[i, j: j + 1] = tmp
        end
    end
    bitstring_Sx .= ifelse.(bitstring_Sx .== 1, 0.5, -0.5)
    bitstring_Sz .= ifelse.(bitstring_Sz .== 1, 0.5, -0.5)
    @show bitstring_Sx, bitstring_Sz

    # Compute the expectation value and standard deviation of one-point functions based on the samples 
    sample_ave_Sx = mean(bitstring_Sx, dims = 1)
    sample_std_Sx = std(bitstring_Sx, corrected = true, dims = 1) / sqrt(Nₛ)
    sample_ave_Sz = mean(bitstring_Sz, dims = 1)
    sample_std_Sz = std(bitstring_Sz, corrected = true, dims = 1) / sqrt(Nₛ)
    @show sample_ave_Sx, sample_ave_Sz

    #************************************************************************************
    # Sample the wavefunction using the density matrix rotation approach
    #************************************************************************************
    
    # Put the orthogonaality center of the MPS at site 1
    orthogonalize!(ψ, 1)
    if orthocenter(ψ) != 1
        error("sample: MPS must have orthocenter(psi) == 1")
    end
    @show expect(ψ, "Sz"; sites = 1 : 1)

    # Compute the 1-body reduced density Matrix
    psidag = dag(ψ)
    prime!(psidag[1])

    bond_index = commonind(ψ[1], ψ[2])
    prime!(ψ[1], bond_index)
    
    rho = ψ[1] * psidag[1]
    @show rho
    @show typeof(rho), size(rho)

    # Diagonalize the reduced density matrix and obtain the eigenvalues and eigenvectors    
    # @show inds(rho)[1], inds(rho)[2]
    matrix = Matrix(rho, inds(rho)[1], inds(rho)[2])
    vals, vecs = eigen(matrix)
    @show matrix
    @show vals, vecs

    density_Sx[1] = tr(matrix * Sx_matrix)
    density_Sz[1] = tr(matrix * Sz_matrix)
    # @show density_Sz[1], Sz[1]
    
    # Perform a singular value decomposition (SVD) of the density matrix
    # U, S, V = svd(rho, inds(rho)[1], inds(rho)[2])
    # @show U, S, V 
    # @show norm(rho - U * S * V) <= 10 * eps() * norm(rho)

    # Remove prime indices from the first tensor of the MPS
    noprime!(ψ[1])
    tmp_site = siteind(ψ, 1)

    # Iterate over eigenvectors to compute the projected states
    for i in 1:2
        # Construct the projection tensor for the current eigenvector
        projection = ITensor(tmp_site)
        for j in 1:2
            projection[tmp_site => j] = vecs[i, j]
        end

        # Compute and normalize the projected state 
        ψn = ψ[1] * dag(projection)
        # ψn *= 1 / sqrt(vals[i])                 

        # Store the eigenvalue, eigenvector, and projected state in a dictionary
        push!(projected_states, Dict(
            "eigenvalue" => vals[i],
            "eigenvector" => projection,
            "state" => ψn
        ))
    end
    # projected_states = sort(projected_states, by = x -> x["eigenvalue"], rev = true)

    Prob[1] = sum(
        state["eigenvalue"] for state in projected_states
    )
    @show Prob[1]
    push!(Prob_density, [state["eigenvalue"] for state in projected_states])

    # rotation_matrix = [projected_states[1]["eigenvector"][1] projected_states[2]["eigenvector"][1];
    #                    projected_states[1]["eigenvector"][2] projected_states[2]["eigenvector"][2]]
    # eigenvalues = [projected_states[1]["eigenvalue"], projected_states[2]["eigenvalue"]]
    # @show projected_states[1]["eigenvector"][1]^2 + projected_states[1]["eigenvector"][2]^2 
    # @show eigenvalues[1]+ eigenvalues[2], eigenvalues[1] - eigenvalues[2]
    # rotated_eigenvalues = rotation_matrix * eigenvalues
    # @show rotated_eigenvalues[1], rotated_eigenvalues[2], rotated_eigenvalues[1]^2 + rotated_eigenvalues[2]^2
    # @show 0.5 * (rotated_eigenvalues[1]^2 - rotated_eigenvalues[2]^2)
    # rotated_matrix = [projected_states[1]["eigenvalue"] 0; 0 projected_states[2]["eigenvalue"]]
    # density_Sz[1] = tr(rotated_matrix * Sz_matrix)
    # density_Sz[1] = sum(
    #     state["eigenvalue"] * 0.5 * (state["eigenvector"][1] - state["eigenvector"][2])
    #     for (index, state) in enumerate(projected_states)
    # )
    # @show density_Sz[1]


    for idx1 in 2 : N
        # println(" ")
        # println(" ")
        # @show length(projected_states)
        # println(" ")
        # println(" ")

        upper_bound = min(length(projected_states), Nₛ_density)
        for idx2 in 1 : upper_bound
            ψ_copy = deepcopy(ψ)
            # @show ψ_copy

            tmp = popfirst!(projected_states)
            ψ_copy[idx1] *= tmp["state"]
            # @show ψ_copy

            # Compute the 1-body reduced density Matrix
            psidag_copy = dag(ψ_copy)
            prime!(psidag_copy[idx1])
            # @show psidag_copy
            
            if idx1 < N
                bond_index = commonind(ψ_copy[idx1], ψ_copy[idx1 + 1])
                prime!(ψ_copy[idx1], bond_index)
            end
            # @show ψ_copy
            
            rho = ψ_copy[idx1] * psidag_copy[idx1]
            # @show rho   
            
            # Diagonalize the density matrix and obtain the eigenvalues and eigenvectors
            matrix = Matrix(rho, inds(rho)[1], inds(rho)[2])
            vals, vecs = eigen(matrix)
            # @show matrix
            # @show vals, vecs
            density_Sx[idx1] += tr(matrix * Sx_matrix)
            density_Sz[idx1] += tr(matrix * Sz_matrix)
            Prob[idx1] += vals[1] + vals[2]
            @show vals[1], vals[2]

            # Remove prime indices from the first tensor of the MPS
            noprime!(ψ_copy[idx1])
            tmp_site = siteind(ψ, idx1)
            # @show tmp_site

            # Iterate over eigenvectors to compute the projected states
            for i in 1:2
                if vals[i] < 0
                    continue
                end

                # Construct the projection tensor for the current eigenvector
                projection = ITensor(tmp_site)
                for j in 1:2
                    projection[tmp_site => j] = vecs[i, j]
                end

                # Compute the projected state
                ψn_copy = ψ_copy[idx1] * dag(projection)
                # ψn_copy *= 1 / sqrt(vals[i])
                # @show vals[i], tmp["eigenvalue"]

                # Store the eigenvalue, eigenvector, and projected state in a dictionary
                push!(projected_states, Dict(
                    "eigenvalue" => vals[i],
                    "eigenvector" => projection,
                    "state" => ψn_copy
                ))
            end
        end

        density_Sx[idx1] = density_Sx[idx1] / Prob[idx1]
        density_Sz[idx1] = density_Sz[idx1] / Prob[idx1]
        @show idx1, Prob[idx1]
        
        # Sort the projected states based on the eigenvalues and compute local observables based on the projected states
        projected_states = sort(projected_states, by = x -> x["eigenvalue"], rev = true)
        if length(projected_states) > Nₛ_density
            projected_states = projected_states[1 : Nₛ_density]
        end
        # @show projected_states[1]["eigenvalue"], projected_states[2]["eigenvalue"]
        
        # density_Sz[idx1] = sum(
        #     state["eigenvalue"] * 0.5 * (state["eigenvector"][1] - state["eigenvector"][2])
        #     for state in projected_states
        # )
        # Prob[idx1] = sum(
        #     state["eigenvalue"] for state in projected_states
        # )
        # @show Prob[idx1]
        push!(Prob_density, [state["eigenvalue"] for state in projected_states])
    end


    # Check the total probability of the projected states
    @show compute_probability(projected_states, "eigenvalue")
    @show density_Sx
    @show Sx
    @show density_Sz
    @show Sz
    @show Prob 
    # @show Prob_density

    # sorted_projected_states = sort(projected_states, by = x -> x["eigenvalue"], rev = true)
    # for i in 1 : 5
    #     println(" ")
    #     println(" ")
    #     @show sorted_projected_states[i]["eigenvalue"]
    #     @show sorted_projected_states[i]["eigenvector"]
    #     println(" ")
    #     println(" ")
    # end


    #*********************************************************************************************************
    # Save results into a HDF5 file
    #*********************************************************************************************************
    h5open("data/Transverse_Ising_N$(N)_h$(h)_Sample$(Nₛ_density).h5", "w") do file
        write(file, "Sx", Sx)
        write(file, "Sz", Sz)
        write(file, "Bitstring Sx", bitstring_Sx)
        write(file, "Bitstring Sz", bitstring_Sz)
        write(file, "Density Sx", density_Sx)
        write(file, "Density Sz", density_Sz)
        write(file, "Probability", Prob)
        # write(file, "Probability Density", Prob_density)
    end

    return
end