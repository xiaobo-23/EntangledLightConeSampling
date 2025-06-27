# 5/15/2025
# Random sampling of the wavefunction using the density matrix rotation algorithm (DMRS)


using ITensors, ITensorMPS
using Random
using Statistics
using LinearAlgebra
using HDF5

# include("Sample.jl")
include("Projection.jl")


# Set up the local observables in the Sz basis  
const Sx_matrix = 0.5 * [0 1; 1 0]
const Sy_matrix = 0.5 * [0.0 -1.0im; 1.0im 0.0]
const Sz_matrix = 0.5 * Diagonal([1, -1])


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
        error("sample: observable doesn't exist or is not implement in the sampling function")
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
            projn[tmpS => 1] = tmp_projn[n, 1]
            projn[tmpS => 2] = tmp_projn[n, 2]
        
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



function sample_density_matrix(input_ψ :: MPS, string_length :: Int)
    # Put the orthogonaality center of the MPS at site 1
    orthogonalize!(input_ψ, 1)
    if orthocenter(input_ψ) != 1
        error("sample: MPS must have orthocenter(psi) == 1")
    end
    # @show expect(ψ, "Sz"; sites = 1 : 1)

    if length(input_ψ) != string_length
        error("density matrix sampling: input wavefunction must have the same length as the number of physical sites")
    end

    # Initialize arrays to store the sampled expectation values
    Sx_sample = zeros(Float64, string_length)
    Sz_sample = zeros(Float64, string_length)

    A = input_ψ[1]
    for j in 1 : string_length
        tmp_site = siteind(input_ψ, j)
        dimension = dim(tmp_site)

        psidag = dag(A)
        prime!(psidag)

        if j < string_length
            prime!(A, commonind(input_ψ[j], input_ψ[j + 1]))
        end
        # @show A 
        # @show psidag

        # Compute the 1-body reduced density matrix
        rho = A * psidag 
        @show rho
        matrix = Matrix(rho, inds(rho)[1], inds(rho)[2])
        vals, vecs = eigen(matrix)    
        # @show rho

        # Compute the expectation values of Sx and Sz based on the density matrix
        Sx_sample[j] = tr(matrix * Sx_matrix)
        Sz_sample[j] = tr(matrix * Sz_matrix)

        noprime!(A) 
        An = ITensor()
        r = rand()
        probability = 0.0
        n = 1

        while n <= dimension
            probability += vals[n]
            (r < probability) && break
            n += 1
        end
        # @show n 
        
        projection = ITensor(vecs[:, n], tmp_site)
        An = A * dag(projection)
        if j < string_length
            A = input_ψ[j + 1] * An
            A /= sqrt(vals[n])
        end
        # @show A

        
        # Collapse the site based on the measurement
        input_ψ[j] *= dag(projection)
        tmp = input_ψ[j]
        if j < string_length
            input_ψ[j + 1] *= tmp
            input_ψ[j + 1] /= sqrt(vals[n])
        end 
        input_ψ[j] = ITensor(vecs[:, n], tmp_site)
        @show input_ψ[j], input_ψ[j + 1]

        
        # # An alternative way to collapse the site based on the measurement     
        # projection_matrix = vecs[:, n] * vecs[:, n]'
        # wavefunction_projector = ITensor(projection_matrix, tmp_site', tmp_site)
        # input_ψ[j] *= wavefunction_projector
        # noprime!(input_ψ[j])
        # @show input_ψ[j], input_ψ[j + 1]
    end
    return Sx_sample, Sz_sample
end


# function sample_density_matrix(input_ψ :: MPS, string_length :: Int)
#     # Put the orthogonaality center of the MPS at site 1
#     orthogonalize!(input_ψ, 1)
#     if orthocenter(input_ψ) != 1
#         error("sample: MPS must have orthocenter(psi) == 1")
#     end
#     # @show expect(ψ, "Sz"; sites = 1 : 1)

#     if length(input_ψ) != string_length
#         error("density matrix sampling: input wavefunction must have the same length as the number of physical sites")
#     end

#     # Initialize arrays to store the sampled expectation values
#     Sx_sample = zeros(Float64, string_length)
#     Sz_sample = zeros(Float64, string_length)

#     A = input_ψ[1]
#     for j in 1 : string_length
#         tmp_site = siteind(input_ψ, j)
#         dimension = dim(tmp_site)

#         psidag = dag(A)
#         prime!(psidag)

#         if j < string_length
#             prime!(A, commonind(input_ψ[j], input_ψ[j + 1]))
#         end
#         # @show A 
#         # @show psidag

#         # Compute the 1-body reduced density matrix
#         rho = A * psidag 
#         @show rho
#         matrix = Matrix(rho, inds(rho)[1], inds(rho)[2])
#         vals, vecs = eigen(matrix)    
#         # @show rho

#         # Compute the expectation values of Sx and Sz based on the density matrix
#         Sx_sample[j] = tr(matrix * Sx_matrix)
#         Sz_sample[j] = tr(matrix * Sz_matrix)

#         noprime!(A) 
#         An = ITensor()
#         r = rand()
#         probability = 0.0
#         n = 1

#         while n <= dimension
#             probability += vals[n]
#             (r < probability) && break
#             n += 1
#         end
#         # @show n 
        

#         projection = ITensor(vecs[:, n], tmp_site)
#         An = A * dag(projection)
#         if j < string_length
#             A = input_ψ[j + 1] * An
#             A /= sqrt(vals[n])
#         end
#         # @show A

#         input_ψ[j] *= dag(projection)
#         tmp = input_ψ[j]
#         if j < string_length
#             input_ψ[j + 1] *= tmp
#             input_ψ[j + 1] /= sqrt(vals[n])
#         end
#     end

#     return Sx_sample, Sz_sample
# end


let
    # Initialize the system and set up parameters
    N = 8              
    J = 1                  
    h = 0.5                                
    Nₛ_dmrs = 2                               # Number of states sampled from density matrix 
    Nₛ = 2                                    # Number of bitstrings sampled according to the Born rule                   

    # # Read in the ground-state wavefunction from a file and sample the wavefunction
    # println(" ")
    # println("*************************************************************************************")
    # println("Read in the wavefunction from a file and start the sampling process.")
    # println("*************************************************************************************")
    # println(" ")

    # Δ = 1.5 * J
    # file = h5open("data/XXZ_N128_Delta2.0_Psi.h5", "r")
    # ψ_original = read(file, "Psi", MPS)
    # Sz = expect(ψ_original, "Sz"; sites = 1 : N)
    # Sx = expect(ψ_original, "Sx"; sites = 1 : N)
    # Czz = correlation_matrix(ψ_original, "Sz", "Sz"; sites = 1 : N)
    # @show linkdims(ψ_original)
    # ************************************************************************************************
    # ************************************************************************************************


    # Set up the initial wavefunction as a product state or a random MPS
    sites = siteinds("S=1/2", N; conserve_qns = false) 
    state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]            # Initialize the state as a Neel state
    # # state = fill("+", N)                                  # Initialize the state as a product state of |+>
    # # ψ = MPS(sites, state)

    # # Alternatively, initialize the state of the system as a random MPS
    # # ψ = random_mps(sites, state; linkdims = 8)
    
    # ************************************************************************************
    # Set up the transverse field Ising Hamiltonian
    # ************************************************************************************
    os = OpSum()
    for j = 1 : N - 1
        os += J, "Sz", j, "Sz", j + 1
        os += -h, "Sx", j
    end
    os += -h, "Sx", N
    H = MPO(os, sites)
    ψ₀ = randomMPS(sites, state, linkdims = 2)
    # ************************************************************************************
    # ************************************************************************************


    #************************************************************************************
    # Set up the XXZ Heisenberg Hamiltonian
    #************************************************************************************
    # Δ = 1.5 * J
    # os = OpSum()
    # for j = 1 : N - 1
    #     os += 0.5J, "S+", j, "S-", j + 1
    #     os += 0.5J, "S-", j, "S+", j + 1
    #     os += Δ, "Sz", j, "Sz", j + 1
    # end
    # H = MPO(os, sites)
    # ψ₀ = randomMPS(sites, state, linkdims = 2)
    #************************************************************************************
    #************************************************************************************


    #***********************************************************************************
    # # Obtain the ground-state wavefunction using DMRG
    #***********************************************************************************
    cutoff = [1E-10]
    nsweeps = 10
    maxdim = [10,20,100,100,200]
    energy, ψ_original = dmrg(H, ψ₀; nsweeps,maxdim,cutoff)
    
    Sx = expect(ψ_original, "Sx"; sites = 1 : N)
    Sz = expect(ψ_original, "Sz"; sites = 1 : N)
    @show Sx, Sz
    @show linkdims(ψ_original)
    #************************************************************************************
    #************************************************************************************


    #***********************************************************************************
    # Generate bistrings in the Sx basis according to the Born rule
    #***********************************************************************************
    bitstring_Sx = zeros(Float64, Nₛ, N)
    bitstring_Sz = zeros(Float64, Nₛ, N)

    for i = 1 : Nₛ
        ψ_copy = deepcopy(ψ_original)
        println("Generate bitstring #$(i)")
        println("")
        for j = 1 : 2 : N
            tmp = sample(ψ_copy, j, "Sx")
            normalize!(ψ_copy)
            bitstring_Sx[i, j: j + 1] = tmp
        end

        ψ_copy = deepcopy(ψ_original)
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

    
    # Compute the expectation value and standard deviation of one-point functions based on the samples 
    Sx_ave, Sx_err = mean(bitstring_Sx, dims = 1), std(bitstring_Sx, corrected = true, dims = 1) / sqrt(Nₛ)
    Sz_ave, Sz_err = mean(bitstring_Sz, dims = 1), std(bitstring_Sz, corrected = true, dims = 1) / sqrt(Nₛ)

    
    #************************************************************************************
    # Sample the wavefunction using the density matrix rotation algorithm
    #************************************************************************************
    dmrs_Sx = zeros(Float64, Nₛ_dmrs, N)
    dmrs_Sz = zeros(Float64, Nₛ_dmrs, N)

    for i in 1 : Nₛ_dmrs
        ψ = deepcopy(ψ_original)
        dmrs_Sx[i, :], dmrs_Sz[i, :] = sample_density_matrix(ψ, N)
    end 

    # for i in 1 : Nₛ_dmrs
    #     ψ = deepcopy(ψ_original)
        
    #     # Put the orthogonaality center of the MPS at site 1
    #     orthogonalize!(ψ, 1)
    #     if orthocenter(ψ) != 1
    #         error("sample: MPS must have orthocenter(psi) == 1")
    #     end
    #     # @show expect(ψ, "Sz"; sites = 1 : 1)

    #     A = ψ[1] 
    #     for j in 1 : N
    #         tmp_site = siteind(ψ, j)
    #         dimension = dim(tmp_site)

    #         psidag = dag(A)
    #         prime!(psidag)

    #         if j < N
    #             prime!(A, commonind(ψ[j], ψ[j + 1]))
    #         end
    #         # @show A 
    #         # @show psidag

    #         # Compute the 1-body reduced density matrix
    #         rho = A * psidag 
    #         matrix = Matrix(rho, inds(rho)[1], inds(rho)[2])
    #         vals, vecs = eigen(matrix)    
    #         @show rho

    #         # Compute the expectation values of Sx and Sz based on the density matrix
    #         dmrs_Sx[i, j] = tr(matrix * Sx_matrix)
    #         dmrs_Sz[i, j] = tr(matrix * Sz_matrix)
            
    #         noprime!(A) 
    #         An = ITensor()
    #         r = rand()
    #         probability = 0.0
    #         n = 1

    #         while n <= dimension
    #             probability += vals[n]
    #             (r < probability) && break
    #             n += 1
    #         end
    #         @show n 
            
    #         projection = ITensor(vecs[:, n], tmp_site)
    #         An = A * dag(projection)
    #         if j < N
    #             A = ψ[j + 1] * An
    #             A /= sqrt(vals[n])
    #         end
    #     end
    # end 
    
    @show linkdims(ψ_original)
    #*********************************************************************************************************
    # Save results into a HDF5 file
    #*********************************************************************************************************
    if h > 1e-8
        files_name = "data/TFIM_N$(N)_h$(h)_sample$(Nₛ_dmrs).h5"
    else
        files_name = "data/XXZ_N$(N)_Delta$(Δ)_sample$(Nₛ_dmrs).h5"
    end

    h5open(files_name, "w") do file
        write(file, "Sx", Sx)
        write(file, "Sz", Sz)
        write(file, "Bitstring Sx", bitstring_Sx)
        write(file, "Bitstring Sz", bitstring_Sz)
        write(file, "RDM Sx", dmrs_Sx)
        write(file, "RDM Sz", dmrs_Sz)
    end

    return
end