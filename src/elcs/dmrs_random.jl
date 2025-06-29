# 5/15/2025
# Implement the idea of density matrix rotation sampling

using ITensors
using ITensorMPS
using Random
using Statistics
using LinearAlgebra
using HDF5

# include("Sample.jl")
# include("Projection.jl")


# Set up the local observables in the Sz basis  
const Sx_matrix = 0.5 * [0 1; 1 0]
const Sy_matrix = 0.5 * [0.0 -1.0im; 1.0im 0.0]
const Sz_matrix = 0.5 * Diagonal([1, -1])


function sample_density_matrix(input_ψ :: MPS, sample_idx :: Int, string_length :: Int)
    # Put the orthogonality center of the MPS at the site to be sampled
    orthogonalize!(input_ψ, sample_idx)
    if orthocenter(input_ψ) != sample_idx
        error("sample: MPS must have orthocenter(psi) == $sample_idx")
    end

    # Check the normalization of the MPS
    if abs(1.0 - norm(input_ψ[sample_idx])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(input_ψ[sample_idx]))")
    end

    # Initialize arrays to store the sampled expectation values
    Sx_sample = zeros(Float64, 2)
    Sz_sample = zeros(Float64, 2)

    A = input_ψ[sample_idx]
    for j in sample_idx : sample_idx + 1
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
        

        # Compute the expectation values of Sx and Sz based on the density matrix
        if mod(j, 2) == 1
            tmp_idx = 1
        else
            tmp_idx = 2
        end
        
        Sx_sample[tmp_idx] = real(tr(matrix * Sx_matrix))
        Sz_sample[tmp_idx] = real(tr(matrix * Sz_matrix))

        noprime!(A) 
        An = ITensor()
        r = rand()
        probability = 0.0
        n = 1

        while n <= dimension
            probability += real(vals[n])
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
        @show A

        
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
        # @show input_ψ[j]
    end
    # @show A 
    return Sx_sample, Sz_sample
end




function sample_density_matrix(input_ψ :: MPS, sample_idx :: Int, string_length :: Int)
    # Put the orthogonality center of the MPS at the site to be sampled
    orthogonalize!(input_ψ, sample_idx)
    if orthocenter(input_ψ) != sample_idx
        error("sample: MPS must have orthocenter(psi) == $sample_idx")
    end

    # Check the normalization of the MPS
    if abs(1.0 - norm(input_ψ[sample_idx])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(input_ψ[sample_idx]))")
    end

    # Initialize arrays to store the sampled expectation values
    Sx_sample = zeros(Float64, 2)
    Sz_sample = zeros(Float64, 2)

    A = input_ψ[sample_idx]
    for j in sample_idx : sample_idx + 1
        tmp_site = siteind(input_ψ, j)
        dimension = dim(tmp_site)

        psidag = dag(A)
        prime!(psidag)

        if j < 3
            error("sample density matrix: site index j must be at least 3")
        end
        prime!(A, commonind(input_ψ[j - 1], input_ψ[j]))
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
        

        # Compute the expectation values of Sx and Sz based on the density matrix
        if mod(j, 2) == 1
            tmp_idx = 1
        else
            tmp_idx = 2
        end
        
        Sx_sample[tmp_idx] = real(tr(matrix * Sx_matrix))
        Sz_sample[tmp_idx] = real(tr(matrix * Sz_matrix))

        noprime!(A) 
        An = ITensor()
        r = rand()
        probability = 0.0
        n = 1

        while n <= dimension
            probability += real(vals[n])
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
        @show A

        
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
        # @show input_ψ[j]
    end
    # @show A 
    return Sx_sample, Sz_sample
end