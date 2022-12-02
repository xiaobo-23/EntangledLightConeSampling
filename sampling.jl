#@ 11/28/2022
## Implement the quantum circuit for the SDKI model using classical MPS.

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy
using Base: Float64, project_deps_get
ITensors.disable_warn_order()

# Implement the function to generate one sample of the probability distribution 
function sample(m::MPS, j::Int)
    mpsLength = length(m)

    # Move the orthogonality center of the MPS to site j
    orthogonalize!(m, j)
    if orthocenter(m) != j
        error("sample: MPS m must have orthocenter(m) == 1")
    end
    # Check the normalization of the MPS
    if abs(1.0 - norm(m[j])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[1]))")
    end

    projn0_Matrix = [1  0; 0  0]
    projnLower_Matrix = [0  0; 1  0]
    # @show projectionMatrix, sizeof(projectionMatrix)
    result = zeros(Int, 2)
    A = m[j]
    @show A
    @show m[j]

    for ind in j:j+1
        s = siteind(m, ind)
        d = dim(s)
        pdisc = 0.0
        r = rand()

        n = 1
        An = ITensor()
        pn = 0.0

        while n <= d
            # @show A
            # @show m[ind]
            projn = ITensor(s)
            projn[s => n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind - j + 1] = n
        @show result[ind - j + 1]
        # @show An

        if ind < mpsLength
            A = m[ind + 1] * An
            A *= (1. / sqrt(pn))
        end

        @show m[ind]
        if n - 1 < 1E-8
            tmpReset = ITensor(projn0_Matrix, s, s')
        else
            tmpReset = ITensor(projnLower_Matrix, s, s')
        end
        m[ind] *= tmpReset
        noprime!(m[ind])
        @show m[ind]
    end
end 

let 
    N = 20
    initial_s = siteinds("S=1/2", N; conserve_qns = false)  
    # s = siteinds("S=1/2", N; conserve_qns = true)

    # Initialize the wavefunction
    # ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # ψ₀ = deepcopy(ψ)
    # @show eltype(ψ), eltype(ψ[1])
    # states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # @show maxlinkdim(ψ)

    # # Benchmark the accuracy of the two-sute sampling procedure
    # Sz = complex(zeros(2, N))
    # Sz[1, :] = expect(ψ, "Sz"; sites = 1 : N)
    # sample(ψ, 5)
    # Sz[2, :] = expect(ψ, "Sz"; sites = 1 : N)

    # overlap = Complex[]
    # overlap = abs(inner(ψ, ψ₀))
    # @show overlap
    
    initialization_number = 6
    Sz = complex(zeros(2 * initialization_number, N))
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    for ind in 1 : initialization_number
        ψ = randomMPS(initial_s, states, linkdims = 4)
        ψ₀ = deepcopy(ψ)
        @show maxlinkdim(ψ)
        Sz[2 * ind - 1, :] = expect(ψ, "Sz"; sites = 1 : N)
        sample(ψ, 2 * ind - 1)
        Sz[2 * ind, :] = expect(ψ, "Sz"; sites = 1 : N)

        overlap = Complex[]
        overlap = abs(inner(ψ, ψ₀))
        @show overlap
    end


    # Store data into a hdf5 file
    file = h5open("Data/Sample_Test_Random.h5", "w")
    write(file, "Sz", Sz)
    close(file)
    
    return
end 