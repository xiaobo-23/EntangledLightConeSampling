# 08/15/2024
# Implement deterministic sampling for a random MPS

using ITensors, ITensorMPS
using Random
using Statistics
using HDF5
using TimerOutputs

# include("Sample.jl")
# include("Projection.jl")



# Sample a two-site MPS to compute Sx, Sy or Sz
function sample(m :: MPS, j :: Int, observable_type :: AbstractString)
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

    if observable_type == "Sx"
        tmp_projn = Sx_projn
        projn_up = Sx_projn_plus
        projn_dn = Sx_projn_minus
    elseif observable_type == "Sy"
        tmp_projn = Sy_projn
        projn_up = Sy_projn_plus
        projn_dn = Sy_projn_minus
    elseif observable_type == "Sz"
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
            projn[tmpS => n] = 1.0
            # projn[tmpS => 1] = tmp_projn[n][1]
            # projn[tmpS => 2] = tmp_projn[n][2]
        
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


# 08/16/2024
# Sample a single-site MPS in a deterministic way
function deterministic_sampling_single_site_MPS(m :: MPS, j :: Int, observable_type :: AbstractString) 
    # Move the orthogonality center of the MPS to site j
    orthogonalize!(m, j)
    if orthocenter(m) != j
        error("sample: MPS m must have orthocenter(m) == j")
    end
    
    # # Check the normalization of the MPS
    # if abs(1.0 - norm(m[j])) > 1E-8
    #     error("sample: MPS is not normalized, norm=$(norm(m[j]))")
    # end

    # Set the projection operator
    if observable_type == "Sx"
        tmp_projn = Sx_projn
    elseif observable_type == "Sy"
        tmp_projn = Sy_projn
    elseif observable_type == "Sz"
        tmp_projn = Sz_projn
    else
        error("sample: the type of measurement doesn't exist")
    end

    # Sample the target observables
    result_vector = []
    A = m[j]
    tmpS = siteind(m, j)
    d = dim(tmpS)

    for index in 1 : d
        pn = 0.0
        An = ITensor()
        projn = ITensor(tmpS)
        projn[tmpS => index] = 1.0; 
        # projn[tmpS => 1] = tmp_projn[index][1]
        # projn[tmpS => 2] = tmp_projn[index][2]

        @show typeof(A), typeof(projn), A, projn
        An = A * dag(projn); # @show typeof(An)
        pn = real(scalar(dag(An) * An))


        sample_info = [[index], pn, An]
        push!(result_vector, sample_info)
    end
    @show length(result_vector)
    return result_vector
end

# 08/16/2024
# Sample a single-site MPS in a deterministic way
function deterministic_sampling_single_site_Tensor(m :: ITensor, observable_type :: AbstractString) 
    # # Move the orthogonality center of the MPS to site j
    # orthogonalize!(m, j)
    # if orthocenter(m) != j
    #     error("sample: MPS m must have orthocenter(m) == j")
    # end
    
    # # Check the normalization of the MPS
    # if abs(1.0 - norm(m)) > 1E-8
    #     error("sample: MPS is not normalized, norm=$(norm(m))")
    # end

    # Set the projection operator
    if observable_type == "Sx"
        tmp_projn = Sx_projn
    elseif observable_type == "Sy"
        tmp_projn = Sy_projn
    elseif observable_type == "Sz"
        tmp_projn = Sz_projn
    else
        error("sample: the type of measurement doesn't exist")
    end

    # Sample the target observables
    result_vector = []
    A = m
    tmpS = siteind(m)
    d = dim(tmpS)

    for index in 1 : d
        pn = 0.0
        An = ITensor()
        projn = ITensor(tmpS)
        projn[tmpS => index] = 1.0; 
        # projn[tmpS => 1] = tmp_projn[index][1]
        # projn[tmpS => 2] = tmp_projn[index][2]

        @show A, projn
        An = A * dag(projn); # @show typeof(An)
        pn = real(scalar(dag(An) * An))


        sample_info = [[index], pn, An]
        push!(result_vector, sample_info)
    end
    @show length(result_vector)
    return result_vector
end


let 
    # Initialize the random MPS
    N = 10                          # Number of physical sites
    sites = siteinds("S=1/2", N; conserve_qns = false) 
    state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    Random.seed!(1000)
    ψ₀ = randomMPS(sites, state, linkdims = 10)
    normalize!(ψ₀)
    Sz = expect(ψ₀, "Sz"; sites = 1 : N)

    # Generate the samples using Born rule
    Nₛ = 20             # Number of samples
    bitstring = Array{Float64}(undef, Nₛ, N)
    for i = 1 : Nₛ
        ψ_copy = deepcopy(ψ₀)
        for j = 1 : 2 : N
            tmp = sample(ψ_copy, j, "Sz")
            @show i, j, tmp
            normalize!(ψ_copy)
            bitstring[i, j: j + 1] = tmp
        end
    end
    bitstring[bitstring .== 1] .= 0.5
    bitstring[bitstring .== 2] .= -0.5

    
    # @show bitstring
    sample_expectation = mean(bitstring, dims = 1)
    sample_std = std(bitstring, corrected = true, dims = 1) / sqrt(Nₛ)
    # @show Sz[1 : 2], sample_expectation[1 : 2], sample_std[1 : 2]


    # Sample the wave function in a deterministic way
    dsample = []
    ψ_copy = deepcopy(ψ₀)
    dsample = deterministic_sampling_single_site_MPS(ψ_copy, 1, "Sz")
    @show 0.5 * (dsample[1][2] - dsample[2][2]), Sz[1]
    @show typeof(dsample[1][3]), dsample[1][3]
    @show typeof(dsample[2][3]), dsample[2][3]
    
    println(" ")
    println(" ")
    println(" ")
    iterations = length(dsample)
    Sz₂ = 0
    Prob = 0
    for index in 1 : iterations
        # @show typeof(dsample)
        # @show index, dsample[index][1], dsample[index][2]
        tmp = popfirst!(dsample)
        @show tmp

        ψ_tmp = deepcopy(ψ_copy)
        @show typeof(ψ_tmp[2]), ψ_tmp[2]
        ψ_tmp[2] = tmp[3] * ψ_tmp[2]    
        # # normalize!(ψ_tmp)
        @show typeof(ψ_tmp[2]), ψ_tmp[2]
        ψ_update = MPS(ψ_tmp[2 : N])
        # normalize!(ψ_update)
        # @show size(ψ_update)
        # # ψ_update = ψ_tmp[2 : N]; @show typeof(ψ_update), typeof(ψ_tmp)
        tmp_sample = deterministic_sampling_single_site_MPS(ψ_update, 1, "Sz")
        @show tmp_sample
        # @show ψ_update[2]
        # push!(dsample, tmp_sample)
        Sz₂ += 0.5 * (tmp_sample[1][2] - tmp_sample[2][2])
        Prob += tmp_sample[1][2] + tmp_sample[2][2]
        println(" ")
        println(" ")
        println(" ")
    end

    @show Sz₂, Sz[2], Prob

    # @show dsample[1]
   
    # @show typeof(ψ_copy[2]), ψ_copy[2]
    # ψ_copy[2] = dsample[1][3] * ψ_copy[2]
    # @show typeof(ψ_copy[2]), ψ_copy[2]
    # ψ_update = ψ_copy[2 : N]
    # @show typeof(ψ_update), ψ_update[1]
    # normalize!(ψ_copy)
    # tmp = deterministic_sampling_single_site(ψ_copy[2], "Sz")
    # @show tmp



    # # Save the results to a file
    # h5open("Sample_Random_MPS.h5", "w") do file
    #     write(file, "Sz", Sz)
    #     write(file, "Sample ave.", sample_expectation)
    #     write(file, "Sample err.", sample_std)
    # end

end