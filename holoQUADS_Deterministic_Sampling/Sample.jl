## 05/11/2023
## Set up the sample and reset procedure
using ITensors 
include("Projection.jl")

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
            projn[tmpS => 1] = tmp_projn[n][1]
            projn[tmpS => 2] = tmp_projn[n][2]
        
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



# 10/31/2023
# Sample a two-site MPS without projection 
# This part is redundant and will be removed in the future
function sample_without_projection(m :: MPS, j :: Int, observable_type :: AbstractString)
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
    elseif observable_type == "Sy"
        tmp_projn = Sy_projn
    elseif observable_type == "Sz"
        tmp_projn = Sz_projn
    else
        error("sample: the type of measurement doesn't exist")
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
            projn[tmpS => 1] = tmp_projn[n][1]
            projn[tmpS => 2] = tmp_projn[n][2]
        
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
    end
    return result
end 


# 06/26/2024
# Sample a two-site MPS in a deterministic way
struct DeterministicSample
    j :: Int
    probability :: Float64
    sampleMPS :: MPS
end

function deterministic_sample(m :: MPS, j :: Int, observable_type :: AbstractString)
    mpsLength = length(m)
    # @show typeof(m), m 
    
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
    elseif observable_type == "Sy"
        tmp_projn = Sy_projn
    elseif observable_type == "Sz"
        tmp_projn = Sz_projn
    else
        error("sample: the type of measurement doesn't exist")
    end

    # Sample the target observables
    # result = zeros(Int, 2)
    tmp_vector = []
    result_vector = []
    A = m[j]
    
    for ind in j:j+1
        tmpS = siteind(m, ind)
        d = dim(tmpS)
        
        if ind == j
            for index in 1 : d
                An = ITensor()
                pn = 0.0
                projn = ITensor(tmpS)
                projn[tmpS => 1] = tmp_projn[index][1]
                projn[tmpS => 2] = tmp_projn[index][2]

                An = A * dag(projn)
                pn = real(scalar(dag(An) * An))

                sample_info = (index, pn, An)
                push!(tmp_vector, sample_info)
            end
        end

        if ind == j + 1
            while !isempty(tmp_vector)
                tmp_index, tmp_pn, tmp_An = popfirst!(tmp_vector)
                A = m[ind] * tmp_An
                A *= (1. / sqrt(tmp_pn))

                for index in 1 : d
                    projn = ITensor(tmpS)
                    pn = 0.0
                    projn[tmpS => 1] = tmp_projn[index][1]
                    projn[tmpS => 2] = tmp_projn[index][2]
                
                    An = A * dag(projn); @show typeof(An)
                    pn = real(scalar(dag(An) * An))
                    sample_info = (index, tmp_index, pn, An)
                    push!(result_vector, sample_info)
                end
            end
        end
        # if ind < mpsLength
        #     A = m[ind + 1] * An
        #     A *= (1. / sqrt(pn))
        # end
    end
    
    @show result_vector
    return result_vector

end 


# 06/26/2024
# Sample a two-site MPS in a deterministic way
function single_site_deterministic_sample_initialization(m :: MPS, j :: Int, observable_type :: AbstractString) 
    # Move the orthogonality center of the MPS to site j
    orthogonalize!(m, j)
    if orthocenter(m) != j
        error("sample: MPS m must have orthocenter(m) == j")
    end
    
    # Check the normalization of the MPS
    if abs(1.0 - norm(m[j])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[j]))")
    end

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
        An = ITensor()
        pn = 0.0
        projn = ITensor(tmpS)
        projn[tmpS => 1] = tmp_projn[index][1]
        projn[tmpS => 2] = tmp_projn[index][2]

        An = A * dag(projn)
        pn = real(scalar(dag(An) * An))

        # sample_info = ([index], pn, An)
        sample_info = [[index], pn, An]
        push!(result_vector, sample_info)
    end

    # @show result_vector
    return result_vector
end


# 06/26/2024
# Sample a two-site MPS in a deterministic way
function single_site_deterministic_sample(m :: MPS, j :: Int, observable_type :: AbstractString, tmp_An :: ITensor, tmp_pn :: Float64) 
    # Make sure the site to be sampled is not the first site
    if j == 1
        error("sample: the site index j must be larger than 1")
    end
    
    # Move the orthogonality center of the MPS to site j
    orthogonalize!(m, j)
    if orthocenter(m) != j
        error("sample: MPS m must have orthocenter(m) == j")
    end
    
    # Check the normalization of the MPS
    if abs(1.0 - norm(m[j])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[j]))")
    end

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
    A = m[j] * tmp_An
    # A *= (1. / sqrt(tmp_pn))
    tmpS = siteind(m, j)
    d = dim(tmpS)

    for index in 1 : d
        An = ITensor()
        pn = 0.0
        projn = ITensor(tmpS)
        projn[tmpS => 1] = tmp_projn[index][1]
        projn[tmpS => 2] = tmp_projn[index][2]

        An = A * dag(projn)
        pn = real(scalar(dag(An) * An))

        # sample_info = ([index], pn, An)
        sample_info = [[index], pn, An]
        push!(result_vector, sample_info)
    end

    # @show result_vector
    return result_vector
end