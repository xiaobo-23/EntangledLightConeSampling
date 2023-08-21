## 08/15/2023
## Sample and reset procedure used in the holoQADS circuit for the Heisenberg model
using ITensors
include("Projection.jl")

# Sample a two-site MPS and reset the MPS to the initial state
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

    tmp_projn = Sz_projn
    projn_up = Sz_projn_up
    projn_dn = Sz_projn_dn

    result = zeros(Int, 2)
    A = m[j]
    for ind = j:j+1
        tmpS = siteind(m, ind)
        # println("Before taking measurements")
        # @show(m[ind])
        d = dim(tmpS)
        pdisc = 0.0
        r = rand()

        n = 1
        An = ITensor()
        pn = 0.0

        while n <= d
            # projn = ITensor(tmpS)
            # projn[tmpS=>n] = 1.0

            projn = ITensor(tmpS)
            projn[tmpS => 1] = tmp_projn[n][1]
            projn[tmpS => 2] = tmp_projn[n][2]
            
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind-j+1] = n

        if ind < mpsLength
            A = m[ind+1] * An
            A *= (1.0 / sqrt(pn))
        end

        # ## 08/15/2023
        # ## The matrices used in the reset procedure depend on the physical state of the site and the initial wavefunction
        
        # # Reset to the initial Neel state |up, down, up, down, ...> with n=1 --> |up> and n=2 --> |down>
        # if ind % 2 == 1
        #     if n == 1             
        #         tmp_reset = ITensor(Sz_matrix, tmpS', tmpS)
        #     else
        #         tmp_reset = ITensor(S⁺_matrix, tmpS', tmpS)
        #     end
        # else
        #     if n == 1
        #         tmp_reset = ITensor(S⁻_matrix, tmpS', tmpS)
        #     else
        #         tmp_reset = ITensor(Sz_matrix, tmpS', tmpS)
        #     end
        # end
        

        ## 08/20/2023
        ## Debig the projection & sampling part
        ## Using the projection and sampling code from the SDKI model
        if abs(n - 1) < 1E-8
            tmp_reset = ITensor(projn_up, tmpS', tmpS)
        else
            tmp_reset = ITensor(projn_dn, tmpS', tmpS)
        end

        m[ind] *= tmp_reset
        noprime!(m[ind])
    end
    return result
end


# Compute the overlap between the time-evolved wavefunction and 
function compute_overlap(tmp_ψ₁::MPS, tmp_ψ₂::MPS)
    overlap = abs(inner(tmp_ψ₁, tmp_ψ₂))
    println("")
    println("")
    @show overlap
    println("")
    println("")
    return overlap
end