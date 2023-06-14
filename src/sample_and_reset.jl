## 02/21/2023
## Sample and reset procedure in the holographic quantum dynamics circuit. 
## Further modifications might be needed, depending on the initial physical states.

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product
using Random

ITensors.disable_warn_order()


# Sample and reset one two-site MPS
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

    # Take measurements and reset the two-site MPS to |up, down> Neel state
    # Need to be modified based on the initialization of MPS
    projn_up_matrix = [
        1 0
        0 0
    ]
    S⁻_matrix = [
        0 0
        1 0
    ]
    projn_dn_matrix = [
        0 0
        0 1
    ]
    S⁺_matrix = [
        0 1
        0 0
    ]

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
            projn = ITensor(tmpS)
            projn[tmpS=>n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind-j+1] = n
        # @show result[ind - j + 1]
        # @show An

        if ind < mpsLength
            A = m[ind+1] * An
            A *= (1.0 / sqrt(pn))
        end

        '''
        # 01/27/2022
        # Comment: the reset procedure needs to be revised 
        # Use a product state of entangled (two-site) pairs and reset the state to |Psi (t=0)> instead of |up, down>. 
        '''

        # n denotes the corresponding physical state: n=1 --> |up> and n=2 --> |down>
        if ind % 2 == 1
            if n - 1 < 1E-8
                tmpReset = ITensor(projn_up_matrix, tmpS', tmpS)
            else
                tmpReset = ITensor(S⁺_matrix, tmpS', tmpS)
            end
        else
            if n - 1 < 1E-8
                tmpReset = ITensor(S⁻_matrix, tmpS', tmpS)
            else
                tmpReset = ITensor(projn_dn_matrix, tmpS', tmpS)
            end
        end
        m[ind] *= tmpReset
        noprime!(m[ind])
        # println("After resetting")
        # @show m[ind]
    end
    # println("")
    # println("")
    # println("Measure sites $j and $(j+1)!")
    # println("")
    # println("")
    return result
end
