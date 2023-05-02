## 05/02/2023
## IMPLEMENT THE HOLOQAUDS CIRCUITS WITHOUT RECYCLING AND LONG-RANGE GATES.

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex, real
using Base: Float64
using Base: product
using Random
ITensors.disable_warn_order()




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

    # Define projectors in the Sz basis
    Sx_projn = 1/sqrt(2) * [[1, 1], [1, -1]]
    Sy_projn = 1/sqrt(2) * [[1, 1.0im], [1, -1.0im]]
    Sz_projn = [[1, 0], [0, 1]]
    
    if observable_type == "Sx":
        tmp_projn = Sx_projn
    else if observable_type == "Sy":
        tmp_projn = Sy_projn
    else if observable_type == "Sz":
        tmp_projn = Sz_projn
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
    end
    return result
end 

# Contruct layers of two-site gates including the Ising interaction and longitudinal fileds in the left light cone.
function left_light_cone(number_of_gates :: Int, parity :: Int, longitudinal_field :: Float64, Δτ :: Float64, tmp_sites)
    gates = ITensor[]

    for ind in 1 : number_of_gates
        tmp_index = 2 * ind - parity
        s1 = tmp_sites[tmp_index]
        s2 = tmp_sites[tmp_index + 1]
       
        if tmp_index - 1 < 1E-8
            coeff₁ = 2
            coeff₂ = 1
        else
            coeff₁ = 1
            coeff₂ = 1
        end

        # hj = coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        # hj = π * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        hj = π/2 * op("Sz", s1) * op("Sz", s2) + coeff₁ * longitudinal_field * op("Sz", s1) * op("Id", s2) + coeff₂ * longitudinal_field * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * Δτ * hj)
        push!(gates, Gj)
    end
    return gates
end