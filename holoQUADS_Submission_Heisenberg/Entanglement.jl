## 05/11/2023
## Measure entanglement Entropy
using ITensors
using ITensors: orthocenter, sites, copy, complex, real

# Measure von Neumann entanglment entropy on a sequence of bonds
function entanglement_entropy(tmp_ψ :: MPS, length :: Int)
    entropy = []
    for site_index in 1 : length - 1 
        orthogonalize!(tmp_ψ, site_index)
        if site_index == 1
            i₁ = siteind(tmp_ψ, site_index)
            _, C1, _ = svd(tmp_ψ[site_index], i₁)
        else
            i₁, j₁ = siteind(tmp_ψ, site_index), linkind(tmp_ψ, site_index - 1)
            _, C1, _ = svd(tmp_ψ[site_index], i₁, j₁)
        end
        C1 = matrix(C1)
        SvN₁ = compute_entropy(C1)
        
        # @show site_index, SvN₁
        push!(entropy, SvN₁)
    end
    return entropy
end


# Compute von Neumann entanglement entropy given the eigen values
function compute_entropy(input_matrix)
    local tmpEntropy = 0.0
    for index in 1 : size(input_matrix, 1) 
        tmp = input_matrix[index, index]^2
        if tmp > 1E-8
            tmpEntropy += -tmp * log(tmp)
        end
    end
    return tmpEntropy
end