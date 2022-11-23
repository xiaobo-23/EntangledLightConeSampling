## Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, Printf
using Base: Float64
ITensors.disable_warn_order()


let 
    n = 10; h = 1.0
    tau = 0.1
    s = siteinds("S=1/2", n; conserve_qns = false)

    gates = ITensor[]
    s1 = s[1]
    s2 = s[n]
    
    # Notice the difference in coefficients due to the system is half-infinite chain
    hj = π * op("Sz", s1) * op("Sz", s2) + 2 * h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
    Gj = exp(-1.0im * tau / 2 * hj)
    push!(gates, Gj)
    
    # @show hj
    # @show Gj
    # @show inds(Gj)
    (s1ᵢ, snᵢ, s1ⱼ, snⱼ) = inds(Gj)
    @show s1ᵢ, snᵢ, s1ⱼ, snⱼ

    U, S, V = svd(Gj, (s1ᵢ, s1ⱼ))
    @show norm(U*S*V - Gj)
    @show S
    @show U
    @show V

    bondIndices = Vector(undef, n - 1)

    # Grab the bond indices of U and V matrices
    if hastags(inds(U)[3], "Link,u") != true
        error("SVD: fail to grab the bond indice of matrix U by its tag!")
    else 
        replacetags(inds(U)[3], "Link,u", "i1")
    end
    @show U
    bondIndices[1] = inds(U)[3]

    if hastags(inds(V)[3], "Link,v") != true
        error("SVD: fail to grab the bond indice of matrix V by its tag!")
    else
        replacetags(inds(V)[3], "Link,v", "i" * string(n - 1))
    end
    @show V
    bondIndices[n - 1] = inds(V)[3]

    @show (bondIndices[1], bondIndices[n - 1])

    longrangeGate = ITensor
    @show longrangeGate
    for ind in 2 : n - 1
        # Set up site indices
        siteStr₁, siteStr₂ = "S" * string(ind) * "_income", "S" * string(ind) * "_outcome"
        indiceIn = Index(2, siteStr₁)
        indiceOut = Index(2, siteStr₂)

        if abs(ind - (n - 1)) > 1E-8
            bondString = "i" * string(ind)
            bondIndices[ind] = Index(4, bondString)
        end

        tmpIdentity = delta(indiceIn, indiceOut) * delta(bondIndices[ind - 1], bondIndices[ind])

        if ind - 2 < 1E-8
           longrangeGate = U * tmpIdentity
        else
            longrangeGate = longrangeGate * tmpIdentity
        end 
        
        @show longrangeGate

        
        # bondStr₁, bondStr₂ = "i" * string(ind - 1), "i" * string(ind)
        # @show siteStr₁, siteStr₂, bondStr₁, bondStr₂
        # tmp₁, tmp₂ = Index(2, siteStr₁), Index(2, siteStr₂)
        # bondIndices[ind - 1], bondIndices[ind] = Index(4, bondStr₁), Index(4, bondStr₂)
        # @show bondIndices[ind - 1], bondIndices[ind]
        # tmpTensor = delta(tmp₁, tmp₂) * delta(bondIndices[ind - 1], bondIndices[ind])
        # # @show tmpTensor

        # tensorHolder = U * tmpTensor
    end

    longrangeGate = longrangeGate * V
    @show longrangeGate

    # i = Index(2, "i")
    # j = Index(2, "j") 
    # k = Index(2, "k")
    # l = Index(2, "l")
    # T = ITensor(i, j, k, l)
    # @show T; @show inds(T)

    return
end 