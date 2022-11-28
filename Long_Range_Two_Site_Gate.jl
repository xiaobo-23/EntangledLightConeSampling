## 11/23/2022
## Implement the function to generate a long-range two-site gate with a bunch identity tensor inserted in the middle
## TO-DO: generalize the function to apply to any two sites with arbitrary distance; For the holoQUADS circuit, the long-range gate is always applied to the first and last site.

using ITensors
using ITensors: orthocenter, sites, copy 
using Base: Float64
ITensors.disable_warn_order()

let 
    n = 10; h = 1.0
    tau = 0.1
    
    s = siteinds("S=1/2", n; conserve_qns = false)
    s1 = s[1]
    s2 = s[n]
    
    # Notice the difference in coefficients due to the system is half-infinite chain
    hj = π * op("Sz", s1) * op("Sz", s2) + 2 * h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
    Gj = exp(-1.0im * tau / 2 * hj)
    # @show hj
    # @show Gj
    @show inds(Gj)

    # for ind in 1 : n
    #     @show s[ind], s[ind]'
    # end

    U, S, V = svd(Gj, (s[1], s[1]'))
    @show norm(U*S*V - Gj)
    # @show S
    # @show U
    # @show V

    # Absorb the S matrix into the U matrix on the left
    U = U * S
    # @show U

    # Make a vector to store the bond indices
    bondIndices = Vector(undef, n - 1)

    # Grab the bond indices of U and V matrices
    if hastags(inds(U)[3], "Link,v") != true           # The original tag of this index of U matrix should be "Link,u".  But we absorbed S matrix into the U matrix.
        error("SVD: fail to grab the bond indice of matrix U by its tag!")
    else 
        replacetags!(U, "Link,v", "i1")
    end
    # @show U
    bondIndices[1] = inds(U)[3]

    if hastags(inds(V)[3], "Link,v") != true
        error("SVD: fail to grab the bond indice of matrix V by its tag!")
    else
        replacetags!(V, "Link,v", "i" * string(n))
    end
    # @show V
    bondIndices[n - 1] = inds(V)[3]

    @show (bondIndices[1], bondIndices[n - 1])

    longrangeGate = ITensor[]; push!(longrangeGate, U)
    @show sizeof(longrangeGate)
    # @show longrangeGate
    for ind in 2 : n - 1
        # Set up site indices
        if abs(ind - (n - 1)) > 1E-8
            bondString = "i" * string(ind)
            bondIndices[ind] = Index(4, bondString)
        end

        # Make the identity tensor
        @show s[ind], s[ind]'
        tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind])
        push!(longrangeGate, tmpIdentity)

        @show sizeof(longrangeGate)
        # @show longrangeGate
    end

    push!(longrangeGate, V)
    @show sizeof(longrangeGate)
    # @show longrangeGate

    return
end 