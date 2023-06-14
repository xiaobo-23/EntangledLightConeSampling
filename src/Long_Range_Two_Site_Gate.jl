## 11/23/2022
## Implement the function to generate a long-range two-site gate with a bunch identity tensor inserted in the middle
## TO-DO: generalize the function to apply to any two sites with arbitrary distance; For the holoQUADS circuit, the long-range gate is always applied to the first and last site.

using ITensors
using ITensors: orthocenter, sites, copy
using Base: Float64
using ITensors.HDF5
ITensors.disable_warn_order()
let
    n = 50
    h = 1.0
    tau = 0.1
    cutoff = 1E-8
    measurements = 20

    s = siteinds("S=1/2", n; conserve_qns = false)
    s1 = s[1]
    s2 = s[n]

    # Notice the difference in coefficients due to the system is half-infinite chain
    hj =
        π * op("Sz", s1) * op("Sz", s2) +
        2 * h * op("Sz", s1) * op("Id", s2) +
        h * op("Id", s1) * op("Sz", s2)
    Gj = exp(-1.0im * tau / 2 * hj)
    # @show hj
    # @show Gj
    @show inds(Gj)

    # Benchmark gate that employs swap operations
    benchmarkGate = ITensor[]
    push!(benchmarkGate, Gj)

    # for ind in 1 : n
    #     @show s[ind], s[ind]'
    # end

    U, S, V = svd(Gj, (s[1], s[1]'))
    @show norm(U * S * V - Gj)
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
    bondIndices[n-1] = inds(V)[3]

    # @show (bondIndices[1], bondIndices[n - 1])

    # longrangeGate = ITensor[]; push!(longrangeGate, U)
    # @show typeof(U), U
    # # @show sizeof(longrangeGate)
    # # @show longrangeGate
    # for ind in 2 : n - 1
    #     # Set up site indices
    #     if abs(ind - (n - 1)) > 1E-8
    #         bondString = "i" * string(ind)
    #         bondIndices[ind] = Index(4, bondString)
    #     end

    #     # Make the identity tensor
    #     # @show s[ind], s[ind]'
    #     tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]); # @show typeof(tmpIdentity)
    #     push!(longrangeGate, tmpIdentity)

    #     # @show sizeof(longrangeGate)
    #     # @show longrangeGate
    # end

    # push!(longrangeGate, V)
    # @show typeof(V), V
    # # @show sizeof(longrangeGate)
    # # @show longrangeGate

    # tmpGate = MPO(n)
    # for ind in 1 : n
    #     tmpGate[ind] = longrangeGate[ind]
    # end

    #####################################################################################################################################
    # Construct the long-range two-site gate as an MPO
    longrangeGate = MPO(n)
    longrangeGate[1] = U

    for ind = 2:n-1
        # Set up site indices
        if abs(ind - (n - 1)) > 1E-8
            bondString = "i" * string(ind)
            bondIndices[ind] = Index(4, bondString)
        end

        # Make the identity tensor
        # @show s[ind], s[ind]'
        tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind-1], bondIndices[ind])
        longrangeGate[ind] = tmpIdentity

        # @show sizeof(longrangeGate)
        # @show longrangeGate
    end

    @show typeof(V), V
    longrangeGate[n] = V
    # @show sizeof(longrangeGate)
    # @show longrangeGate
    @show typeof(longrangeGate), typeof(benchmarkGate)
    #####################################################################################################################################



    # Benchmark the accuracy of this implemented two-site gate. 
    Sx₁ = complex(zeros(measurements, n))
    Sx₂ = complex(zeros(measurements, n))
    Sy₁ = complex(zeros(measurements, n))
    Sy₂ = complex(zeros(measurements, n))
    Sz₁ = complex(zeros(measurements, n))
    Sz₂ = complex(zeros(measurements, n))
    overlap₁ = zeros(measurements)
    overlap₂ = zeros(measurements)
    overlap = zeros(measurements)

    # states = [isodd(tmpSite) ? "Up" : "Dn" for tmpSite = 1:n]
    # ψ = randomMPS(s, states, linkdims = 4)
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    ψ₁ = deepcopy(ψ)
    ψ₂ = deepcopy(ψ)

    for index = 1:measurements
        # ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

        # ψ_copy = deepcopy(ψ)
        @time ψ₁ = apply(longrangeGate, ψ₁; cutoff)
        Sx₁[index, :] = expect(ψ₁, "Sx"; sites = 1:n)
        Sy₁[index, :] = expect(ψ₁, "Sy"; sites = 1:n)
        Sz₁[index, :] = expect(ψ₁, "Sz"; sites = 1:n)

        # ψ_copy = deepcopy(ψ)
        @time ψ₂ = apply(benchmarkGate, ψ₂; cutoff)
        Sx₂[index, :] = expect(ψ₂, "Sx"; sites = 1:n)
        Sy₂[index, :] = expect(ψ₂, "Sy"; sites = 1:n)
        Sz₂[index, :] = expect(ψ₂, "Sz"; sites = 1:n)

        overlap[index] = abs(inner(ψ₁, ψ₂))
        @show abs(inner(ψ₁, ψ₂))
        overlap₁[index] = abs(inner(ψ₁, ψ))
        @show abs(inner(ψ₁, ψ))
        overlap₂[index] = abs(inner(ψ₂, ψ))
        @show abs(inner(ψ₂, ψ))
    end

    file = h5open("Data/Long_Range_Gate_Test_Time_Series_AF.h5", "w")
    write(file, "Sx1", Sx₁)
    write(file, "Sx2", Sx₂)
    write(file, "Sy1", Sy₁)
    write(file, "Sy2", Sy₂)
    write(file, "Sz1", Sz₁)
    write(file, "Sz2", Sz₂)
    write(file, "overlap", overlap)
    write(file, "overlap1", overlap₁)
    write(file, "overlap2", overlap₂)

    return
end
