#@ 11/28/2022
## Implement the quantum circuit for the SDKI model using classical MPS.

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex
using Base: Float64
using Base: product
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

    projn0_Matrix = [1  0; 0  0]
    projnLower_Matrix = [0  0; 1  0]
    # @show projectionMatrix, sizeof(projectionMatrix)
    result = zeros(Int, 2)
    A = m[j]
    # @show A
    # @show m[j]

    for ind in j:j+1
        tmpS = siteind(m, ind)
        d = dim(tmpS)
        pdisc = 0.0
        r = rand()

        n = 1
        An = ITensor()
        pn = 0.0

        while n <= d
            # @show A
            # @show m[ind]
            projn = ITensor(tmpS)
            projn[tmpS => n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind - j + 1] = n
        # @show result[ind - j + 1]
        # @show An

        if ind < mpsLength
            A = m[ind + 1] * An
            A *= (1. / sqrt(pn))
        end

        # @show m[ind]
        if n - 1 < 1E-8
            # tmpReset = ITensor(projn0_Matrix, s, s')
            tmpReset = ITensor(projn0_Matrix, tmpS, tmpS')
        else
            # tmpReset = ITensor(projnLower_Matrix, s, s')
            tmpReset = ITensor(projnLower_Matrix, tmpS, tmpS')
        end
        m[ind] *= tmpReset
        noprime!(m[ind])
        # @show m[ind]
    end
end 

# # Implement a long-range two-site gate
# function long_range_gate(tmp_s, position_index::Int)
#     s1 = tmp_s[1]
#     s2 = tmp_s[position_index]
    
#     # Notice the difference in coefficients due to the system is half-infinite chain
#     hj = π * op("Sz", s1) * op("Sz", s2) + 2 * h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
#     Gj = exp(-1.0im * tau / 2 * hj)
#     # @show hj
#     # @show Gj
#     @show inds(Gj)

#     # Benchmark gate that employs swap operations
#     benchmarkGate = ITensor[]
#     push!(benchmarkGate, Gj)
    
#     # for ind in 1 : n
#     #     @show s[ind], s[ind]'
#     # end

#     U, S, V = svd(Gj, (tmp_s[1], tmp_s[1]'))
#     @show norm(U*S*V - Gj)
#     # @show S
#     # @show U
#     # @show V

#     # Absorb the S matrix into the U matrix on the left
#     U = U * S
#     # @show U

#     # Make a vector to store the bond indices
#     bondIndices = Vector(undef, n - 1)

#     # Grab the bond indices of U and V matrices
#     if hastags(inds(U)[3], "Link,v") != true           # The original tag of this index of U matrix should be "Link,u".  But we absorbed S matrix into the U matrix.
#         error("SVD: fail to grab the bond indice of matrix U by its tag!")
#     else 
#         replacetags!(U, "Link,v", "i1")
#     end
#     # @show U
#     bondIndices[1] = inds(U)[3]

#     if hastags(inds(V)[3], "Link,v") != true
#         error("SVD: fail to grab the bond indice of matrix V by its tag!")
#     else
#         replacetags!(V, "Link,v", "i" * string(n))
#     end
#     # @show V
#     @show position_index
#     bondIndices[position_index - 1] = inds(V)[3]
#     # @show (bondIndices[1], bondIndices[n - 1])

    

#     #####################################################################################################################################
#     # Construct the long-range two-site gate as an MPO
#     longrangeGate = MPO(n)
#     longrangeGate[1] = U

#     for ind in 2 : position_index - 1
#         # Set up site indices
#         if abs(ind - (position_index - 1)) > 1E-8
#             bondString = "i" * string(ind)
#             bondIndices[ind] = Index(4, bondString)
#         end

#         # Make the identity tensor
#         # @show s[ind], s[ind]'
#         tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]) 
#         longrangeGate[ind] = tmpIdentity

#         # @show sizeof(longrangeGate)
#         # @show longrangeGate
#     end

#     @show typeof(V), V
#     longrangeGate[position_index] = V
#     # @show sizeof(longrangeGate)
#     # @show longrangeGate
#     @show typeof(longrangeGate), typeof(benchmarkGate)
#     #####################################################################################################################################
# end

let 
    N = 10
    cutoff = 1E-8
    tau = 0.5
    ttotal = 4.0                                       # floquet time = 1 / 2 * circuit time 
    h = 10.0                                           # an integrability-breaking longitudinal field h 
    time_step = 8
    max_gates = 4
    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);    # s = siteinds("S=1/2", N; conserve_qns = true)

    # Implement a long-range two-site gate
    function long_range_gate(tmp_s, position_index::Int)
        s1 = tmp_s[1]
        s2 = tmp_s[position_index]
        
        # Notice the difference in coefficients due to the system is half-infinite chain
        hj = π * op("Sz", s1) * op("Sz", s2) + 2 * h * op("Sz", s1) * op("Id", s2) + h * op("Id", s1) * op("Sz", s2)
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

        U, S, V = svd(Gj, (tmp_s[1], tmp_s[1]'))
        @show norm(U*S*V - Gj)
        # @show S
        # @show U
        # @show V

        # Absorb the S matrix into the U matrix on the left
        U = U * S
        # @show U

        # Make a vector to store the bond indices
        bondIndices = Vector(undef, position_index - 1)

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
            replacetags!(V, "Link,v", "i" * string(position_index))
        end
        # @show V
        @show position_index
        bondIndices[position_index - 1] = inds(V)[3]
        # @show (bondIndices[1], bondIndices[n - 1])

        

        #####################################################################################################################################
        # Construct the long-range two-site gate as an MPO
        longrangeGate = MPO(position_index)
        longrangeGate[1] = U

        for ind in 2 : position_index - 1
            # Set up site indices
            if abs(ind - (position_index - 1)) > 1E-8
                bondString = "i" * string(ind)
                bondIndices[ind] = Index(4, bondString)
            end

            # Make the identity tensor
            # @show s[ind], s[ind]'
            tmpIdentity = delta(s[ind], s[ind]') * delta(bondIndices[ind - 1], bondIndices[ind]) 
            longrangeGate[ind] = tmpIdentity

            # @show sizeof(longrangeGate)
            # @show longrangeGate
        end

        @show typeof(V), V
        longrangeGate[position_index] = V
        # @show sizeof(longrangeGate)
        # @show longrangeGate
        @show typeof(longrangeGate), typeof(benchmarkGate)
        #####################################################################################################################################
        return longrangeGate
    end
    
    
    ###############################################################################################################################
    ## Constructing gates used in the TEBD algorithm
    ###############################################################################################################################
    # # Construct a two-site gate that implements Ising interaction and longitudinal field
    # gates = ITensor[]
    # for ind in 1:(N - 1)
    #     s1 = s[ind]
    #     s2 = s[ind + 1]

    #     if (ind - 1 < 1E-8)
    #         tmp1 = 2 
    #         tmp2 = 1
    #     elseif (abs(ind - (N - 1)) < 1E-8)
    #         tmp1 = 1
    #         tmp2 = 2
    #     else
    #         tmp1 = 1
    #         tmp2 = 1
    #     end

    #     println("")
    #     println("Coefficients are $(tmp1) and $(tmp2)")
    #     println("Site index is $(ind) and the conditional sentence is $(ind - (N - 1))")
    #     println("")

    #     hj = π * op("Sz", s1) * op("Sz", s2) + tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2)
    #     Gj = exp(-1.0im * tau / 2 * hj)
    #     push!(gates, Gj)
    # end
    # # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    # append!(gates, reverse(gates))

    # Construct the kicked gate that applies transverse Ising fields at integer time using single-site gate
    kick_gate = ITensor[]
    for ind in 1 : N
        s1 = s[ind]
        hamilt = π / 2 * op("Sx", s1)
        tmpG = exp(-1.0im * hamilt)
        push!(kick_gate, tmpG)
    end

    # # Construct the transverse field as the kicked gate
    # ampo = OpSum()
    # for ind in 1 : N
    #     ampo += π/2, "Sx", ind
    # end
    # Hₓ = MPO(ampo, s)
    # Hamiltonianₓ = Hₓ[1]
    # for ind in 2 : N
    #     Hamiltonianₓ *= Hₓ[ind]
    # end
    # expHamiltoinianₓ = exp(-1.0im * Hamiltonianₓ)
    

    #################################################################################################################################################
    # Construct the holographic quantum dynamics simulation (holoQUADS) circuit
    #################################################################################################################################################
    # Construct time evolution for one floquet time step
    # function timeEvolutionCorner(numGates :: Int, numSites :: Int, tmp_gates)
    #     # In the corner case, two-site gates are applied to site 1 --> site N
    #     # gates = ITensor[]
    #     if 2 * numGates >= numSites
    #         error("the number of time evolution gates is larger than what can be accommodated based on the number of sites!")
    #     end

    #     for ind₁ in 1:2
    #         for ind₂ in 1:numGates
    #             parity = (ind₁ - 1) % 2
    #             s1 = s[2 * ind₂ - parity]; @show inds(s1)
    #             s2 = s[2 * ind₂ + 1 - parity]; @show inds(s2)

    #             if 2 * ind₂ - parity - 1 < 1E-8
    #                 coeff₁ = 2
    #                 coeff₂ = 1
    #             else
    #                 coeff₁ = 1
    #                 coeff₂ = 1
    #             end

    #             hj = (π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2))
    #             Gj = exp(-1.0im * tau / 2 * hj)
    #             push!(tmp_gates, Gj)
    #         end
    #     end
    #     # return gates
    # end


    # Construct time evolution for one floquet time step
    function time_evolution_corner(num_gates :: Int, parity :: Int)
        # In the corner case, two-site gates are applied to site 1 --> site N
        gates = ITensor[]

        for ind₁ in 1 : num_gates
            s1 = s[2 * ind₁ - parity]
            s2 = s[2 * ind₁ + 1 - parity]
            @show inds(s1)
            @show inds(s2)

            if 2 * ind₁ - parity - 1 < 1E-8
                coeff₁ = 2
                coeff₂ = 1
            else
                coeff₁ = 1
                coeff₂ = 1
            end

            hj = (π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2))
            Gj = exp(-1.0im * tau / 2 * hj)
            push!(gates, Gj)
        end
        return gates
    end

    function time_evolution(initialPosition :: Int, numSites :: Int, tmp_sites)
        # In the general case, two-site gates are applied to selected site with inverse order (N --> 1)
        gates = ITensor[]
        if initialPosition - 1 < 1E-8
            tmpGate = long_range_gate(tmp_sites, numSites)
            push!(gates, tmpGate)
        else
            if initialPosition - 2 < 1E-8
                coeff₁ = 1
                coeff₂ = 2 
            else
                coeff₁ = 1
                coeff₂ = 1
            end
            s1 = tmp_sites[initialPosition]
            s2 = tmp_sites[initialPosition - 1]
            hj = (π * op("Sz", s1) * op("Sz", s2) + coeff₁ * h * op("Sz", s1) * op("Id", s2) + coeff₂ * h * op("Id", s1) * op("Sz", s2))
            Gj = exp(-1.0im * tau / 2 * hj)
            push!(gates, Gj)
        end
        return gates
    end

    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # @show eltype(ψ), eltype(ψ[1])
    # states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # @show maxlinkdim(ψ)

    # Locate the central site
    # centralSite = div(N, 2)

    # Compute the overlap between the original and time evolved wavefunctions
    ψ_copy = deepcopy(ψ)
    ψ_overlap = Complex{Float64}[]

    # Compute local observables e.g. Sz, Czz 
    # timeSlices = Int(ttotal / tau) + 1; println("Total number of time slices that need to be saved is : $(timeSlices)")
    # Sx = complex(zeros(timeSlices, N))
    # Sy = complex(zeros(timeSlices, N))
    # Sz = complex(zeros(timeSlices, N))
    # Cxx = complex(zeros(timeSlices, N))
    # Czz = complex(zeros(timeSlices, N))
    # index = 1
    

    options = [4, 4, 3, 3, 2, 2, 1, 1]
    odd_even = [0, 1, 0, 1, 0, 1, 0, 1]
    @time for ind in 1 : time_step
        tmp_overlap = abs(inner(ψ, ψ_copy))
        println("The inner product is: $tmp_overlap")
        append!(ψ_overlap, tmp_overlap)

        # Local observables e.g. Sx, Sz
        tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); @show tmpSx; # Sx[index, :] = tmpSx
        tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); @show tmpSy; # Sy[index, :] = tmpSy
        tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); @show tmpSz; # Sz[index, :] = tmpSz

        # Apply kicked gate at integer times
        if ind % 2 == 1
            ψ_copy = apply(kick_gate, ψ_copy; cutoff)
            normalize!(ψ_copy)
            println("")
            tmp_overlap = abs(inner(ψ, ψ_copy))
            @show tmp_overlap
            println("")
        end

        # Apply a sequence of two-site gates
        tmp_two_site_gates = ITensor[]
        tmp_two_site_gates = time_evolution_corner(options[ind], odd_even[ind])
        # println("")
        # @show tmp_two_site_gates 
        # @show typeof(tmp_two_site_gates)
        # println("")

        ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
        println("")
        tmp_overlap = abs(inner(ψ, ψ_copy))
        @show tmp_overlap
        println("")
    end
    
    sample(ψ_copy, 1)
    println("")
    tmp_overlap = abs(inner(ψ, ψ_copy))
    @show tmp_overlap
    println("")

    options = [1, 10, 9, 8, 7, 6, 5, 4]
    @time for ind in 1 : time_step
        tmp_overlap = abs(inner(ψ, ψ_copy))
        println("The inner product is: $tmp_overlap")
        append!(ψ_overlap, tmp_overlap)

        # Local observables e.g. Sx, Sz
        tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); @show tmpSx; # Sx[index, :] = tmpSx
        tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); @show tmpSy; # Sy[index, :] = tmpSy
        tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); @show tmpSz; # Sz[index, :] = tmpSz

        # Apply kicked gate at integer times
        if ind % 2 == 1
            ψ_copy = apply(kick_gate, ψ_copy; cutoff)
            normalize!(ψ_copy)
            println("")
            tmp_overlap = abs(inner(ψ, ψ_copy))
            @show tmp_overlap
            println("")
        end

        # Apply a sequence of two-site gates
        tmp_two_site_gates = ITensor[]
        tmp_two_site_gates = time_evolution(options[ind], N, s)
        # println("")
        # @show tmp_two_site_gates 
        # @show typeof(tmp_two_site_gates)
        # println("")

        ψ_copy = apply(tmp_two_site_gates, ψ_copy; cutoff)
        println("")
        tmp_overlap = abs(inner(ψ, ψ_copy))
        @show tmp_overlap
        println("")
    end
    sample(ψ_copy, 3); @show abs(inner(ψ, ψ_copy))
    
    
    # @time for time in 0.0:tau:ttotal
    #     tmp_overlap = abs(inner(ψ, ψ_copy))
    #     println("The inner product is: $tmp_overlap")
    #     append!(ψ_overlap, tmp_overlap)

    #     # Local observables e.g. Sx, Sz
    #     tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); @show tmpSx; # Sx[index, :] = tmpSx
    #     tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); @show tmpSy; # Sy[index, :] = tmpSy
    #     tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); @show tmpSz; # Sz[index, :] = tmpSz

    #     # Spin correlaiton functions e.g. Cxx, Czz
    #     # tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N);  Cxx[index, :] = tmpCxx[4, :]
    #     # tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N);  Czz[index, :] = tmpCzz[4, :]
    #     # # Czz[index, :] = vec(tmpCzz')
    #     # index += 1
    #     # tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1, N); @show tmpCxx
    #     # tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1, N); @show tmpCzz
        
    #     # time ≈ ttotal && break
    #     time > ttotal && break
    #     @show Int(time / tau), time, tau

    #     # Real-time evolution corner case
    #     for ind in 1 : time_step
    #         if ind % 2 == 1
    #             println("")
    #             println("Apply the kicked gates at integer time $time")
    #             println("")
    #             ψ_copy = apply(kick_gate, ψ_copy; cutoff)
    #             normalize!(ψ_copy)
    #         end
    #         println("")
    #         @show ind // 2
    #         println("")
    #     end

    #     # if Int(time / tau) % 2 == 0
    #     #     println("")
    #     #     println("Apply the kicked gates at integer time $time")
    #     #     println("")
    #     #     ψ_copy = apply(kick_gate, ψ_copy; cutoff)
    #     #     normalize!(ψ_copy)
    #     # end

    #     # if (abs((time / tau) % 10) < 1E-8)
    #     #     println("")
    #     #     println("Apply the kicked gates at integer time $time")
    #     #     println("")
    #     #     # ψ_copy = apply(expHamiltoinianₓ, ψ_copy; cutoff)
    #     #     ψ_copy = apply(kick_gate, ψ_copy; cutoff)
    #     #     normalize!(ψ_copy)
    #     # end

    #     # ψ_copy = apply(gates, ψ_copy; cutoff)
    #     # normalize!(ψ_copy)

    #     #********************************************************************************
    #     # Previous ideas of using a square wave to represent a delta function
    #     #********************************************************************************
    #     # if (abs((time / tau) % 10) < 1E-8)
    #     #     # ψ = apply(kickGates, ψ; cutoff = cutoff)
    #     #     ψ = apply(expHamiltoininaₓ, ψ; cutoff)
    #     # # else
    #     # #     ψ = apply(gates, ψ; cutoff)
    #     # end

    #     # tmp_overlap = abs(inner(ψ, ψ_copy))
    #     # append!(ψ_overlap, tmp_overlap)
    #     # println("The inner product is: $tmp_overlap")
    # end

    # # Store data into a hdf5 file
    # file = h5open("RawData/TEBD_N$(N)_h$(h)_Info.h5", "w")
    # write(file, "Sx", Sx)
    # write(file, "Sz", Sz)
    # write(file, "Cxx", Cxx)
    # write(file, "Czz", Czz)
    # write(file, "Wavefunction Overlap", ψ_overlap)
    # close(file)
    
    return
end 