## 03/29/2023
## Implement time evolution block decimation (TEBD) using a brick wall pattern

using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy, complex
using Base: Float64, Real
using Random
ITensors.disable_warn_order()




function build_a_layer_of_gates(starting_index :: Int, ending_index :: Int, upper_bound :: Int, 
    amplitude :: Real, delta_tau :: Real, tmp_sites)
    tmp_gates = []
    for ind in starting_index : 2 : ending_index
        s1 = tmp_sites[ind]
        s2 = tmp_sites[ind + 1]

        if (ind - 1 < 1E-8)
            tmp1 = 2 
            tmp2 = 1
        elseif (abs(ind - (upper_bound - 1)) < 1E-8)
            tmp1 = 1
            tmp2 = 2
        else
            tmp1 = 1
            tmp2 = 1
        end

        # hj = tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2)
        hj = π/2 * op("Sz", s1) * op("Sz", s2) + tmp1 * amplitude * op("Sz", s1) * op("Id", s2) + tmp2 * amplitude * op("Id", s1) * op("Sz", s2)
        # hj = π * op("Sz", s1) * op("Sz", s2) + tmp1 * amplitude * op("Sz", s1) * op("Id", s2) + tmp2 * amplitude * op("Id", s1) * op("Sz", s2) 
        Gj = exp(-1.0im * delta_tau * hj)
        push!(tmp_gates, Gj)
    end
    return tmp_gates
end


# 04/17/2023
# Implement a function to compute the von Neumann entanglement and monitor its time dependence
function compute_entropy(input_matrix)
    local tmpEntropy = 0
    for index in 1 : size(input_matrix, 1) 
        # entropy += -2 * input_matrix[index, index]^2 * log(input_matrix[index, index])
        tmp = input_matrix[index, index]^2
        tmpEntropy += -tmp * log(tmp)
    end
    return tmpEntropy
end


let 
    N = 12
    cutoff = 1E-8
    Δτ = 0.1; ttotal = 7.0
    h = 0.2                                            # an integrability-breaking longitudinal field h 

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);     # s = siteinds("S=1/2", N; conserve_qns = true)

    # Construct layers of gates used in TEBD for the kicked Ising model
    gates = ITensor[]
    even_layer = build_a_layer_of_gates(2, N-2, N, h, Δτ, s)
    for tmp₁ in even_layer
        push!(gates, tmp₁)
    end

    odd_layer = build_a_layer_of_gates(1, N-1, N, h, Δτ, s)
    for tmp₂ in odd_layer
        push!(gates, tmp₂)
    end

    # Construct the kicked gate that are only applied at integer time
    kick_gates = ITensor[]
    for ind in 1:N
        s1 = s[ind]
        hamilt = π / 2 * op("Sx", s1)
        tmpG = exp(-1.0im * hamilt)
        push!(kick_gates, tmpG)
    end
    
    # Initialize the wavefunction as a Neel state
    # ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # states = [isodd(n) ? "+" : "-" for n = 1 : N]
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    ψ = MPS(s, states)
    ψ_copy = deepcopy(ψ)
    ψ_overlap = Complex{Float64}[]

    # # Intialize the wvaefunction as a random MPS   
    # states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    # Random.seed!(87900)
    # ψ = randomMPS(s, states, linkdims = 2)
    # # @show eltype(ψ), eltype(ψ[1])
    # # @show maxlinkdim(ψ)
    # ψ_copy = deepcopy(ψ)
    # ψ_overlap = Complex{Float64}[]

    # Take a measurement of the initial random MPS to make sure the same random MPS is used through all codes.
    Sx₀ = expect(ψ_copy, "Sx"; sites = 1 : N)
    Sy₀ = expect(ψ_copy, "Sy"; sites = 1 : N)
    Sz₀ = expect(ψ_copy, "Sz"; sites = 1 : N)

    # Compute local observables e.g. Sz, Czz 
    timeSlices = Int(ttotal / Δτ) + 1; println("Total number of time slices that need to be saved is : $(timeSlices)")
    Sx = complex(zeros(timeSlices, N))
    Sy = complex(zeros(timeSlices, N))
    Sz = complex(zeros(timeSlices, N))
    Cxx = complex(zeros(timeSlices, N * N)) 
    Cyy = complex(zeros(timeSlices, N * N))
    Czz = complex(zeros(timeSlices, N * N))
    entropy = complex(zeros(timeSlices, N - 1))

    # Take measurements of the initial setting before time evolution
    # tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); Sx[1, :] = tmpSx
    # tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); Sy[1, :] = tmpSy
    tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); Sz[1, :] = tmpSz

    # tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N); Cxx[1, :] = tmpCxx[:]
    # tmpCyy = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N); Cyy[1, :] = tmpCyy[:]
    tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N); Czz[1, :] = tmpCzz[:]
    append!(ψ_overlap, abs(inner(ψ, ψ_copy)))

    distance = Int(1.0 / Δτ); index = 2
    @time for time in 0.0 : Δτ : ttotal
        time ≈ ttotal && break
        println("")
        println("")
        @show time
        println("")
        println("")

        for site_index in 2 : N - 1 
            orthogonalize!(ψ_copy, site_index)

            i₀, j₀ = inds(ψ_copy[site_index])[1], inds(ψ_copy[site_index])[3]
            # i₀, j₀ = siteind(ψ, site_index), linkind(ψ, site_index - 1)
            _, C0, _ = svd(ψ_copy[site_index], i₀, j₀)
            C0 = matrix(C0)
            SvN = compute_entropy(C0)

            i₁, j₁ = siteind(ψ_copy, site_index), linkind(ψ_copy, site_index - 1)
            _, C1, _ = svd(ψ_copy[site_index], i₁, j₁)
            C1 = matrix(C1)
            SvN₁ = compute_entropy(C1)
            
            @show dim(linkind(ψ_copy, site_index - 1))
            @show site_index, SvN, SvN₁
            entropy[index - 1, site_index - 1] = SvN₁
        end

        if (abs((time / Δτ) % distance) < 1E-8)
            println("")
            println("Apply the kicked gates at integer time $time")
            println("")
            ψ_copy = apply(kick_gates, ψ_copy; cutoff)
            normalize!(ψ_copy)
            append!(ψ_overlap, abs(inner(ψ, ψ_copy)))

            # println("")
            # println("")
            # tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); @show tmpSx
            # tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); @show tmpSy
            # tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); @show tmpSz
            # println("")
            # println("")
        end

        ψ_copy = apply(gates, ψ_copy; cutoff)
        normalize!(ψ_copy)

        # Local observables e.g. Sx, Sz
        tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); Sx[index, :] = tmpSx; # @show tmpSx
        tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); Sy[index, :] = tmpSy; # @show tmpSy
        tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); Sz[index, :] = tmpSz; # @show tmpSz

        # Spin correlaiton functions e.g. Cxx, Czz
        # tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N);  Cxx[index, :] = tmpCxx[:]
        # tmpCyy = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N);  Cyy[index, :] = tmpCyy[:]
        tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N);  Czz[index, :] = tmpCzz[:]
        # @show tmpCzz[1, :], tmpCzz[2, :]

        # Czz[index, :] = vec(tmpCzz')
        index += 1

        tmp_overlap = abs(inner(ψ, ψ_copy))
        println("The inner product is: $tmp_overlap")
        append!(ψ_overlap, tmp_overlap)
    end

    println("################################################################################")
    println("################################################################################")
    println("Projective measurements of the initial MPS in the Sz basis")
    @show Sz₀
    println("################################################################################")
    println("################################################################################")

    # Store data into a hdf5 file
    # file = h5open("Data/TEBD_N$(N)_h$(h)_tau$(tau)_Longitudinal_Only_Random_QN_Link2.h5", "w")
    file = h5open("TEBD_N$(N)_h$(h)_tau$(Δτ)_T$(ttotal)_AFM.h5", "w")
    write(file, "Sx", Sx)
    write(file, "Sy", Sy)
    write(file, "Sz", Sz)
    write(file, "Cxx", Cxx)
    write(file, "Cyy", Cyy)
    write(file, "Czz", Czz)
    write(file, "Wavefunction Overlap", ψ_overlap)
    write(file, "Entropy", entropy)
    write(file, "Initial Sx", Sx₀)
    write(file, "Initial Sy", Sy₀)
    write(file, "Initial Sz", Sz₀)
    close(file)
    
    return
end  