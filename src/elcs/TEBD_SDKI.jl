#
# Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model

using ITensors
using ITensorMPS
# using ITensors: orthocenter, sites, copy, complex
# using Base: Float64
# using Random
# using ITensors.HDF5
using HDF5

ITensors.disable_warn_order()


# Implement the function to generate one sample of the probability distirbution 
# defined by squaring the components of the tensor
function sample(m::MPS)
    mpsLength = length(m)

    if orthocenter(m) != 1
        error("sample: MPS m must have orthocenter(m) == 1")
    end
    if abs(1.0 - norm(m[1])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[1]))")
    end

    result = zeros(Int, mpsLength)
    A = m[1]

    for ind = 1:mpsLength
        s = siteind(m, ind)
        d = dim(s)
        pdisc = 0.0
        r = rand()

        n = 1
        An = ITensor()
        pn = 0.0

        while n <= d
            projn = ITensor(s)
            projn[s=>n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind] = n

        if ind < mpsLength
            A = m[ind+1] * An
            A *= (1.0 / sqrt(pn))
        end
    end
end


let
    N = 32
    cutoff = 1E-8
    tau = 0.1
    ttotal = 3.0
    h = 0.2                                            # an integrability-breaking longitudinal field h 

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false)
    
    # Intialize the wavefunction as a random MPS
    states = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    Random.seed!(123)
    ψ = randomMPS(s, states, linkdims = 2)
    ψ_copy = deepcopy(ψ)
    ψ_overlap = Complex{Float64}[]

    # Construct the gate for the Ising model with longitudinal field
    gates = ITensor[]
    for ind = 1:(N-1)
        s1 = s[ind]
        s2 = s[ind+1]

        if (ind - 1 < 1E-8)
            tmp1 = 2
            tmp2 = 1
        elseif (abs(ind - (N - 1)) < 1E-8)
            tmp1 = 1
            tmp2 = 2
            # tmp2 = 1                # TO COMPARE WITH A HALF-INFINITE CHAIN
        else
            tmp1 = 1
            tmp2 = 1
        end

        # println("")
        # println("Coefficients are $(tmp1) and $(tmp2)")
        # println("Site index is $(ind) and the conditional sentence is $(ind - (N - 1))")
        # println("")

        hj =
            π * op("Sz", s1) * op("Sz", s2) +
            tmp1 * h * op("Sz", s1) * op("Id", s2) +
            tmp2 * h * op("Id", s1) * op("Sz", s2)
        # hj = tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * tau / 2 * hj)
        push!(gates, Gj)
    end
    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(gates, reverse(gates))

    # @show gates
    # @show size(gates)
    # @show reverse(gates)

    # Construct the kicked gate that are only applied at integer time
    kickGates = ITensor[]
    for ind = 1:N
        s1 = s[ind]
        hamilt = π / 2 * op("Sx", s1)
        tmpG = exp(-1.0im * hamilt)
        push!(kickGates, tmpG)
    end

    # Initialize the wavefunction as a Neel state
    # ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    # ψ = MPS(s, states)
    # ψ_copy = deepcopy(ψ)
    # ψ_overlap = Complex{Float64}[]


    # Initialize the random MPS by reading in from a file
    # wavefunction_file = h5open("random_MPS.h5", "r")
    # ψ = read(wavefunction_file, "Psi", MPS)
    # close(wavefunction_file)
    # ψ_copy = deepcopy(ψ)
    # ψ_overlap = Complex{Float64}[]

    # Take a measurement of the initial random MPS to make sure the same random MPS is used through all codes.
    Sz₀ = expect(ψ_copy, "Sz"; sites = 1:N)

    # Compute local observables e.g. Sz, Czz 
    time_slices = Int(round(ttotal / tau)) + 1
    println("Total number of time slices to be saved: $time_slices")

    # Preallocate arrays for observables and correlations
    Sx  = Matrix{ComplexF64}(undef, time_slices, N); fill!(Sx, 0)
    Sy  = Matrix{ComplexF64}(undef, time_slices, N); fill!(Sy, 0)
    Sz  = Matrix{ComplexF64}(undef, time_slices, N); fill!(Sz, 0)
    Cxx = Matrix{ComplexF64}(undef, time_slices, N * N); fill!(Cxx, 0)
    Cyy = Matrix{ComplexF64}(undef, time_slices, N * N); fill!(Cyy, 0)
    Czz = Matrix{ComplexF64}(undef, time_slices, N * N); fill!(Czz, 0)

    # Take measurements of the initial setting before time evolution
    # tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); Sx[1, :] = tmpSx
    # tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); Sy[1, :] = tmpSy
    tmpSz = expect(ψ_copy, "Sz"; sites = 1:N)
    Sz[1, :] = tmpSz

    # tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N); Cxx[1, :] = tmpCxx[:]
    # tmpCyy = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N); Cyy[1, :] = tmpCyy[:]
    tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1:N)
    Czz[1, :] = tmpCzz[:]
    append!(ψ_overlap, abs(inner(ψ, ψ_copy)))

    distance = Int(1.0 / tau)
    index = 2
    @time for time = 0.0:tau:ttotal
        time ≈ ttotal && break
        println("")
        println("")
        @show time
        println("")
        println("")

        if (abs((time / tau) % distance) < 1E-8)
            println("")
            println("Apply the kicked gates at integer time $time")
            println("")
            ψ_copy = apply(kickGates, ψ_copy; cutoff)
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
        # tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); Sx[index, :] = tmpSx; @show tmpSx
        # tmpSy = expect(ψ_copy, "Sy"; sites = 1 : N); Sy[index, :] = tmpSy; @show tmpSy
        tmpSz = expect(ψ_copy, "Sz"; sites = 1:N)
        Sz[index, :] = tmpSz
        @show tmpSz

        # Spin correlaiton functions e.g. Cxx, Czz
        # tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N);  Cxx[index, :] = tmpCxx[:]
        # tmpCyy = correlation_matrix(ψ_copy, "Sy", "Sy"; sites = 1 : N);  Cyy[index, :] = tmpCyy[:]
        tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1:N)
        Czz[index, :] = tmpCzz[:]
        @show tmpCzz[1, :], tmpCzz[2, :]

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
    h5open("../data/rdm_sample/SDKI_TEBD_N$(N)_h$(h)_t$(ttotal).h5", "w") do file
        write(file, "Sx", Sx)
        write(file, "Sy", Sy)
        write(file, "Sz", Sz)
        write(file, "Cxx", Cxx)
        write(file, "Cyy", Cyy)
        write(file, "Czz", Czz)
        write(file, "Wavefunction Overlap", ψ_overlap)
        write(file, "Sz0", Sz₀)
    end

    return
end