## Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy
using Base: Float64
ITensors.disable_warn_order()
let 
    N = 8
    cutoff = 1E-8
    tau = 0.1
    ttotal = 10.0
    h = 10.0                                            # an integrability-breaking longitudinal field h 


    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);     # s = siteinds("S=1/2", N; conserve_qns = true)

    # Implement the function to generate one sample of the probability distirbution 
    # defined by squaring the components of the tensor
    function sample(m :: MPS)
        mpsLength = length(m)
        
        if orthocenter(m) != 1 
            error("sample: MPS m must have orthocenter(m) == 1")
        end
        if abs(1.0 - norm(m[1])) > 1E-8
            error("sample: MPS is not normalized, norm=$(norm(m[1]))")
        end

        result = zeros(Int, mpsLength)
        A = m[1]

        for ind in 1:mpsLength
            s = siteind(m, ind)
            d = dim(s)
            pdisc = 0.0
            r = rand()

            n = 1
            An = ITensor()
            pn = 0.0

            while n <= d
                projn = ITensor(s)
                projn[s => n] = 1.0
                An = A * dag(projn)
                pn = real(scalar(dag(An) * An))
                pdisc += pn

                (r < pdisc) && break
                n += 1
            end
            result[ind] = n
            
            if ind < mpsLength
                A = m[ind + 1] * An
                A *= (1. / sqrt(pn))
            end
        end
    end

    
    # Construct the gate for the Ising model with longitudinal field
    gates = ITensor[]
    for ind in 1:(N - 1)
        s1 = s[ind]
        s2 = s[ind + 1]
        hj = π * op("Sz", s1) * op("Sz", s2) + 2 * h * op("Sz", s1) * op("Id", s2) 
        #     # + 2 * h * op("Sz", s2) * op("Id", s1)
        # println(typeof(hj))
        Gj = exp(-1.0im * tau / 2 * hj)
        push!(gates, Gj)
    end
    
    # Add the last site using single-site operator
    hn = 2 * h * op("Sz", s[N])
    Gn = exp(-1.0im * tau / 2 * hn)
    push!(gates, Gn)

    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(gates, reverse(gates))
    # @show size(gates)


    # # Construct the gate for the Ising model with longitudinal and transverse fields
    # kickGates = ITensor[]
    # for ind in 1:(N - 1)
    #     s1 = s[ind]
    #     s2 = s[ind + 1]
    #     tmpH = π  * op("Sz", s1) * op("Sz", s2)
    #         + 2 * h * op("Sz", s1) * op("I", s2) 
    #         # + 2 * h * op("Sz", s2) * op("Id", s1)
    #         + π / 2 * (1 / (2 * tau)) * op("Sx", s1) * op("I", s2)
    #         # + π / 2 * op("Sx", s2) * op("Id", s1)
    #     tmpG = exp(-1.0im * tau / 2 * tmpH)
    #     push!(kickGates, tmpG)
    # end

    # hn = 2 * h * op("Sz", s[N]) + π / 2 * (1 / (2 * tau)) * op("Sx", s[N])
    # Gn = exp(-1.0im * tau / 2 * hn)
    # push!(kickGates, Gn)

    # # Append the reverse gates (N - 1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    # append!(kickGates, reverse(kickGates))

    # Construct the kicked gate similar to the ED code 
    ampo = OpSum()
    for ind in 1 : N
        ampo += π/2, "Sx", ind
    end
    Hₓ = MPO(ampo, s)
    Hamiltonianₓ = Hₓ[1]
    for ind in 2 : N
        Hamiltonianₓ *= Hₓ[ind]
    end
    expHamiltoininaₓ = exp(-1.0im * Hamiltonianₓ)
    
    # An alternative approach to add kicked fields. 
    # Seems incorrect since the two parts of the original Hamiltonian will have different time steps
    # Construct the gate for the transverse Ising field applied only at integer time
    # kickGates = ITensor[]
    # for ind in 1:N
    #     s1 = s[ind]
    #     hamilt = π / 4 * op("Sx", s1)
    #     tmpG = exp(-im * hamilt)
    #     push!(kickGates, tmpG)
    # end
    
    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # @show maxlinkdim(ψ)

    # Locate the central site
    centralSite = div(N, 2)

    # Compute <Sz> at rach time step and apply gates to go to the next step
    ψ_copy = copy(ψ)
    Sz = zeros(101, N); Sz = complex(Sz)
    # Sz = Complex{Float64}[]
    ψ_overlap = Complex{Float64}[]
    Czz = zeros(101,  N^2); Czz = complex(Czz)
    index = 1
    
    @time for time in 0.0:tau:ttotal
        tmp_overlap = abs(inner(ψ, ψ_copy))
        println("The inner product is: $tmp_overlap")
        append!(ψ_overlap, tmp_overlap)
        
        time ≈ ttotal && break
        # if (abs((time / tau) % 10) < 1E-8 || abs((time + tau) / tau % 10) < 1E-8)
        if (abs((time / tau) % 10) < 1E-8)
            println("At time $time, applying the kicked fields")
            # ψ = apply(kickGates, ψ; cutoff = cutoff)
            ψ = apply(expHamiltoininaₓ, ψ; cutoff)
        else
            ψ = apply(gates, ψ; cutoff = cutoff)
        end
        normalize!(ψ)

        # tmpSz = expect(ψ, "Sz"; sites = centralSite)
        tmpSz = expect(ψ, "Sz"; sites = 1 : N)
        Sz[index, :] = tmpSz
        # append!(Sz, tmpSz)

        tmpCzz = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
        Czz[index, :] = vec(tmpCzz')
        # Czz[index, :] = tmpCzz[1, :]
        index += 1
    
        # println("At time step $time, Sz is $tmpSz")
        # println("At time step $time, Czz is $(tmpCzz[1, :])")
        # @show size(Sz); @show typeof(Sz)
        # @show size(tmpCzz); @show typeof(tmpCzz)
        @show size(Czz); @show typeof(Czz)


        # tmp_overlap = abs(inner(ψ, ψ_copy))
        # append!(ψ_overlap, tmp_overlap)
        # println("The inner product is: $tmp_overlap")
    end

    # Store data into a hdf5 file
    file = h5open("TEBD_N$(N)_h$(h)_Info.h5", "w")
    write(file, "Sz", Sz)
    write(file, "Czz", Czz)
    write(file, "Wavefunction Overlap", ψ_overlap)
    close(file)
    
    return
end 