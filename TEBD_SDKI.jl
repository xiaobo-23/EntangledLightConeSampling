using ITensors: orthocenter, sites
## Implement time evolution block decimation (TEBD) for the self-dula kicked Ising (SDKI) model
using ITensors

let 
    N = 10
    cutoff = 1E-8
    tau = 0.1
    ttotal = 5.0
    h = 0.0                 #  an integrability-breaking longitudinal field h 

    # Make an array of 'site' indices && 
    s = siteinds("S=1/2", N; conserve_qns = false); # s = siteinds("S=1/2", N; conserve_qns = true)

    
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
        hj = π / 4 * op("Sz", s1) * op("Sz", s2) 
            # + h * op("Sz", s1) * op("Id", s2) 
            # + h * op("Sz", s2) * op("Id", s1)
        # println(typeof(hj))
        Gj = exp(-im * tau / 2 * hj)
        push!(gates, Gj)
    end

    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(gates, reverse(gates))

    # Construct the gate for the Ising model with longitudinal and transverse fields
    kickGates = ITensor[]
    for ind in 1:(N - 1)
        s1 = s[ind]
        s2 = s[ind + 1]
        tmpH = π / 4 * op("Sz", s1) * op("Sz", s2)
            + h * op("Sz", s1) * op("Id", s2) 
            + h * op("Sz", s2) * op("Id", s1)
            + π / 4 * op("Sx", s1) * op("Id", s2)
            + π / 4 * op("Sx", s2) * op("Id", s1)
        tmpG = exp(-im * tau / 2 * tmpH)
        push!(kickGates, tmpG)
    end

    # Append the reverse gates (N - 1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(kickGates, reverse(kickGates))

    # Construct the gate for the transverse Ising field applied only at integer time
    # kickGates = ITensor[]
    # for ind in 1:N
    #     s1 = s[ind]
    #     hamilt = π / 4 * op("Sx", s1)
    #     tmpG = exp(-im * hamilt)
    #     push!(kickGates, tmpG)
    # end

    # # An alternative way to construct the transverse Ising field 
    # # Need to be fixed: add identity operators? 
    # kickGates = ITensor[]
    # hamilt = ITensor()
    # for ind in 1:N
    #     s1 = s[ind]
    #     hamilt += π / 4 * op("Sx", s1)
    # end
    # tmpG = exp(-im * hamilt)
    # push!(kickGates, tmpG)
    
    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

    # Locate the central site
    centralSite = div(N, 2)

    # Compute <Sz> at rach time step and apply gates to go to the next step
    @time for time in 0.0:tau:ttotal
        Sz = expect(ψ, "Sz"; sites = centralSite)
        println("At time step $time, Sz is $Sz")

        time ≈ ttotal && break
        if (abs(time / tau % 10) < 1E-8 || abs((time + tau)/tau % 10) < 1E-8)
            println("At time $(time/tau), applying the kicked fields")
            ψ = apply(kickGates, ψ; cutoff)
        elseif abs((time + tau)/tau % 10) < 1E-8
            println("At time $(time/tau), applying the kicked fields")
            ψ = apply(kickGates, ψ; cutoff)
        else
            ψ = apply(gates, ψ; cutoff)
        end
        normalize!(ψ)
    end
    
    
    
    # Compute <Sz> at each time step and apply the gates to go to the next step
    # @time for time in 0.0:tau:ttotal
    #     Sz = expect(ψ, "Sz"; site = central_site)
    #     println("At time step $time, Sz is $Sz")

    #     time ≈ ttotal && break
    #     ψ = apply(gates, ψ; cutoff)
        
    #     # Apply the kick when time is an integer
    #     if abs(time/tau % 10) < 1E-8
    #         println("At time $(time/tau/10)")
    #         ψ = apply(kickGates, ψ; cutoff)
    #     end
    #     normalize!(ψ)
    # end

    return
end 