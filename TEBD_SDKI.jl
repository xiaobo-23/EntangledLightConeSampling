## Implement time evolution block decimation (TEBD) for the self-dula kicked Ising (SDKI) model
using ITensors

let 
    N = 100
    cutoff = 1E-8
    tau = 0.1
    ttotal = 5.0
    h = 0.1

    # Make an array of 'site' indices && 
    s = siteinds("S=1/2", N; conserve_qns = false); # s = siteinds("S=1/2", N; conserve_qns = true)

    
    # Construct the gate for the Ising model with longitudinal longitudinal_field
    gates = ITensor[]
    for ind in 1:(N - 1)
        s1 = s[ind]
        s2 = s[ind + 1]
        hj = π / 4 * op("Sz", s1) * op("Sz", s2) 
            # + h * (op("Sz", s1) + op("Sz", s2))
        Gj = exp(-im * tau / 2 * hj)
        push!(gates, Gj)
    end

    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(gates, reverse(gates))


    # Construct the gate for the transverse Ising model applied only at integer time
    kickGates = ITensor[]
    hamilt = OpSum()
    for ind in 1:N
        s1 = s[ind]
        hamilt += π / 4, "Sx", s1
        # hamilt += π / 4 * op("Sx", s1)
    end
    tmpG = exp(-im * hamilt)
    push!(kickGate, tmpG)

    
    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

    # Locate the central site
    central_site = div(N, 2)

    # Compute <Sz> at each time step and apply the gates to go to the next step
    @time for time in 0.0:tau:ttotal
        Sz = expect(ψ, "Sz", site = central_site)
        println("At time step $time, Sz is $Sz")

        time ≈ ttotal && break
        ψ = apply(gates, ψ; cutoff)
        
        # Apply the kick when time is an integer
        if time/tau % 10 == 0
            ψ = apply(kickGates, ψ; cutoff)
        end
        normalize!(ψ)
    end

    return
end 