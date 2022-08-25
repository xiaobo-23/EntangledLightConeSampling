## Implement time evolution block decimation (TEBD) for the one-dimensional Heisenberg model
## Preparation for the implementation of TEBD for the self-dula kicked Ising (SDKI) model
using ITensors

let 
    N = 100
    cutoff = 1E-8
    tau = 0.1
    ttotal = 5.0

    # Make an array of 'site' indices
    s = siteinds("S=1/2", N; conserve_qns = true)

    # Make gates (1, 2), (2, 3), (3, 4) ...
    gates = ITensor[]
    for ind in 1:(N - 1)
        s1 = s[ind]
        s2 = s[ind + 1]
        hj = op("Sz", s1) * op("Sz", s2) + 
            1/2 * op("S+", s1) * op("S-", s2) + 
            1/2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * tau / 2* hj)
        push!(gates, Gj)
    end

    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(gates, reverse(gates))

    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")

    central_site = div(N, 2)

    # Compute <Sz> at each time step and apply the gates to go to the next step
    for time in 0.0:tau:ttotal
        Sz = expect(ψ, "Sz", site = central_site)
        println("At time step $time, Sz is $Sz")

        time ≈ ttotal && break
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
    end

    return
end 