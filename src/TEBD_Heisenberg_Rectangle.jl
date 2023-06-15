## Implement time evolution block decimation (TEBD) for the one-dimensional Heisenberg model
using ITensors
using ITensors.HDF5

let 
    N = 42
    cutoff = 1E-8
    tau = 0.05
    ttotal = 5.0

    # Make an array of 'site' indices
    s = siteinds("S=1/2", N; conserve_qns = false)

    
    # Construct layers of two-site gates used in TEBD
    gates = ITensor[]

    # Construct the layer with div(N, 2) - 1 two-site gates
    for ind in 2 : 2 : (N - 2)
        s1 = s[ind]
        s2 = s[ind + 1]
        hj = op("Sz", s1) * op("Sz", s2) + 1/2 * op("S+", s1) * op("S-", s2) + 1/2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-1.0im * tau * hj)
        push!(gates, Gj)
    end

    # Construct the layer with div(N, 2) two-site gates
    for ind in 1 : 2 : (N - 1)
        s1 = s[ind]
        s2 = s[ind + 1]
        hj = op("Sz", s1) * op("Sz", s2) + 1/2 * op("S+", s1) * op("S-", s2) + 1/2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-1.0im * tau * hj)
        push!(gates, Gj)
    end

    # Initialize the wavefunction
    ψ₀ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    ψ = deepcopy(ψ₀)

    # Take a measurement of the initial random MPS to make sure the same random MPS is used through all codes.
    Sx₀ = expect(ψ, "Sx", sites = 1 : N)
    Sy₀ = expect(ψ, "Sy", sites = 1 : N)
    Sz₀ = expect(ψ, "Sz", sites = 1 : N)

    # Take and store the local measurements
    number_of_measurements = Int(ttotal / tau) + 1
    Sx = complex(zeros(number_of_measurements, N))
    Sy = complex(zeros(number_of_measurements, N))
    Sz = complex(zeros(number_of_measurements, N))
    Overlap = complex(zeros(number_of_measurements))
     
    # Using TEBD to evolve the wavefunction in real time && taking measurements of local observables
    index = 1
    @time for time in 0.0:tau:ttotal
        # tmp_Sx = expect(ψ, "Sx", sites = 1 : N)
        # Sx[index, :] = tmp_Sx
        # tmp_Sy = epxect(ψ, "Sy", sites = 1 : N)
        # Sy[index, :] = tmp_Sy
        tmp_Sz = expect(ψ, "Sz", sites = 1 : N)
        Sz[index, :] = tmp_Sz

        tmp_overlap = abs(inner(ψ, ψ₀))
        Overlap[index] = tmp_overlap
        index += 1

        println("")
        println("At time step $time, Sz is $tmp_Sz")
        @show tmp_overlap
        println("")

        time ≈ ttotal && break
        @show time
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
    end

    file = h5open("TEBD_Heisenberg_N$(N)_T$(ttotal)_tau$(tau)_AFM_Rectangle.h5", "w")
    # write(file, "Sx", Sx)
    # write(file, "Sy", Sy)
    write(file, "Sz", Sz)
    write(file, "Overlap", Overlap)
    write(file, "Initial Sx", Sx₀)
    write(file, "Initial Sy", Sy₀)
    write(file, "Initial Sz", Sz₀)
    close(file)

    return
end 