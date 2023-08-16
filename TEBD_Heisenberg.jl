## Implement time evolution block decimation (TEBD) for the one-dimensional Heisenberg model
using ITensors
using ITensors.HDF5

function generate_gates_in_brickwall_pattern!(starting_index :: Int, ending_index :: Int, input_gates, tmp_sites)
    counting_index = 0
    for tmp_index = starting_index : 2 : ending_index
        s1 = tmp_sites[tmp_index]
        s2 = tmp_sites[tmp_index + 1]
        hj =
            op("Sz", s1) * op("Sz", s2) +
            1 / 2 * op("S+", s1) * op("S-", s2) +
            1 / 2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * Δτ * hj)
        push!(input_gates, Gj)
        counting_index += 1
    end
    @show counting_index
end


function generate_gates_in_staircase_pattern!(length_of_chain :: Int, input_gates, tmp_sites)
    count_index = 0
    
    # Make gates (1, 2), (2, 3), (3, 4) ...
    for ind = 1 : length_of_chain - 1
        s1 = tmp_sites[ind]
        s2 = tmp_sites[ind+1]
        hj =
            op("Sz", s1) * op("Sz", s2) +
            1 / 2 * op("S+", s1) * op("S-", s2) +
            1 / 2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * Δτ / 2 * hj)
        push!(input_gates, Gj)
        count_index += 1
    end

    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(input_gates, reverse(input_gates))
    @show 2 * counting_index
end

let
    N = 500
    cutoff = 1E-8
    ttotal = 5.0
    global Δτ = 0.1

    # Make an array of 'site' indices
    s = siteinds("S=1/2", N; conserve_qns = false)
    
    # ## 08/16/2023
    # ## Generate the time evolution gates using the staircase pattern for one time slice/step
    # gates = ITensor[]
    # generate_gates_in_staircase_pattern!(N, gates, s)

    ## 08/16/2023
    ## Geenrate the time evolution gates using brickwall pattern for one time slice/step
    gates = ITensor[]
    generate_gates_in_brickwall_pattern!(2, N - 2, gates, s)
    generate_gates_in_brickwall_pattern!(1, N - 1, gates, s)
    
    
    # Initialize the wavefunction
    ψ₀ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    ψ = deepcopy(ψ₀)

    # Take and store the local measurements
    number_of_measurements = Int(ttotal / Δτ) + 1
    Sx = complex(zeros(number_of_measurements, N))
    Sy = complex(zeros(number_of_measurements, N))
    Sz = complex(zeros(number_of_measurements, N))
    Overlap = complex(zeros(number_of_measurements))
    
    
    # Using TEBD to evolve the wavefunction in real time && taking measurements of local observables
    index = 1
    @time for time = 0.0 : Δτ : ttotal
        # tmp_Sx = expect(ψ, "Sx", sites = 1 : N)
        # Sx[index, :] = tmp_Sx
        # tmp_Sy = epxect(ψ, "Sy", sites = 1 : N)
        # Sy[index, :] = tmp_Sy
        tmp_Sz = expect(ψ, "Sz", sites = 1 : N)
        Sz[index, :] = expect(ψ, "Sz", sites = 1 : N)
        tmp_overlap = abs(inner(ψ, ψ₀))
        Overlap[index] = abs(inner(ψ, ψ₀))
        index += 1

        println("")
        println("At time step $time, Sz is $tmp_Sz")
        @show tmp_overlap
        println("")

        time ≈ ttotal && break
        # @show time
        ψ = apply(gates, ψ; cutoff)
        normalize!(ψ)
    end

    h5open("Data_Benchmark/TEBD_Heisenberg_N$(N)_T$(ttotal)_tau$(Δτ)_Brickwall.h5", "w") do file
        # write(file, "Sx", Sx)
        # write(file, "Sy", Sy)
        write(file, "Sz", Sz)
        write(file, "Overlap", Overlap)
    end
    return
end
