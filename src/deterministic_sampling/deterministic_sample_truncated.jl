# 10/21/2024
# Implement deterministic sampling for the ground-state of model Hamiltonians

using ITensors
using ITensorMPS
using Random
using Statistics
using HDF5
using TimerOutput

# include("Sample.jl")
include("Projection.jl")


# Sample a two-site MPS to compute Sx, Sy or Sz
function sample(m :: MPS, j :: Int, observable :: AbstractString)
    mpsLength = length(m)

    # Move the orthogonality center of the MPS to site j
    orthogonalize!(m, j)
    if orthocenter(m) != j
        error("sample: MPS m must have orthocenter(m) == j")
    end
    
    # Check the normalization of the MPS
    if abs(1.0 - norm(m[j])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[j]))")
    end

    if observable == "Sx"
        tmp_projn = Sx_projn
        projn_up = Sx_projn_plus
        projn_dn = Sx_projn_minus
    elseif observable == "Sy"
        tmp_projn = Sy_projn
        projn_up = Sy_projn_plus
        projn_dn = Sy_projn_minus
    elseif observable == "Sz"
        tmp_projn = Sz_projn
        projn_up = Sz_projn_up
        projn_dn = Sz_projn_dn
    else
        error("sample: Measurement type doesn't exist")
    end

    # Sample the target observables
    result = zeros(Int, 2)
    A = m[j]
    
    for ind in j:j+1
        tmpS = siteind(m, ind)
        d = dim(tmpS)
        pdisc = 0.0
        r = rand()

        n = 1 
        An = ITensor()
        pn = 0.0

        while n <= d
            projn = ITensor(tmpS)
            # projn[tmpS => n] = 1.0
            projn[tmpS => 1] = tmp_projn[n][1]
            projn[tmpS => 2] = tmp_projn[n][2]
        
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn

            (r < pdisc) && break
            n += 1
        end
        result[ind - j + 1] = n

        if ind < mpsLength
            A = m[ind + 1] * An
            A *= (1. / sqrt(pn))
        end

        # Collapse the site based on the measurements 
        if n == 1
            tmp_reset = ITensor(projn_up, tmpS', tmpS)
        else
            tmp_reset = ITensor(projn_dn, tmpS', tmpS)
        end

        m[ind] *= tmp_reset
        noprime!(m[ind])
    end
    return result
end


# 08/16/2024
# Sample a single-site MPS deterministically  
function single_site_dsample(m :: MPS, j :: Int, observable :: AbstractString) 
    # Set up the projection operator
    if observable == "Sx"
        tmp_projn = Sx_projn
    elseif observable == "Sy"
        tmp_projn = Sy_projn
    elseif observable == "Sz"
        tmp_projn = Sz_projn
    else
        error("sample: the type of measurement doesn't exist")
    end

    # Sample the target observables
    result_vector = []
    A = m[j]
    tmpS = siteind(m, j)
    d = dim(tmpS)

    for index in 1 : d
        pn = 0.0
        An = ITensor()
        projn = ITensor(tmpS)
        # projn[tmpS => index] = 1.0
        projn[tmpS => 1] = tmp_projn[index][1]
        projn[tmpS => 2] = tmp_projn[index][2]

        An = A * dag(projn);
        pn = real(scalar(dag(An) * An))

        # println("")
        # println("")
        # @show typeof(A), typeof(An), typeof(projn) 
        # @show A, projn
        # println("")
        # println("")

        sample_info = [string(index), pn, An]
        push!(result_vector, sample_info)
    end
    # @show length(result_vector)
    # @show result_vector
    return result_vector
end


let 
    # Initialize the random MPS
    N = 100                  # Number of physical sites
    h = 1.2                  # Transverse field
    # sites = siteinds("S=1/2", N; conserve_qns = false) 
    # state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]
    
    println(" ")
    println("*************************************************************************************")
    println("Read in the wavefunction from the file and start the sampling process.")
    println("*************************************************************************************")
    println(" ")
    file = h5open("data/Transverse_Ising_N100_h1.2_Wavefunction.h5", "r")
    ψ = read(file, "Psi", MPS)
    Sz = expect(ψ, "Sz"; sites = 1 : N)
    Sx = expect(ψ, "Sx"; sites = 1 : N)
    Czz = correlation_matrix(ψ, "Sz", "Sz"; sites = 1 : N)
    @show Sx
    @show Czz

    #***********************************************************************************
    # Generate samples in the Sz basis based on the Born rule
    #***********************************************************************************
    Nₛ = 5000              # Number of samples
    bitstring_Sz  = Array{Float64}(undef, Nₛ, N)
    bitstring_Czz = Array{Float64}(undef, Nₛ, N * N)
    for i = 1 : Nₛ
        ψ_copy = deepcopy(ψ)
        println("Generate bitstring #$(i)")
        println("")
        println("")
        for j = 1 : 2 : N
            tmp = sample(ψ_copy, j, "Sz")
            normalize!(ψ_copy)
            bitstring_Sz[i, j: j + 1] = tmp
        end
    end
    bitstring_Sz[bitstring_Sz .== 1] .= 0.5
    bitstring_Sz[bitstring_Sz .== 2] .= -0.5
    # @show bitstring
    
    # Compute the expectation value and standard deviation of one-point functions based on the samples
    sample_ave_Sz = mean(bitstring_Sz, dims = 1)
    sample_std_Sz = std(bitstring_Sz, corrected = true, dims = 1) / sqrt(Nₛ)
    # @show sample_expectation 
    # @show sample_std


    # Compute the expectation value and standard deviation of two-point functions based on the samples
    for index in 1 : Nₛ
        for i = 1 : N 
            for j = 1 : N
                bitstring_Czz[index, (i - 1) * N + j] = bitstring_Sz[index, i] * bitstring_Sz[index, j]
            end
        end
    end
    sample_ave_Czz = mean(bitstring_Czz, dims = 1)
    sample_std_Czz = std(bitstring_Czz, corrected = true, dims = 1) / sqrt(Nₛ)
    

    #***********************************************************************************
    # Generate samplesin the Sx basis based on the Born rule.
    #***********************************************************************************

    bitstring_Sx = Array{Float64}(undef, Nₛ, N)
    for i = 1 : Nₛ
        ψ_copy = deepcopy(ψ)
        println("Generate bitstring #$(i)")
        println("")
        println("")
        for j = 1 : 2 : N
            tmp = sample(ψ_copy, j, "Sx")
            normalize!(ψ_copy)
            bitstring_Sx[i, j: j + 1] = tmp
        end
    end
    bitstring_Sx[bitstring_Sx .== 1] .= 0.5
    bitstring_Sx[bitstring_Sx .== 2] .= -0.5
    # @show bitstring_Sx
    
    # Compute the expectation value and standard deviation of one-point functions based on the samples 
    sample_ave_Sx = mean(bitstring_Sx, dims = 1)
    sample_std_Sx = std(bitstring_Sx, corrected = true, dims = 1) / sqrt(Nₛ)


    #************************************************************************************ 
    # Sample the ground-state wavefunction using deterministic sampling
    #************************************************************************************
    Sz_deterministic = Array{Float64}(undef, N)
    Prob = Array{Float64}(undef, N)
    N_deterministic = 1024                              # Number of deterministic samples


    # Sample the first site
    # dsample = []
    ψ_copy = deepcopy(ψ)
    dsample = single_site_dsample(ψ_copy, 1, "Sx")
    Sz_deterministic[1] = 0.5 * (dsample[1][2] - dsample[2][2])
    Prob[1] = dsample[1][2] + dsample[2][2]
    deterministic_bitstring = [dsample[1][1], dsample[2][1]]
    @show Sz_deterministic[1], Sz[1], Sx[1]
    @show dsample[1][2] + dsample[2][2]
    @show deterministic_bitstring

    state_probability = Vector{Float64}()
    for index in 2 : N
        iteration = length(dsample)
        for _ in 1 : iteration
            # @show typeof(dsample)
            # @show index, dsample[index][1], dsample[index][2]
            tmp = popfirst!(dsample)
            # tmp_string = popfirst!(deterministic_bitstring)

            ψ_tmp = deepcopy(ψ_copy)
            # @show typeof(ψ_tmp[index]), ψ_tmp[index]
            ψ_tmp[index] = tmp[3] * ψ_tmp[index]    
            # # normalize!(ψ_tmp)
            # @show typeof(ψ_tmp[index]), ψ_tmp[index]
            ψ_update = MPS(ψ_tmp[index : N])
            # normalize!(ψ_update)
            
            tmp_sample = single_site_dsample(ψ_update, 1, "Sx")
            # @show length(ψ_update), length(tmp_sample)
            
            push!(dsample, tmp_sample[1])
            push!(dsample, tmp_sample[2])   

            # # @show tmp_string, tmp_string * tmp_sample[1][1], tmp_string * tmp_sample[2][1]
            # push!(deterministic_bitstring, tmp_string * tmp_sample[1][1])
            # push!(deterministic_bitstring, tmp_string * tmp_sample[2][1])
            

            # if length(dsample) != length(deterministic_bitstring)
            #     error("deterministic sample: the length of projection vectors and bitstrings are not equal.")
            # end

            # @show index, tmp_sample[1][2], tmp_sample[2][2]
            Sz_deterministic[index] += 0.5 * (tmp_sample[1][2] - tmp_sample[2][2])
            Prob[index] += tmp_sample[1][2] + tmp_sample[2][2]
            if index == N 
                push!(state_probability, tmp_sample[1][2])
                push!(state_probability, tmp_sample[2][2])
            end
        end
        Sz_deterministic[index] = Sz_deterministic[index] / Prob[index]

        # println(" ")
        # println(" ")
        # @show length(dsample)
        # println(" ")
        # println(" ")

        # Truncated the number of deterministic samples to keep if the number of samples is larger than some threshold
        if length(dsample) > N_deterministic
            sorted_dsample = sort(dsample, by = x -> x[2], rev = true)
            dsample = sorted_dsample[1 : N_deterministic]
            @show length(dsample)
            @show dsample[1][2], dsample[2][2], dsample[3][2], dsample[4][2], dsample[5][2], dsample[6][2]
        end
    end
    
    # # @show Sz 
    # # @show Sz_deterministic
    # @show Prob
    # # @show state_probability
    # # @show bitstring
    # state_distribution = zip(deterministic_bitstring, state_probability)

    # # Save results to a file
    # h5open("data/Transverse_Ising_N$(N)_h$(h)_Sample$(N_deterministic).h5", "w") do file
    #     write(file, "Sx", Sx)
    #     write(file, "Sz", Sz)
    #     write(file, "Czz", Czz)
    #     write(file, "Sz ave.", sample_ave_Sz)
    #     write(file, "Sz err.", sample_std_Sz)
    #     write(file, "Sx ave.", sample_ave_Sx)
    #     write(file, "Sx err.", sample_std_Sx)
    #     write(file, "Czz ave.", sample_ave_Czz)
    #     write(file, "Czz err.", sample_std_Czz)
    #     write(file, "Deterministic Sample Sx", Sz_deterministic)
    #     write(file, "Deterministic Sample Probability", Prob)
    #     write(file, "State Probability", state_probability)
    #     # write(file, "State Distribution", state_distribution)
    # end

end