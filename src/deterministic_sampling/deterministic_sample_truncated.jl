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
            projn[tmpS => n] = 1.0
            # projn[tmpS => 1] = tmp_projn[n][1]
            # projn[tmpS => 2] = tmp_projn[n][2]
        
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
        projn[tmpS => index] = 1.0
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
    N = 20                  # Number of physical sites
    sites = siteinds("S=1/2", N; conserve_qns = false) 
    state = [isodd(n) ? "Up" : "Dn" for n = 1 : N]

    # Set up the Heisenberg model Hamiltonian on a one-dimensional chain
    os = OpSum()
    for j = 1 : N  - 1
        os += "Sz",j,"Sz",j+1
        os += 1/2,"S+",j,"S-",j+1
        os += 1/2,"S-",j,"S+",j+1
    end
    H = MPO(os, sites)
    ψ₀ = randomMPS(sites, state, linkdims = 2)

    # Set up the parameters for the DMRG simulation
    cutoff = [1E-10]
    nsweeps = 10
    maxdim = [10,20,100,100,200]
    
    energy, ψ = dmrg(H, ψ₀; nsweeps,maxdim,cutoff)
    Sz = expect(ψ, "Sz"; sites = 1 : N)
    # @show Sz

    # Generate the samples using Born rule
    Nₛ = 1000           # Number of samples
    bitstring = Array{Float64}(undef, Nₛ, N)
    for i = 1 : Nₛ
        ψ_copy = deepcopy(ψ)
        println("Generate bitstring #$(i)")
        println("")
        println("")
        for j = 1 : 2 : N
            tmp = sample(ψ_copy, j, "Sz")
            normalize!(ψ_copy)
            bitstring[i, j: j + 1] = tmp
        end
    end
    bitstring[bitstring .== 1] .= 0.5
    bitstring[bitstring .== 2] .= -0.5

    
    # Compute the expectation value and standard deviation of the samples
    # @show bitstring
    sample_expectation = mean(bitstring, dims = 1)
    sample_std = std(bitstring, corrected = true, dims = 1) / sqrt(Nₛ)
    # @show Sz 
    # @show sample_expectation 
    # @show sample_std 


    # Sample the ground-state of the Heisenberg model using deterministic sampling
    Sz_deterministic = Array{Float64}(undef, N)
    Prob = Array{Float64}(undef, N)
    N_deterministic = 20000     # Number of deterministic samples

    # Sample the first site
    # dsample = []
    ψ_copy = deepcopy(ψ)
    dsample = single_site_dsample(ψ_copy, 1, "Sz")
    Sz_deterministic[1] = 0.5 * (dsample[1][2] - dsample[2][2])
    Prob[1] = dsample[1][2] + dsample[2][2]
    deterministic_bitstring = [dsample[1][1], dsample[2][1]]
    @show Sz_deterministic[1], Sz[1]
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
            
            tmp_sample = single_site_dsample(ψ_update, 1, "Sz")
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
    
    # @show Sz 
    # @show Sz_deterministic
    @show Prob
    # @show state_probability
    # @show bitstring
    state_distribution = zip(deterministic_bitstring, state_probability)

    # Save results to a file
    h5open("data/Heisenberg_N$(N)_State$(N_deterministic).h5", "w") do file
        write(file, "Sz", Sz)
        write(file, "Sample ave.", sample_expectation)
        write(file, "Sample err.", sample_std)
        write(file, "Deterministic Sample Sz", Sz_deterministic)
        write(file, "Deterministic Sample Probability", Prob)
        write(file, "State Probability", state_probability)
        # write(file, "State Distribution", state_distribution)
    end
end