## Implement time evolution block decimation (TEBD) for the self-dual kicked Ising (SDKI) model
using ITensors
using ITensors.HDF5
using ITensors: orthocenter, sites, copy
using Base: Float64
ITensors.disable_warn_order()
let 
    N = 10
    cutoff = 1E-8
    tau = 0.1
    ttotal = 10.0
    h = 10.0                                             # an integrability-breaking longitudinal field h 

    # Make an array of 'site' indices && quantum numbers are not conserved due to the transverse fields
    s = siteinds("S=1/2", N; conserve_qns = false);      # s = siteinds("S=1/2", N; conserve_qns = true)

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

    ###############################################################################################################################
    ## Construct the gates for TEBD algorithm
    ###############################################################################################################################
    
    # Construct the gate for the Ising model with longitudinal field
    gates = ITensor[]
    for ind in 1:(N - 1)
        s1 = s[ind]
        s2 = s[ind + 1]

        if (ind - 1 < 1E-8)
            tmp1 = 2 
            tmp2 = 1
        elseif (abs(ind - (N - 1)) < 1E-8)
            tmp1 = 1
            tmp2 = 2
        else
            tmp1 = 1
            tmp2 = 1
        end

        println("")
        println("Coefficients are $(tmp1) and $(tmp2)")
        println("Site index is $(ind) and the conditional sentence is $(ind - (N - 1))")
        println("")

        hj = π * op("Sz", s1) * op("Sz", s2) + tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2)
        Gj = exp(-1.0im * tau / 2 * hj)
        push!(gates, Gj)
    end
    # Append the reverse gates (N -1, N), (N - 2, N - 1), (N - 3, N - 2) ...
    append!(gates, reverse(gates))

    # Construct the transverse field as the kicked gate
    ampo = OpSum()
    for ind in 1 : N
        ampo += π/2, "Sx", ind
    end
    Hₓ = MPO(ampo, s)
    Hamiltonianₓ = Hₓ[1]
    for ind in 2 : N
        Hamiltonianₓ *= Hₓ[ind]
    end
    expHamiltoinianₓ = exp(-1.0im * Hamiltonianₓ)
    

    ##################################################################################################################################################
    ## Construct the holographic quantum dynamics simulation (holoQUADS) circuit
    ##################################################################################################################################################
    # Loose end: figure out the Coefficients and long-distance two-site gates



    # Construct the classical version of the holographic quantum dynamics circuit
    function constructLayer(numQubits :: Int, qubitToMeasure :: Int)
        gates = ITensor()

        if qubitToMeasure == 1
            for ind₁ in reverse(1 : 2 * (numQubits - 1))                                                # Number of MPS = 2 * number of Qubits
                for ind₂ in 1 : ((ind₁ + 1) // 2)
                    s1 = s[2 * ind₂ - (ind₁ % 2)]
                    s2 = s[2 * ind₂ + 1 - (ind₁ % 2)]

                    if (ind - 1 < 1E-8)
                        tmp1 = 2 
                        tmp2 = 1
                    elseif (abs(ind - (N - 1)) < 1E-8)
                        tmp1 = 1
                        tmp2 = 2
                    else
                        tmp1 = 1
                        tmp2 = 1
                    end

                    hj += (π * op("Sz", s1) * op("Sz", s2) + tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2))
                end
                Gj = exp(-1.0im * tau / 2 * hj) 
                push!(gates, Gj)
            end
        else                                                                                        
            for ind₁ in 1 : 2 * (numQubits - 1):
                tmpInd₁ = (2 * (qubitToMeasure - 1) - ind₁ + N) % N
                tmpInd₂ = (2 * (qubitToMeasure - 1) - ind₁ - 1 + N) % N 

                if tmpInd₁ == 0
                    tmpInd₁ = N
                end

                if tmpInd₂ == 0
                    tmpInd₂ = N
                end

                s1 = s[tmpInd₁]
                s2 = s[tmpInd₂]

                hj = (π * op("Sz", s1) * op("Sz", s2) + tmp1 * h * op("Sz", s1) * op("Id", s2) + tmp2 * h * op("Id", s1) * op("Sz", s2))
                Gj = exp(-1.0im * tau / 2 * hj) 
                push!(gates, Gj)
            end
        end
        return gates
    end 


    # Initialize the wavefunction
    ψ = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
    # @show eltype(ψ), eltype(ψ[1])
    # states = [isodd(n) ? "Up" : "Dn" for n = 1:N]
    # ψ = randomMPS(s, states, linkdims = 2)
    # @show maxlinkdim(ψ)

    # Locate the central site
    # centralSite = div(N, 2)

    # Compute the overlap between the original and time evolved wavefunctions
    ψ_copy = copy(ψ)
    ψ_overlap = Complex{Float64}[]

    # Compute local observables e.g. Sz, Czz 
    timeSlices = Int(ttotal / tau) + 1; println("Total number of time slices that need to be saved is : $(timeSlices)")
    Sx = zeros(timeSlices, N); Sx = complex(Sx)
    Sy = zeros(timeSlices, N); Sy = complex(Sy)
    Sz = zeros(timeSlices, N); Sz = complex(Sz)
    Cxx = zeros(timeSlices, N); Cxx = complex(Cxx)
    Czz = zeros(timeSlices, N); Czz = complex(Czz)
    index = 1
    
    @time for time in 0.0:tau:ttotal
        tmp_overlap = abs(inner(ψ, ψ_copy))
        println("The inner product is: $tmp_overlap")
        append!(ψ_overlap, tmp_overlap)

        # Local observables e.g. Sx, Sz
        tmpSx = expect(ψ_copy, "Sx"; sites = 1 : N); Sx[index, :] = tmpSx
        tmpSz = expect(ψ_copy, "Sz"; sites = 1 : N); Sz[index, :] = tmpSz

        # Spin correlaiton functions e.g. Cxx, Czz
        tmpCxx = correlation_matrix(ψ_copy, "Sx", "Sx"; sites = 1 : N);  Cxx[index, :] = tmpCxx[4, :]
        tmpCzz = correlation_matrix(ψ_copy, "Sz", "Sz"; sites = 1 : N);  Czz[index, :] = tmpCzz[4, :]
        # Czz[index, :] = vec(tmpCzz')
        index += 1
        
        time ≈ ttotal && break
        if (abs((time / tau) % 10) < 1E-8)
            println("")
            println("Apply the kicked gates at integer time $time")
            println("")
            ψ_copy = apply(expHamiltoinianₓ, ψ_copy; cutoff)
            normalize!(ψ_copy)
        end


        ψ_copy = apply(gates, ψ_copy; cutoff)
        normalize!(ψ_copy)

        #********************************************************************************
        # Previous ideas of using a square wave to represent a delta function
        #********************************************************************************
        # if (abs((time / tau) % 10) < 1E-8)
        #     # ψ = apply(kickGates, ψ; cutoff = cutoff)
        #     ψ = apply(expHamiltoininaₓ, ψ; cutoff)
        # # else
        # #     ψ = apply(gates, ψ; cutoff)
        # end

        # tmp_overlap = abs(inner(ψ, ψ_copy))
        # append!(ψ_overlap, tmp_overlap)
        # println("The inner product is: $tmp_overlap")
    end

    # Store data into a hdf5 file
    file = h5open("RawData/TEBD_N$(N)_h$(h)_Info.h5", "w")
    write(file, "Sx", Sx)
    write(file, "Sz", Sz)
    write(file, "Cxx", Cxx)
    write(file, "Czz", Czz)
    write(file, "Wavefunction Overlap", ψ_overlap)
    close(file)
    
    return
end 