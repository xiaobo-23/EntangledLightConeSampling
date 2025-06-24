# 05/11/2023
# Define projection operators for Sx, Sy, and Sz
# Define projectors in the Sz basis

using ITensors
using ITensorMPS

# Relevant operators for Sx
Sx_projn = (1 / sqrt(2)) * [1  1; 1  -1]
Sx_projn_plus  = 0.5 * [1  1;  1  1]
Sx_projn_minus = 0.5 * [1 -1; -1  1]



# Relevant operators for Sy
Sy_projn = (1 / sqrt(2)) * [1  1.0im; 1  -1.0im]
Sy_projn_plus  = 0.5 * [1  im; -im  1]
Sy_projn_minus = 0.5 * [1  -im; im  1]



# Relevant operators for Sz
Sz_projn = Matrix{Float64}(I, 2, 2)
Sz_projn_up = [1  0;  0  0]
Sz_projn_dn = [0  0;  0  1]