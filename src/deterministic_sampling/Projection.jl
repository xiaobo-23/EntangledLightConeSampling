## 05/11/2023
## Set up the projection matrices of Sx, Sy, and Sz and measure in the Sz basis
using ITensors

# Define projectors in the Sz basis

# Measure Sx
Sx_projn = 1/sqrt(2) * [[1, 1], [1, -1]]

Sx_projn_plus = 1/2 * [
    1  1
    1  1
]

Sx_projn_minus = 1/2 * [
     1   -1
    -1    1
]


# Measure Sy
Sy_projn = 1/sqrt(2) * [[1, 1.0im], [1, -1.0im]]

Sy_projn_plus = 1/2 * [
    1   -1.0im
    -1.0im  -1
]


Sy_projn_minus = 1/2 * [
    1   1.0im
    1.0im  -1
]

## DEBUG NEED TO FIGURE OUT |+> AND |-> states
# Sy_projn_plus = 1/2 * [
#     1   1.0im
#     1.0im  -1
# ]

# Sy_projn_minus = 1/2 * [
#     1   -1.0im
#     -1.0im  -1
# ]

# Measure Sz
Sz_projn = [[1, 0], [0, 1]]

Sz_projn_up = [
    1  0
    0  0
]

Sz_projn_dn = [
    0  0
    0  1
]