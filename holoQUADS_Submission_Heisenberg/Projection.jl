## 08/15/2023
## Define projection matrices used in the holoQUADS circuit
using ITensors

# Define projectors ni the Sz basis

# Measure Sz
## Need to be modified based on the initialization of MPS
Sz_matrix = [
    1 0
    0 1
]

S⁺_matrix = [
    0 1
    0 0
]

S⁻_matrix = [
    0 0
    1 0
]