using ITensors
using LinearAlgebra

sigmax, sigmay, sigmaz, Id = [0 1; 1 0], [0 -im; im 0], [1 0; 0 -1],  Matrix(I, 2,2)

#Think they use these parameters in the two papers
K1, K2, K3 = 0.3, 0.5, 1.25


Kp, Km = K1 + K2, K1 - K2
V = -im*(K1*kron(sigmax, sigmax) + K2*kron(sigmay, sigmay) + K3*kron(sigmaz, sigmaz) )
V = exp(V)

W_reshape = reshape(V, (2,2,2,2))

j, sigmap, i, sigma = Index(2, "j"), prime(Index(2, "sigma")), Index(2, "i"), Index(2, "sigma")
W_tensor = ITensor(W_reshape, j, sigmap, i, sigma)