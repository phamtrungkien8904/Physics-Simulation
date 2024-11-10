from math import sin, cos, pi
from numpy import zeros 
import matplotlib.pyplot as plt

# Inputs
t_end = 20
t_start= 0
dt = 0.01
N = int((t_end - t_start)/dt)
t = zeros(N)
N_A = zeros(N)
N_B = zeros(N)
N_C = zeros(N)

# Constants
k1 = 0.3
k2 = 0.2

# Initial conditions
t[0] = 0
N_A[0] = 1000
N_B[0] = 500
N_C[0] = 0

# Derivative function
for i in range(1, N):
    N_A[i] = N_A[i-1] - dt*k1*N_A[i-1]
    N_B[i] = N_B[i-1] + dt*(k1*N_A[i-1] - k2*N_B[i-1])
    N_C[i] = N_C[i-1] + dt*k2*N_B[i-1] 
    t[i] = t[i-1] + dt

# Plot
plt.plot(t, N_A, label='A')
plt.plot(t, N_B, label='B')
plt.plot(t, N_C, label='C')
plt.legend(['A', 'B', 'C']) 
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Chain of Radioactive Decay')
plt.grid()
plt.show()


## Radiative balance: k1*N_A = k2*N_B