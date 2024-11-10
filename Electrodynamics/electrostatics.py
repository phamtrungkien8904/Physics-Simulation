import numpy as np
import matplotlib.pyplot as plt

# INPUT DATA
L = 2.5  # plate length
R = 5e-2  # sphere radius
a = 0.5  # 2a - distance between plates
Vmax = 100
Vmin = -100

Nx = 1000
Ny = 500

eps = (Vmax - Vmin) / 1e5

contour_range_V = np.linspace(Vmin, Vmax, 41)
xmin, xmax = -2, 2
ymin, ymax = -0.5, 1.5

# CALCULATION
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)

mpx = Nx // 2  # Mid-point of x
mpy = Ny // 4  # Mid-point of y

hx = (xmax - xmin) / (Nx - 1)
hy = (ymax - ymin) / (Ny - 1)

A = 2 / hx**2 + 2 / hy**2
B = 1 / hx**2
C = 1 / hy**2

V = np.zeros((Nx, Ny))  # Potential (Voltage) matrix
V_const = np.zeros((Nx, Ny))


# Initializing edge potentials
V[0, :] = 0
V[-1, :] = 0
V[:, 0] = 0
V[:, -1] = 0

# Initializing corner potentials
V[0, 0] = 0.5 * (V[0, 1] + V[1, 0])
V[-1, 0] = 0.5 * (V[-2, 0] + V[-1, 1])
V[0, -1] = 0.5 * (V[0, -2] + V[1, -1])
V[-1, -1] = 0.5 * (V[-1, -2] + V[-2, -1])

# Length of plate in terms of number of grids
length_plate = int(Nx * L / (xmax - xmin))
lp = length_plate // 2

# Position of plate on y axis
position_sphere = int(Ny * a / (ymax - ymin))
pp1 = mpy
pp2 = mpy + position_sphere

# Initializing Sphere Potentials
phi = np.linspace(0, 2 * np.pi, 120)
for i in range(len(phi)):
    x_sphere = R * np.cos(phi[i])
    px_sphere = int(Nx * (x_sphere - xmin) / (xmax - xmin))
    y_sphere = a + R * np.sin(phi[i])
    py_sphere = int(Ny * (y_sphere - ymin) / (ymax - ymin))

    if 0 <= px_sphere < Nx and 0 <= py_sphere < Ny:
        V[px_sphere, py_sphere] = Vmax
        V_const[px_sphere, py_sphere] = 1

# Initializing Plate Potentials
V[mpx - lp:mpx + lp, pp1] = Vmin
V[mpx, pp2] = Vmax
V_const[mpx - lp:mpx + lp, pp1] = 1
V_const[mpx, pp2] = 1

# Iterative solution using finite difference method
p = 1e100
V_old = V.copy()

while p > eps:
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if V_const[i, j] == 0:
                V[i, j] = (B * (V[i + 1, j] + V[i - 1, j]) + C * (V[i, j + 1] + V[i, j - 1])) / A

    Delta_V = np.abs(V - V_old)
    p = np.max(Delta_V)
    V_old = V.copy()
    error = p / (Vmax - Vmin)
    print(f'error={error}')

# Calculating electric field
V = V.T  # Transpose for proper x-y orientation
Ey, Ex = np.gradient(-V)
E = np.sqrt(Ex**2 + Ey**2)
Emax = np.max(E)

# Masking the vectors at the surface
mask = (V == Vmax) | (V == Vmin)
Ex[mask] = 0
Ey[mask] = 0

# Commented out the plotting section
step = 2  # Change this value to adjust density (higher = fewer vectors)
plt.figure(figsize=(10, 6))
plt.title('Electric Field and Potential Distribution')
c = plt.pcolor(x, y, V, shading='auto', cmap='jet', vmin=Vmin, vmax=Vmax)  # Set vmin and vmax
plt.contour(x, y, V, levels=contour_range_V, colors='k', linewidths=1, linestyles='solid')
plt.quiver(x[::step], y[::step], Ex[::step, ::step], Ey[::step, ::step], scale=800, headwidth=1, headlength=1)  # Adjust scale and head size
cbar = plt.colorbar(c, label='Potential [V]')  # Set colorbar label
cbar.set_ticks(np.linspace(Vmin, Vmax, num=9))  # Set colorbar ticks to range of V
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('tight')  # Fit the image to the axis border

# FIGURES
plt.figure(figsize=(10, 6))
plt.title('Electric Field Magnitude')
c = plt.pcolor(x, y, E, shading='auto', cmap='jet', vmin=0, vmax=Emax)  # Set vmin and vmax
cbar = plt.colorbar(c, label='Electric Field [V/m]')  # Set colorbar label
cbar.set_ticks(np.linspace(0, Emax, num=9))  # Set colorbar ticks to range of V
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('tight')  # Fit the image to the axis border

# Show both figures
plt.show()