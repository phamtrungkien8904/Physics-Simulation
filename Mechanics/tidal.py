import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants and Parameters
G = 6.67430e-11  # Gravitational constant (m³ kg⁻¹ s⁻²)
M_moon = 7.34767309e22  # Mass of the Moon (kg)
R_earth = 6.371e6  # Earth's radius (m)
d_moon = 384400e3  # Earth-Moon distance (m)
g = 9.81  # Gravitational acceleration on Earth (m/s²)
omega = 2 * np.pi / (27.3 * 86400)  # Moon's angular velocity (rad/s)

# Tidal coefficient calculation
tidal_coefficient = (G * M_moon * R_earth**2) / (2 * g * d_moon**3)
print(f"Tidal Coefficient: {tidal_coefficient:.5f} meters")

# Discretize Earth's circumference
num_points = 360
phis = np.linspace(0, 2 * np.pi, num_points)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axis('off')

# Earth as a unit circle
earth_circle = plt.Circle((0, 0), 1, color='blue', alpha=0.3)
ax.add_patch(earth_circle)

# Moon's orbit (dashed circle)
moon_orbit = plt.Circle((0, 0), 2, color='gray', fill=False, linestyle='--')
ax.add_patch(moon_orbit)

# Water level line and Moon dot
water_line, = ax.plot([], [], 'b', lw=2)
moon_dot, = ax.plot([], [], 'ko', markersize=10)

def init():
    water_line.set_data([], [])
    moon_dot.set_data([], [])
    return water_line, moon_dot

def update(frame):
    t = frame * 3600  # Convert frame to seconds (1 hour per frame)
    theta = omega * t  # Moon's angle
    
    # Update Moon's position
    moon_x = 2 * np.cos(theta)
    moon_y = 2 * np.sin(theta)
    moon_dot.set_data([moon_x], [moon_y])
    
    # Angle difference between Moon and each Earth point
    alpha = theta - phis
    
    # Calculate normalized tidal deformation (3cos²α - 1)
    h_normalized = 3 * np.cos(alpha)**2 - 1
    
    # Scale deformation for visibility
    deformation_scale = 0.1
    r = 1 + deformation_scale * h_normalized
    
    # Update water line coordinates
    x = r * np.cos(phis)
    y = r * np.sin(phis)
    water_line.set_data(x, y)
    
    # Debugging: Print shapes of x and y
    if frame == 0:
        print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")
    
    return water_line, moon_dot

# Create animation
ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, interval=100)

# Debugging: Print a message when the animation starts
print("Starting animation...")

# Show the animation
plt.show()