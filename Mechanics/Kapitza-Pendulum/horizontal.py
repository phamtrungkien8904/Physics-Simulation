from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

FRAMES_DIR = Path('frames')

# Parameters
m, L = 1, 1
a = 0.2
w = 100
g = 9.81

def deriv(t, y, L, a, w):
    """
    Return the derivatives for the inverted pendulum with a horizontally oscillating pivot.
    For a pivot oscillating as x_p = a*cos(w*t), the equation of motion (projected in the
    tangential direction) becomes:
      theta'' = - (g/L) * sin(theta) - (a*w**2/L)*cos(w*t)*cos(theta)
    """
    theta, thetadot = y
    wt = w * t
    cwt = np.cos(wt)
    cth, sth = np.cos(theta), np.sin(theta)
    thetadotdot = - g / L * sth - a * w**2 / L * cwt * cth
    return [thetadot, thetadotdot]

if __name__ == "__main__":
    # Time parameters.
    tmax, dt = 3, 1/w/20
    t = np.arange(0, tmax, dt)
    y0 = [0.1, 0]  # initial conditions: small angle near the inverted position.
    
    tspan = (0, tmax)
    sol = solve_ivp(deriv, tspan, y0, t_eval=t, args=(L, a, w))
    theta = sol.y[0]
    
    # Compute the bob position.
    # The pivot now oscillates horizontally: (a*cos(w*t), 0)
    # The bob's position is given by adding the relative displacement from the pivot.
    x = a * np.cos(w * t) + L * np.sin(theta)
    y = L * np.cos(theta)
    
    # Set up the figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=72)
    
    r = 0.05  # radius for the bob
    
    # Pre-plot the full theta vs time curve.
    ax2.plot(t, theta, 'b-', label=r'$\theta(t)$')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (rad)")
    ax2.set_title("Angle vs Time")
    ax2.grid(True)
    
    # Marker to indicate current time on the theta plot.
    marker, = ax2.plot([], [], 'ro', markersize=8)
    
    def update(i):
        # Clear the pendulum axes.
        ax1.cla()
        
        # Pivot position: now horizontally oscillating.
        pivot_x = a * np.cos(w * t[i])
        pivot_y = 0
        # Draw the rod.
        ax1.plot([pivot_x, x[i]], [pivot_y, y[i]], 'k-', lw=2)
        # Draw the pivot and bob.
        c0 = plt.Circle((pivot_x, pivot_y), r/2, fc='k', zorder=10)
        c1 = plt.Circle((x[i], y[i]), r, fc='r', ec='r', zorder=10)
        ax1.add_patch(c0)
        ax1.add_patch(c1)
        
        # Set limits (taking into account the horizontal oscillation range).
        ax1.set_xlim(-L - a - r, L + a + r)
        ax1.set_ylim(-L - r, L + r)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Pendulum Simulation")
        ax1.axis('off')
    
        # Update the marker on the θ vs t plot.
        marker.set_data(t[i], theta[i])
        return [marker]
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=3, blit=False)
    #ani.save('kapitza_horizontal.mp4', writer='ffmpeg', fps=30)
    plt.tight_layout()
    plt.show()
