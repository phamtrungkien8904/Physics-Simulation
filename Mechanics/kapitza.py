from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

FRAMES_DIR = Path('frames')

m, L = 1, 1
a = 0.1
w = 80
g = 9.81

def deriv(t, y, L, a, w):
    """Return the first derivatives of y = theta, dtheta/dt."""
    theta, thetadot = y
    wt = w * t
    cwt = np.cos(wt)
    cth, sth = np.cos(theta), np.sin(theta)
    thetadotdot = a * w**2 / L * sth * cwt - g / L * sth
    return thetadot, thetadotdot

if __name__ == "__main__":
    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = 3, 1/w/20
    t = np.arange(0, tmax, dt)
    # Initial conditions: theta, dtheta/dt.
    y0 = [0.1, 0]  # 3 is not quite pi: the bob points not quite straight up.
    
    tspan = (0, tmax)
    sol = solve_ivp(deriv, tspan, y0, t_eval=t, args=(L, a, w))
    theta = sol.y[0]
    # Compute pendulum bob position.
    # The pivot oscillates vertically: (0, -a*cos(w*t)).
    # To place the bob above the pivot, add L*cos(theta) instead of subtracting.
    x = L * np.sin(theta)
    y = -a * np.cos(w * t) + L * np.cos(theta)
    
    # Set up the figure with two subplots: one for the pendulum and one for θ vs t.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=72)
    
    r = 0.05  # radius for the bob
    
    # Pre-plot the full theta vs time curve on ax2.
    ax2.plot(t, theta, 'b-', label=r'$\theta(t)$')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (rad)")
    ax2.set_title("Angle vs Time")
    ax2.grid(True)
    
    # This marker will indicate the current time on the theta plot.
    marker, = ax2.plot([], [], 'ro', markersize=8)
    
    def update(i):
        # Pendulum animation in ax1.
        ax1.cla()
        pivot_y = -a * np.cos(w * t[i])
        ax1.plot([0, x[i]], [pivot_y, y[i]], 'k-', lw=2)
        c0 = plt.Circle((0, pivot_y), r/2, fc='k', zorder=10)
        c1 = plt.Circle((x[i], y[i]), r, fc='r', ec='r', zorder=10)
        ax1.add_patch(c0)
        ax1.add_patch(c1)
        ax1.set_xlim(-L - r, L + r)
        ax1.set_ylim(-L - a - r, L + a + r)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Pendulum Simulation")
        ax1.axis('off')
    
        # Update the marker on the θ vs t plot in ax2.
        marker.set_data(t[i], theta[i])
        return [marker]  # returning animated artist
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=3, blit=False)
    #ani.save('kapitza.mp4', writer='ffmpeg', fps=30)
    plt.tight_layout()
    plt.show()