from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

FRAMES_DIR = Path('frames')

m1, m2 = 1, 1  # masses of the bobs
L1, L2 = 1, 1  # lengths of the pendulums
a = 0.1  # amplitude of the pivot oscillation
w = 80  # frequency of the pivot oscillation
g = 9.81  # acceleration due to gravity

def deriv(t, y, L1, L2, a, w):
    """Return derivatives [theta1, thetadot1, theta2, thetadot2] for the double Kapitza pendulum."""
    theta1, thetadot1, theta2, thetadot2 = y
    wt = w * t
    cwt = np.cos(wt)
    sth1, cth1 = np.sin(theta1), np.cos(theta1)
    sth2, cth2 = np.sin(theta2), np.cos(theta2)
    
    # Define masses (adjust as needed)
    m1 = 1
    m2 = 1

    # Compute an intermediate term D that helps decouple the equations.
    # D = a*w^2/L1*sth1*cwt - g/L1*sth1 - (m2*L2/(m1+m2)/L1)*thetadot2^2*sth2
    D = a * w**2 / L1 * sth1 * cwt - g / L1 * sth1 - (m2 * L2) / ((m1 + m2) * L1) * thetadot2**2 * sth2
    # C is a coupling term.
    C = (m2 * L2) / ((m1 + m2) * L1) * cth2

    # Solve the simultaneous equations:
    # X = thetadotdot1, Y = thetadotdot2
    # X + C*Y = D
    # Y = - (g*sth2 + L1*cth1*X_offset - L1*thetadot1^2*sth1) / L2, where X_offset = X appears in the second rod’s dynamics.
    #
    # Substitute X = D - C*Y into the second equation:
    denom = 1 - (m2 / (m1 + m2)) * cth1 * cth2  # Simplified denominator
    Y = - (g * sth2 + L1 * cth1 * D - L1 * thetadot1**2 * sth1) / (L2 * denom)
    X = D - C * Y

    return [thetadot1, X, thetadot2, Y]

if __name__ == "__main__":
    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = 5, 1/w/20
    t = np.arange(0, tmax, dt)
    # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
    y0 = [0.1, 0, 0.1, 0]  # initial angles and angular velocities
    
    tspan = (0, tmax)
    sol = solve_ivp(deriv, tspan, y0, t_eval=t, args=(L1, L2, a, w))
    theta1, theta2 = sol.y[0], sol.y[2]
    
    # Compute pendulum bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -a * np.cos(w * t) + L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 + L2 * np.cos(theta2)
    
    # Set up the figure with two subplots: one for the pendulum and one for θ vs t.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=72)
    
    r = 0.05  # radius for the bobs
    
    # Pre-plot the full theta vs time curves on ax2.
    ax2.plot(t, theta1, 'b-', label=r'$\theta_1(t)$')
    ax2.plot(t, theta2, 'g-', label=r'$\theta_2(t)$')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (rad)")
    ax2.set_title("Angle vs Time")
    ax2.grid(True)
    ax2.legend()
    
    # This marker will indicate the current time on the theta plot.
    marker1, = ax2.plot([], [], 'ro', markersize=8)
    marker2, = ax2.plot([], [], 'go', markersize=8)
    
    def update(i):
        # Pendulum animation in ax1.
        ax1.cla()
        pivot_y = -a * np.cos(w * t[i])
        ax1.plot([0, x1[i]], [pivot_y, y1[i]], 'k-', lw=2)
        ax1.plot([x1[i], x2[i]], [y1[i], y2[i]], 'k-', lw=2)
        c0 = plt.Circle((0, pivot_y), r/2, fc='k', zorder=10)
        c1 = plt.Circle((x1[i], y1[i]), r, fc='r', ec='r', zorder=10)
        c2 = plt.Circle((x2[i], y2[i]), r, fc='b', ec='b', zorder=10)
        ax1.add_patch(c0)
        ax1.add_patch(c1)
        ax1.add_patch(c2)
        ax1.set_xlim(-L1 - L2 - r, L1 + L2 + r)
        ax1.set_ylim(-L1 - L2 - a - r, L1 + L2 + a + r)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Double Kapitza Pendulum Simulation")
        ax1.axis('off')
    
        # Update the markers on the θ vs t plot in ax2.
        marker1.set_data(t[i], theta1[i])
        marker2.set_data(t[i], theta2[i])
        return [marker1, marker2]  # returning animated artists
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=5, blit=False)
    #ani.save('double_kapitza.mp4', writer='ffmpeg', fps=30)
    plt.tight_layout()
    plt.show()