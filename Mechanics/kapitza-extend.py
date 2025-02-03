from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

# You can create a directory for frames if you wish
FRAMES_DIR = Path('frames')

# Physical parameters.
m = 1        # mass of each bob (kg)
L = 1        # length of each rod (m)
a = 0.2      # amplitude of the pivot oscillation (m)
w = 100       # angular frequency of the pivot (rad/s)
g = 9.81     # gravitational acceleration (m/s^2)

# Number of pendulum bobs.
N = 1   # Try N = 1 for the given case; for N>1 you get a multi–bob pendulum.

def deriv(t, y, L, a, w, g, N):
    """
    Returns dy/dt for the N–bob Kapitza pendulum.
    
    The state vector y is of length 2N:
      y = [θ₁, θ₂, …, θ_N,  dθ₁/dt, dθ₂/dt, …, dθ_N/dt].
    
    The equations are obtained from the Lagrangian:
      M(theta) * theta_ddot = b(theta, theta_dot, t),
    where the mass matrix and right-hand side are built below.
    """
    theta = y[:N]       # angles: theta[0] is the first bob, etc.
    theta_dot = y[N:]   # corresponding angular velocities

    # The pivot oscillates vertically: y_p(t) = -a*cos(w*t)
    # so its acceleration is:
    y_p_ddot = a * w**2 * np.cos(w*t)
    
    # Build the mass matrix M (size N x N) and the RHS vector b (length N).
    M = np.zeros((N, N))
    b = np.zeros(N)
    
    # Loop over the bobs.
    # We use 0-indexing: bob i corresponds to i=0,...,N-1.
    for i in range(N):
        coeff_i = (N - i)  # for the i-th equation the “weight” is (N-i)
        M[i, i] = coeff_i * m * L**2
        # Gravitational and pivot acceleration terms:
        b[i] = - coeff_i * m * g * L * np.sin(theta[i]) \
               + coeff_i * m * y_p_ddot * L * np.sin(theta[i])
        # Now add the coupling (nonlinear) terms.
        for j in range(i+1, N):
            coeff_ij = (N - j) * m * L**2
            # Off-diagonal elements (note symmetry):
            M[i, j] = coeff_ij * np.cos(theta[i] - theta[j])
            M[j, i] = coeff_ij * np.cos(theta[i] - theta[j])
            # Coupling (coriolis-like) terms:
            b[i] += coeff_ij * np.sin(theta[i] - theta[j]) * theta_dot[j]**2
            b[j] += - coeff_ij * np.sin(theta[i] - theta[j]) * theta_dot[i]**2

    # Solve for the angular accelerations.
    theta_ddot = np.linalg.solve(M, b)
    return np.concatenate([theta_dot, theta_ddot])

if __name__ == "__main__":
    # Time parameters.
    tmax = 3
    dt = 1/w/20
    t = np.arange(0, tmax, dt)
    tspan = (0, tmax)
    
    # Initial conditions.
    # For N=1: y0 = [theta, theta_dot]. For N>1, we initialize each bob.
    # Here we set all initial angles to 0.1 rad and all angular velocities to 0.
    y0 = np.array([0.1]*N + [0]*N)
    
    # Solve the ODE.
    sol = solve_ivp(deriv, tspan, y0, t_eval=t, args=(L, a, w, g, N))
    theta_sol = sol.y[:N]  # shape: (N, len(t))
    t = sol.t
    
    # Compute the positions of the bobs.
    # The pivot is at (0, -a*cos(w*t)).
    # Then for each bob the position is built by summing the contributions.
    xs = np.zeros((N+1, len(t)))  # xs[0] is the pivot x, xs[1] is first bob, etc.
    ys = np.zeros((N+1, len(t)))
    xs[0, :] = 0
    ys[0, :] = -a * np.cos(w * t)
    
    # For each bob, add the rod contribution.
    for i in range(N):
        xs[i+1, :] = xs[i, :] + L * np.sin(theta_sol[i, :])
        ys[i+1, :] = ys[i, :] + L * np.cos(theta_sol[i, :])
    
    # Set up the figure with two subplots: one for the pendulum and one for the angles vs time.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=72)
    
    r = 0.05  # radius for drawing each bob
    
    # Plot the angle(s) vs time. For N>1 we plot each theta_i.
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    lines = []
    for i in range(N):
        ln, = ax2.plot(t, theta_sol[i, :], '-', color=colors[i],
                       label=rf'$\theta_{i+1}(t)$')
        lines.append(ln)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (rad)")
    ax2.set_title("Angles vs Time")
    ax2.grid(True)
    ax2.legend()
    
    # This marker will indicate the current time on the angle plot.
    marker, = ax2.plot([], [], 'ko', markersize=8)
    
    def update(i):
        # Clear the pendulum axis.
        ax1.cla()
        # Get current pivot position.
        pivot_x = 0
        pivot_y = -a * np.cos(w * t[i])
        
        # Plot the rods connecting pivot to each bob.
        ax1.plot(xs[:, i], ys[:, i], 'k-', lw=2)
        
        # Draw the pivot as a small circle.
        pivot_circle = plt.Circle((pivot_x, pivot_y), r/2, fc='k', zorder=10)
        ax1.add_patch(pivot_circle)
        # Draw each bob.
        for j in range(1, N+1):
            bob_circle = plt.Circle((xs[j, i], ys[j, i]), r, fc=colors[j-1], ec=colors[j-1], zorder=10)
            ax1.add_patch(bob_circle)
        
        # Set equal aspect and limits.
        ax1.set_xlim(-L*(N+1), L*(N+1))
        ax1.set_ylim(-L*(N+1) - a, L*(N+1) + a)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("N–bob Kapitza Pendulum Simulation")
        ax1.axis('off')
        
        # Update the marker on the angle vs time plot (we mark the angle of the first bob).
        marker.set_data(t[i], theta_sol[0, i])
        return [marker]
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), interval=3, blit=False)
    # If you wish to save the animation, you can uncomment the next line.
    #ani.save('kapitza_Nbobs.mp4', writer='ffmpeg', fps=30)
    plt.tight_layout()
    plt.show()
