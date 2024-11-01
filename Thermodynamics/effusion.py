# SIMULATION OF EFFUSION OF GAS THROUGH A HOLE IN A WALL #
# Original code by Dr. Christian Hill. Email: christian@scipython.com
# URL: https://scipython.com/blog/the-maxwellboltzmann-distribution-in-two-dimensions/
# This code is currently using Matplotlib version 3.6.3, higher versions may not work.

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X, Y = 0, 1
class MDSimulation:

    def __init__(self, pos, vel, r, m):
        """
        Initialize the simulation with identical, circular particles of radius
        r and mass m. The n x 2 state arrays pos and vel hold the n particles'
        positions in their rows as (x_i, y_i) and (vx_i, vy_i).

        """

        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.n = self.pos.shape[0]
        self.r = r
        self.m = m
        self.nsteps = 0

    def advance(self, dt):
        """Advance the simulation by dt seconds."""

        self.nsteps += 1
        # Update the particles' positions according to their velocities.
        self.pos += self.vel * dt
        # Find indices for all unique collisions.
        dist = squareform(pdist(self.pos))
        iarr, jarr = np.where(dist < 2 * self.r)
        k = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        # For each collision, update the velocities of the particles involved.
        for i, j in zip(iarr, jarr):
            pos_i, vel_i = self.pos[i], self.vel[i]
            pos_j, vel_j =  self.pos[j], self.vel[j]
            rel_pos, rel_vel = pos_i - pos_j, vel_i - vel_j
            r_rel = rel_pos @ rel_pos
            v_rel = rel_vel @ rel_pos
            v_rel = 2 * rel_pos * v_rel / r_rel - rel_vel
            v_cm = (vel_i + vel_j) / 2
            self.vel[i] = v_cm - v_rel/2
            self.vel[j] = v_cm + v_rel/2

        # Bounce the particles off the walls where necessary, by reflecting
        # their velocity vectors.
        hit_left_wall = self.pos[:, X] < self.r
        hit_right_wall = self.pos[:, X] > 1 - self.r
        hit_bottom_wall = self.pos[:, Y] < self.r
        hit_top_wall = self.pos[:, Y] > 1 - self.r
        self.vel[hit_left_wall | hit_right_wall, X] *= -1
        self.vel[hit_bottom_wall | hit_top_wall, Y] *= -1

        hit_middle_wall = (  (  (self.pos[:, Y] < (0.5 - hole_r))
                              | (self.pos[:, Y] > (0.5 + hole_r))
                             )
                           & (  (self.pos[:, X] > (0.5 - self.r / 2))
                              & (self.pos[:, X] < (0.5 + self.r / 2))
                             )
                          )
        self.vel[hit_middle_wall] *= -1

# Number of particles.
n = 500
# Scaling factor for distance, m-1. The box dimension is therefore 1/rscale.
rscale = 5.e6
# Use the van der Waals radius of Ar, about 0.2 nm.
r = 20e-10 * rscale
# Scale time by this factor, in s-1.
tscale = 1e9    # i.e. time will be measured in nanoseconds.
# Take the mean speed to be the 150 m.s-1.
sbar = 150 * rscale / tscale
# Time step in scaled time units.
FPS = 60
dt = 1/FPS
# Particle masses, scaled by some factor we're not using yet.
m = 1

# "Radius" (i.e. half-width) of the one-dimensional hole in the middle wall.
hole_r = 0.02

# Initialize the particles' positions randomly in the LHS of the box.
pos = np.random.random((n, 2)) * (0.5, 1)
# Initialize the particles velocities with random orientations and random
# magnitudes  around the mean speed, sbar.
theta = np.random.random(n) * 2 * np.pi
s0 = sbar * np.random.random(n)
vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T

sim = MDSimulation(pos, vel, r, m)

# Set up the Figure and make some adjustments to improve its appearance.
DPI = 100
width, height = 1000, 500
fig = plt.figure(figsize=(width/DPI, height/DPI), dpi=DPI)
fig.subplots_adjust(left=0, right=0.97)
sim_ax = fig.add_subplot(121, aspect='equal', autoscale_on=False)
sim_ax.set_xticks([])
sim_ax.set_yticks([])
# Make the box walls a bit more substantial.
for spine in sim_ax.spines.values():
    spine.set_linewidth(2)

npart_ax = fig.add_subplot(122)
npart_ax.set_xlabel(r'Time, $t\;/\mathrm{ns}$')
npart_ax.set_ylabel('Number of particles')
npart_ax.set_xlim(0, 100)
npart_ax.set_ylim(0, n)
npart_ax.axhline(n/2, 0, 1, color='k', lw=1)

particles, = sim_ax.plot([], [], 'o', color='k')
sim_ax.vlines(0.5, 0, 0.5 - hole_r, lw=2, color='k')
sim_ax.vlines(0.5, 0.5 + hole_r, 1, lw=2, color='k')
sim_ax.axvspan(0, 0.5, 0., 1, facecolor='tab:blue', alpha=0.3)
sim_ax.axvspan(0.5, 1, 0., 1, facecolor='tab:orange', alpha=0.3)

LHSlabel_pos = 0.25, 1.05
LHSlabel = sim_ax.text(*LHSlabel_pos, 'LHS: {:d}'.format(n), ha='center')
RHSlabel_pos = 0.75, 1.05
RHSlabel = sim_ax.text(*RHSlabel_pos, 'RHS: 0', ha='center')

RHSline, = npart_ax.plot([0], [0], c='k', label='RHS')
t, nLHS, nRHS = [], [], []

def animate(i):
    """Advance the animation by one step and update the frame."""
    global sim, verts
    sim.advance(dt)


    particles.set_data(sim.pos[:, X], sim.pos[:, Y])
    particles.set_markersize(4)

    t.append(i*dt)
    nLHS = sum(sim.pos[:, X] < 0.5)
    nRHS.append(n - nLHS)

    LHSlabel.set_text('LHS: {:d}'.format(nLHS))
    RHSlabel.set_text('RHS: {:d}'.format(nRHS[-1]))

    RHSline.set_data(t, nRHS)

    npart_ax.collections.clear()
    npart_ax.fill_between(t, nRHS, facecolor='tab:orange', alpha=0.3)
    npart_ax.fill_between(t, nRHS, n, facecolor='tab:blue', alpha=0.3)

    return particles, LHSlabel, RHSlabel, RHSline

# Number of frames; set to None to run until explicitly quit.
nframes = 100*FPS
anim = FuncAnimation(fig, animate, frames=nframes, interval=10)
# anim.save('effusion.mp4')
plt.show()
