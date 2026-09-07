import numpy as np
import matplotlib.pyplot as plt

from models import bouncing_ball as model
from integrators import explicit_euler as integrator
#from integrators import rk4 as integrator

params = {
    "gravity": 9.81,  
    "mass": 0.2, 
    "restitution": 1.0,  
}

initial_state = np.array([1.0, 0.0])

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

for step, t in enumerate(time_traj[:-1]):
    next_state = integrator(model.dynamics, t, state_traj[:, step], timestep, params) 
    next_state = model.handle_impact(next_state, params)
    state_traj[:, step + 1] = next_state

height_traj = state_traj[0, :]
velocity_traj = state_traj[1, :]

potential_energy, kinetic_energy = model.calculate_energy(state_traj, params) 

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Bouncing ball energy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(time_traj, height_traj)
plt.ylabel("Height (m)")
plt.title("Bouncing ball state")

plt.subplot(2, 1, 2)
plt.plot(time_traj, velocity_traj)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(height_traj, velocity_traj)
plt.scatter(initial_state[0],initial_state[1],color="red",label="Initial state")
plt.axvline(0.0, color="black", linestyle="--", label="Ground")
plt.xlabel("Height h (m)")
plt.ylabel("Vertical velocity v (m/s)")
plt.title("Bouncing ball phase portrait")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()