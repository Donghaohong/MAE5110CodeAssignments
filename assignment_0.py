import numpy as np
import matplotlib.pyplot as plt
import timeit

from models import pendulum as model
from integrators import explicit_euler as integrator
#from integrators import rk4 as integrator

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 1e-5
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

def run_simulation():
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state


    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator(model.dynamics, t, state_traj[:, step], timestep, params)   

    return state_traj 

timing_results = timeit.repeat(run_simulation, repeat=5, number=1)   

time = min(timing_results)

print(f"Integrator: {integrator.__name__}")
print(f"Timestep: {timestep}")
print(f"Timing results: {timing_results}")
print(f"Best time: {time} s")

state_traj = run_simulation()
    
# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

potential_energy, kinetic_energy = model.calculate_energy(state_traj, params)

relative_error = model.calculate_energy_error(model.calculate_energy, state_traj, params)

print(f"Maximum relative energy error: {relative_error:.2%}")

if relative_error < 0.01:
    print("PASS: energy error is below 1%.")
else:
    print("FAIL: energy error is at least 1%.")

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum energy")
plt.legend()
plt.tight_layout()
plt.show()

# TODO: make a phase portrait plot

plt.figure()
plt.plot(state_traj[0, :], state_traj[1, :])
plt.scatter(initial_state[0],initial_state[1],color="red",label="Initial state")
plt.xlabel("Angle θ (rad)")
plt.ylabel("Angular velocity ω (rad/s)")
plt.title("Pendulum phase portrait")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
