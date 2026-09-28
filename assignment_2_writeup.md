## Assignment 2 MAE 5510 - Anna Schaldenbrand
 
## How to Run
 
uv run python assignment_2.py

The code will print `"Assignment 2 script complete!"` when finished.

The writeup is called `assignment_2_writeup.md`.
 
## 1. Sketches

![Model sketches of the inverted pendulum walker](model_sketches.jpg){width=60%}

## 2. Visualization of ROA

![Controller Region of Attraction](output/assignment_2/roa.png){width=60%}

The region of attraction for the PD controller used. Yellow indicates the controller could successfully stabilize the inverted pendulum, while purple indicates the controller could not. 

A damping, Kd term of 2 N*m*s/rad, and a stiffness Kp term of 1 was used in the controller.

Stabilization is determined by if the end of the run time:
1. the angle $\theta$ is within a small tolerance (0.01 rad) from 0 rad
2. the angular velocity is within a small tolerance (0.01 rad/s) from 0 rad/s

The run time for each simulation was 5 s.

## 3. Choice of Poincaré section

A Poincaré section is a surface that the trajectory crosses transversally to be able to create a discrete snapshot of the dynamics at each crossing.

The rimless wheel's Poincaré section was chosen at touchdown, but the angle $\theta$ at touchdown was held constant since alpha (geometry) was fixed. 

However, this walker uses a varying alpha model. The touchdown Poincaré section is no longer ideal as theta $\theta$ is not fixed. Thus, touchdown no longer corresponds to a single fixed $\theta$ value.

Thus, the ideal choice for the Poincaré section is **when the trajectory passes through $\theta=0$**. From the state space plots, all trajectories that are stepping foward will cross $\theta=0$ transversally and just once per step, making it an ideal.

Note: this applies only during normal stepping. It will not be transverse if the angular velocity is 0.

## 4. Verification of grid resolution,

To determine the minimum grid resolution needed, I wanted to determine the lowest resolution at which the number of steps is accurate. I ran simulations for a variety of resolutions (grid sizes) and initial angular velocities ($\dot\theta$) to see what the results converged to. I took the resolution to be the first grid resolution where the results converged.


| Resolution | $\dot\theta$ = 0.5 | $\dot\theta$ = 1.0 | $\dot\theta$ = 1.5 | $\dot\theta$ = 2.0 | $\dot\theta$ = 2.5 | $\dot\theta$ = 3.0 | $\dot\theta$ = 3.5 | $\dot\theta$ = 4.0 |
|---|---|---|---|---|---|---|---|---|
| 16x16 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 4.0 | 4.0 | 4.0 |
| 17x17 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| 18x18 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 4.0 | 4.0 |
| 19x19 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 4.0 |
| 20x20 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| 21x21 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 4.0 |
| 22x22 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 3.0 | 3.0 | 4.0 |


Seen above, small changes in grid size  # of steps appears to converge at 21 x 21. The prior resolution (20x20) appear to be at/around the boundary leading to varying calculations of the number of steps. Thus, 21 x 21 was chosen as the resolution. 


## 5. Sample Trajectory: 3 steps and max steps

![Trajectory comparison: 3-step vs. max-step walk](output/assignment_2/trajectory_comparison.png){width=60%}

The angle ($\theta$) and angular velocity ($\dot\theta$)  trajectory for an initial condition that requires at least 3 steps is shown above.

Additionally, the angle ($\theta$)  and angular velocity ($\dot\theta$) trajectory for an initial condition that requires the maxmimum amount of steps for the range of initial velocities is shown.

## 6. Steps needed to reach standstill

![Steps to standstill vs. initial angular velocity](output/assignment_2/steps_to_standstill.png){width=60%}

For a given inital condition (angular velocity, $\dot\theta$ ), this displays how many steps the walker takes to reach standstill. 

The number of steps rises as the initial angular velocity increases. This makes sense because higher initial angular velocities are harder to correct and would take more steps to get to standstill.