# Assignment 2: Inverted-Pendulum Walker Control

**Donghao Hong (`dh787`)**

## Model and control bounds

The walker is modeled as a point mass supported by a massless stance leg of length $\ell$. The state is $x=[\theta,\dot\theta]$, where $\theta=0$ is the upright configuration and positive rotation is downhill. During continuous motion,

$$
\ddot\theta=\frac{g}{\ell}\sin\theta+\frac{\tau}{m\ell^2}.
$$

A footstrike occurs while moving downhill when $\theta=\gamma+\alpha$. The plastic collision and change of stance leg produce the reset

$$
\theta^+=\theta^- -2\alpha,
\qquad
\dot\theta^+=\dot\theta^-\cos(2\alpha).
$$

The simulations use Earth gravity, $g=9.81\ \mathrm{m/s^2}$, with $m=1\ \mathrm{kg}$, $\ell=1\ \mathrm{m}$, and incline $\gamma=0.06\ \mathrm{rad}$. The control inputs obey

$$
\alpha\in\left[\frac{\pi}{8},\frac{\pi}{7}\right],
\qquad
\tau\in[-0.1mg\ell,\,0.05mg\ell].
$$

## Annotated sketches

The physical sketches show upright mid-stance (A), touchdown with two alternative landing angles (B1 and B2), and a weak post-impact step that reverses before reaching mid-stance (D). B1 and B2 are alternative outcomes from the same initial state, not consecutive footstrikes. The stance and swing legs are separated by $2\alpha$; choosing a larger $\alpha$ moves the downhill touchdown guard to a larger stance angle.

![Hand-drawn walker snapshots at mid-stance, two touchdown angles, and a failed step](assignment_2_sketches/physical_walker.jpg)

*Figure 1. Physical walker snapshots. The slope is inclined by $\gamma$; the ankle torque is applied at the stance foot.*

The corresponding state-space sketch uses $\theta$ on the horizontal axis and $\dot\theta$ on the vertical axis. A lies on the selected Poincaré section $\theta=0$, $\dot\theta>0$. The two touchdown guards are $\theta=\gamma+\pi/8$ and $\theta=\gamma+\pi/7$, with B1 and B2 marking the respective pre-impact states. The dashed jumps from B1 to C1 and from B2 to C2 represent the instantaneous impact resets; the solid curves from C1 and C2 lead toward the next Poincaré crossing. D illustrates a trajectory whose angular velocity reaches zero and then becomes negative before crossing the section. The standing-controller RoA is shown separately in the measured plot below, not estimated from this conceptual sketch.

![Hand-drawn state-space sketch showing the Poincare section, touchdown guards, impact resets, and reversal](assignment_2_sketches/state_space.jpg)

*Figure 2. State-space locations corresponding to the physical snapshots and their post-impact states.*

## Sanity checks

I checked the continuous and impact dynamics separately before constructing the controller. At $\theta=0$, $\dot\theta=0$, and $\tau=0$, the acceleration is zero as expected. For a positive angle with zero torque, gravity accelerates the inverted pendulum farther downhill. The contact guard activates only when the state crosses $\gamma+\alpha$ while moving downhill. At impact, the angle shifts by exactly $2\alpha$, while the angular velocity is reduced by the expected factor $\cos(2\alpha)$. Numerical simulations showed the corresponding discontinuous state jump without an artificial bounce.

## Standing controller and region of attraction

I used feedback linearization to cancel the nonlinear gravity term and imposed proportional-derivative dynamics about the upright equilibrium:

$$
\tau=m\ell^2\left(-k_p\theta-k_d\dot\theta-\frac{g}{\ell}\sin\theta\right),
\qquad k_p=k_d=4.
$$

The commanded torque is clipped to the required torque interval. Outside the estimated region of attraction (RoA), the ankle torque is set to zero. Once a trajectory enters the RoA, the standing controller is enabled.

I estimated the RoA over the finite domain $\theta\in[-\pi/2,\pi/2]$ and $\dot\theta\in[-\sqrt{2g/\ell},\sqrt{2g/\ell}]$ with a $61\times61$ state grid. Once balancing begins, the walking phase is finished and no additional touchdown is scheduled, so this RoA belongs to the controlled one-leg inverted pendulum and is independent of $\alpha$. Each state was simulated for 10 seconds and classified as successful only if it remained within the angular domain and finished with $|\theta|<10^{-3}\ \mathrm{rad}$ and $|\dot\theta|<10^{-3}\ \mathrm{rad/s}$. The controller stabilized 158 of 3721 sampled states, or 4.25% of the selected domain.

![Estimated region of attraction of the standing controller](output/assignment_2/standing_controller_roa.png)

## Poincaré section and lookup-table controller

I chose

$$
\theta=0,\qquad \dot\theta>0
$$

as the Poincaré section. This section is independent of the selected landing angle $\alpha$, unlike the touchdown surface $\theta=\gamma+\alpha$. It is transverse to the relevant downhill flow because $\dot\theta>0$ at each crossing. Since $\theta$ is fixed, the step-to-step state is reduced to the scalar angular velocity $\dot\theta_k$.

The lookup table samples

$$
\dot\theta_k\in\left[0,\sqrt{\frac{2g}{\ell}}\right]
$$

and the full permissible range of $\alpha$. Each state-action pair is simulated until the trajectory either reaches the standing-controller RoA, returns to $\theta=0$, reverses, or falls. Backward reachability then assigns the minimum number of additional steps required to reach the RoA and stores a suitable angle of attack.

### Grid-resolution verification

I tested each lookup policy on 101 independently sampled initial angular velocities. A grid was accepted only if all 101 rollouts reached the RoA and at least 99% of the predicted step counts agreed with the simulated step counts.

| Lookup grid | Successful rollouts | Step-count agreement | Passes criterion |
|---|---:|---:|:---:|
| $51\times21$ | 101/101 (100.00%) | 99/101 (98.02%) | No |
| $61\times25$ | 101/101 (100.00%) | 101/101 (100.00%) | Yes |

The $61\times25$ grid is therefore the coarsest tested grid that meets the stated criterion. The slightly coarser $51\times21$ grid succeeds from every validation state but falls below the required step-count agreement.

## Minimum-step and maximum-step trajectories

I used the initial Poincaré state

$$
(\theta_0,\dot\theta_0)=\left(0,\sqrt{\frac{2g}{\ell}}\right)
=(0,4.42945\ \mathrm{rad/s}).
$$

The minimum-step policy reaches the RoA after three footstrikes, using $\alpha=[25.714^\circ,25.714^\circ,25.580^\circ]$. The full hybrid simulation enters the RoA at $t=0.9740\ \mathrm{s}$ and then converges under the ankle controller.

For the same initial condition, the longest successful sequence found contains six footstrikes, all using $\alpha=22.500^\circ$. It enters the RoA at $t=2.6781\ \mathrm{s}$. In both simulations, the maximum ankle torque before entering the RoA is exactly zero. The final states are approximately $(0.00049,-0.00085)$ for the minimum-step policy and $(0.00050,-0.00085)$ for the maximum-step policy, confirming convergence to the upright equilibrium.

![Minimum-step and maximum-step hybrid trajectories](output/assignment_2/hybrid_policy_trajectories.png)

## Steps required to reach standstill

The final backward-reachability calculation classifies every one of the 61 Poincaré grid states as reachable. Five states are already inside the RoA, 17 require one step, 16 require two steps, and 23 require three steps. The maximum minimum-step count over the sampled state interval is therefore three.

![Minimum steps required to reach the standing-controller RoA](output/assignment_2/steps_to_stand.png)

## Reproduction

From the repository root, run:

```bash
uv run python assignment_2.py
```

This command recomputes the standing-controller RoA, constructs and validates the lookup-table controller, verifies the minimum- and maximum-step hybrid trajectories, and generates the report figures and `walker.gif` under `output/assignment_2/`.

