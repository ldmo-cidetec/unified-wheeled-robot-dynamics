# Wheeled Mobile Robot Dynamics Library

This repository provides a lightweight and self-contained **Python library for the Euler-Lagrange dynamic modeling and simulation of wheeled mobile robots** using a unified null-space-based reduction procedure.
The implementation corresponds to and supports the results presented in the paper:

> **Pantoja-Garcia, J. S., Rodriguez-Molina, A., Villarreal-Cervantes, M. G., Aldape-Perez, M., Sandoval-Gutierrez, J., & Martinez-Vazquez, D. L. (2026).**
> *A Unified Euler-Lagrange Framework for the Dynamic Modeling of Wheeled Mobile Robots: Holonomic and Nonholonomic Architectures.*
> **Mathematics.**

The code focuses on:

- Euler-Lagrange dynamic modeling via a unified eight-stage procedure, yielding either full or reduced dynamic models that preserve the structural properties of Euler-Lagrange mechanical systems
- Null-space-based elimination of Lagrange multipliers (D'Alembert reduction)
- Control-affine state-space representations ready for control synthesis
- Computed-torque plus PD (CT+PD) and plain PD trajectory-tracking controllers
- Clear separation between robot models and simulation scripts
- Readable, extensible, and educational code

The library is intended for:

- Research prototyping in mobile robotics dynamics and control
- Teaching Euler-Lagrange mechanics and nonlinear control
- Reproducible simulation examples

---

## Citation

### APA

```text
Pantoja-Garcia, J. S., Rodriguez-Molina, A., Villarreal-Cervantes, M. G.,
Aldape-Perez, M., Sandoval-Gutierrez, J., & Martinez-Vazquez, D. L. (2026). 
A unified Euler-Lagrange framework for the dynamic modeling of wheeled mobile robots: 
Holonomic and nonholonomic architectures. Mathematics.
```

### BibTeX

```bibtex
@Article{dynamics2026,
  AUTHOR = {Pantoja-Garcia, Jesus Said and Rodriguez-Molina, Alejandro and
            Villarreal-Cervantes, Miguel Gabriel and Aldape-Perez, Mario and 
            Sandoval-Gutierrez, Jacobo and Martinez-Vazquez, Daniel Librado},
  TITLE  = {A Unified {Euler--Lagrange} Framework for the Dynamic Modeling of
            Wheeled Mobile Robots: Holonomic and Nonholonomic Architectures},
  JOURNAL = {Mathematics},
  YEAR   = {2026},
}
```

---

## Supported Robot Models

The following wheeled mobile robot configurations are included:

- **Differential-drive robot**: mobility class (2,0), nonholonomic
- **Ackermann (car-like / bicycle) robot**: mobility class (1,1), nonholonomic
- **Omnidirectional robot (3,0 configuration)**: holonomic, three Swedish wheels
- **Mecanum 4WD robot**: holonomic, redundantly actuated

All models are derived through the same eight-stage Euler-Lagrange procedure and share a common base class and consistent conventions.

---

## Project Structure

```
Project/
|-- mobile_robotics/
|   |-- __init__.py
|   |-- mobile_robot_dynamics.py      # Abstract base class
|   |-- differential_dynamics.py
|   |-- ackermann_dynamics.py
|   |-- omnidirectional_dynamics.py
|   `-- mecanum_dynamics.py
|
`-- dynamics_simulation.py            # CT+PD vs PD trajectory-tracking example
```

- **mobile_robotics/**
  Contains the robot models and the abstract base class `MobileRobotDynamics`.

- **dynamics_simulation.py**
  Main script that simulates circular trajectory tracking for all four architectures
  and displays a $2 \times 2$ comparison plot of CT+PD vs PD controllers.

---

## Dynamic Model

Each architecture is derived from the constrained Euler-Lagrange equations and reduced
to the control-affine state-space form:

$$\dot{\tilde{q}} = f(\tilde{q}) + g(\tilde{q})  u$$

where $\tilde{q} = [q^T,  \eta^T]^T$ stacks the pose $q = [x,  y,  \theta]^T \in \mathbb{R}^3$
and the reduced velocity $\eta \in \mathbb{R}^k$ (dimension $k$ is architecture-dependent).

The reduced dynamic model that drives the lower block is:

$$M_r(q) \dot{\eta} + \bigl(C_r(q,\eta) + B_r(q)\bigr)\eta = E_r(q) u$$

where:

- $M_r = S^T M S \in \mathbb{R}^{k \times k}$: reduced inertia matrix
- $C_r = S^T M \dot{S} \in \mathbb{R}^{k \times k}$: reduced Coriolis/centripetal matrix
- $B_r = S^T B S \in \mathbb{R}^{k \times k}$: reduced viscous damping matrix
- $E_r = S^T E(q) \in \mathbb{R}^{k \times n_w}$: reduced input matrix
- $S(q) \in \mathbb{R}^{n \times k}$: null-space basis of the Pfaffian constraint matrix $A(q)$

For holonomic platforms (omnidirectional and Mecanum), $A = 0$ and $S = I_3$,
so the reduction is trivial and the full world-frame dynamics are recovered directly.

Wheel torques are mapped to the reduced space via the right pseudo-inverse of $\tilde{E}$:

$$u = \tilde{E}^T \bigl(\tilde{E} \tilde{E}^T\bigr)^{-1} \bar{u}$$

---

## Controllers

Two trajectory-tracking control laws are implemented and compared.

**CT+PD (computed-torque plus PD) - model-based:**

$$\bar{u} = M_r \left(\dot{\eta}_\mathrm{ref} + K_d  e_\eta + S^T K_p  e_q\right) + C_r \eta + B_r \eta$$

**PD - model-free:**

$$\bar{u} = K_d  e_\eta + K_p  S^T e_q$$

In both laws:

- $e_q = q_\mathrm{ref} - q \in \mathbb{R}^3$: pose error in world frame
- $e_\eta = \eta_\mathrm{ref} - \eta \in \mathbb{R}^k$: reduced velocity error
- $\eta_\mathrm{ref}$, $\dot{\eta}_\mathrm{ref}$: reference reduced velocity and acceleration, obtained by projecting the world-frame reference through $J_u^+$ and its time derivative

Gain matrices (same scalar values for both controllers, dimensions adapted per architecture):

| Architecture | $K_p$                                      | $K_d$                                 |
|---|--------------------------------------------|---------------------------------------|
| All | $50 \cdot I_3 \in \mathbb{R}^{3 \times 3}$ | $-$                                      |
| Differential | $-$                                        | $30 \cdot I_2 \in \mathbb{R}^{2 \times 2}$ |
| Ackermann | $-$                                        | $30 \in \mathbb{R}$                   |
| Omnidirectional, Mecanum | $-$                                        | $30 \cdot I_3 \in \mathbb{R}^{3 \times 3}$ |

---

## Running the Example

From the project root directory, run:

```bash
python dynamics_simulations.py
```

The script simulates both controllers on a counterclockwise circle of radius $1 \mathrm{m}$
centered at $(0, 1)$ $\mathrm{m}$, traversed in $T = 10$ $\mathrm{s}$ from rest at the origin.
A $2 \times 2$ figure is displayed comparing CT+PD and PD trajectories for all four architectures.

---

## Base Class Interface

`MobileRobotDynamics` (in `mobile_robot_dynamics.py`) defines the abstract interface
that every architecture must implement:

| Method | Returns | Description |
|---|---|---|
| `S(state)` | $\mathbb{R}^{n \times k}$ | Null-space basis (same as $J_u$) |
| `M_r(state)` | $\mathbb{R}^{k \times k}$ | Reduced inertia matrix |
| `C_r(state)` | $\mathbb{R}^{k \times k}$ | Reduced Coriolis/centripetal matrix |
| `B_r(state)` | $\mathbb{R}^{k \times k}$ | Reduced viscous damping matrix |
| `E_r(state)` | $\mathbb{R}^{k \times n_w}$ | Reduced input matrix |
| `J_u_pinv(state)` | $\mathbb{R}^{k \times n}$ | Moore-Penrose pseudo-inverse of $J_u$ |
| `J_u_pinv_dot(state, **kw)` | $\mathbb{R}^{k \times n}$ | Time derivative of $J_u^+$ |
| `f(state)` | $\mathbb{R}^{n+k}$ | Drift vector field (with viscous friction) |
| `g(state)` | $\mathbb{R}^{(n+k) \times n_w}$ | Input matrix field |
| `step(state, u, dt)` | $\mathbb{R}^{n+k}$ | Euler integration step |

The `step` method is shared by all architectures and uses explicit Euler integration.

---

## Adding a New Robot Model

To add a new wheeled mobile robot configuration:

### 1. Create a new robot class

Add a new file in `mobile_robotics/`, for example:

```
mobile_robotics/my_new_robot.py
```

The class should:

- Inherit from `MobileRobotDynamics`
- Implement all abstract methods listed in the table above
- Define geometric parameters (wheel radius, track width, etc.) in `__init__`

Use existing robots as references:

- `differential_dynamics.py`: simplest nonholonomic case, constant matrices
- `ackermann_dynamics.py`: state-dependent scalar matrices, extra kinematic input
- `omnidirectional_dynamics.py`: holonomic, $S = I_3$, $\theta$-dependent $E_r$
- `mecanum_dynamics.py`: holonomic, redundantly actuated ($n_w > k$)

### 2. Add it to the simulation script

Import the new class in `dynamics_simulation.py` and add an entry to the `PARAMS`,
`FRICTION`, `robots`, and `init` dictionaries in `main()`.

---

## Numerical Integration

All simulations use **explicit Euler integration**:

$$\tilde{q}_{k+1} = \tilde{q}_k + \bigl(f(\tilde{q}_k) + g(\tilde{q}_k) u_k\bigr) \Delta t$$

The integration step is $\Delta t = 5$ $\mathrm{ms}$. This choice is intentional:

- Simple and transparent
- Consistent with the paper's simulation setup
- Easy to replace with higher-order integrators if required

---

## Dependencies

The project uses only standard scientific Python libraries:

- Python >= 3.8
- NumPy
- Matplotlib

No additional frameworks or build tools are required.

---

## Design Principles

- Minimal dependencies
- One-to-one correspondence with the paper's mathematical notation
- Explicit assumptions and simplifications (negligible wheel inertia, centered mass)
- Easily extensible to new architectures or higher-fidelity models
- Suitable for education and research
