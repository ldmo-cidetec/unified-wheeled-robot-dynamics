"""
Circular trajectory experiment: CT+PD vs PD, four WMR architectures.
Displays a 2x2 grid of trajectory plots. Nothing is saved to disk.

Controllers:
  CT+PD:  bar_u = M_r(eta_dot_ref + Kd e_eta + S^T Kp e_q) + C_r eta_ref + B_r eta_ref
  PD:     bar_u = Kd e_eta + S^T Kp e_q
  u = E_r^T (E_r E_r^T)^{-1} bar_u  (torque allocation)

Gain matrices (adapted per architecture):
  Kp = 50 I_3  in R^{3x3}  (all architectures)
  Kd = 30 I_2  in R^{2x2}  (differential)
  Kd = 30      in R        (ackermann)
  Kd = 30 I_3  in R^{3x3}  (omnidirectional, mecanum)
"""

from mobile_robotics.dynamics.differential_dynamics   import DifferentialDynamics
from mobile_robotics.dynamics.ackermann_dynamics      import AckermannDynamics
from mobile_robotics.dynamics.omnidirectional_dynamics import OmnidirectionalDynamics
from mobile_robotics.dynamics.mecanum_dynamics        import MecanumDynamics

import numpy as np
import matplotlib.pyplot as plt

# -- Simulation parameters -----------------------------------------------------------------------
T, dt   = 10.0, 0.005
t       = np.arange(0.0, T + dt, dt)
n_steps = len(t)

R_circ  = 1.0
cx, cy  = 0.0, 1.0
omega_r = 2.0 * np.pi / T
V_ref   = R_circ * omega_r

# -- Gain matrices -----------------------------------------------------------------------
KP      = 50.0 * np.eye(3)          # R^{3x3}, shared by all architectures
KD = {
    "differential":    30.0 * np.eye(2),   # R^{2x2}
    "ackermann":       np.array([[30.0]]),  # R^{1x1} (scalar)
    "omnidirectional": 30.0 * np.eye(3),   # R^{3x3}
    "mecanum":         30.0 * np.eye(3),   # R^{3x3}
}

# -- Physical and friction parameters ----------------------------------------------------
PARAMS = {
    "differential":    dict(m=14.10, Iz=0.589,  r=0.100, L=0.40),
    "ackermann":       dict(m=15.30, Iz=1.314,  r=0.150, L=0.40, W=0.60),
    "omnidirectional": dict(m=10.90, Iz=0.351,  r=0.075, R=0.20),
    "mecanum":         dict(m=15.00, Iz=0.834,  r=0.0895, l=0.20, w=0.225),
}
FRICTION = {key: dict(bv=5e-4, bw=5e-4) for key in PARAMS}

# -- Reference trajectory -----------------------------------------------------------------------
def reference_trajectory(t_vec):
    phi  = omega_r * t_vec
    x_r  =  cx + R_circ * np.sin(phi)
    y_r  =  cy - R_circ * np.cos(phi)
    th_r =  phi
    xd   =  R_circ * omega_r * np.cos(phi)
    yd   =  R_circ * omega_r * np.sin(phi)
    thd  =  np.full_like(t_vec, omega_r)
    xdd  = -R_circ * omega_r**2 * np.sin(phi)
    ydd  =  R_circ * omega_r**2 * np.cos(phi)
    thdd =  np.zeros_like(t_vec)
    return x_r, y_r, th_r, xd, yd, thd, xdd, ydd, thdd


# -- Helpers ------------------------------------------------------------------------------------
def torque_alloc(E_r, bar_u):
    """u = E_r^T (E_r E_r^T)^{-1} bar_u"""
    return E_r.T @ np.linalg.solve(E_r @ E_r.T, bar_u)

def compute_refs(robot, state_ref, q_dot_ref, q_ddot_ref, **kw):
    Jp  = robot.J_u_pinv(state_ref)
    Jpd = robot.J_u_pinv_dot(state_ref, **kw)
    return Jp @ q_dot_ref, Jp @ q_ddot_ref + Jpd @ q_dot_ref

# -- Generic simulation loop --------------------------------------------------------------------
def simulate(robot, state0, ref, ct, key):
    x_r, y_r, th_r, xd, yd, thd, xdd, ydd, thdd = ref
    state  = state0.copy()
    hist   = np.zeros((n_steps, robot.state_dim))
    hist[0] = state
    Kd = KD[key]

    for k in range(n_steps - 1):
        q_ref      = np.array([x_r[k],  y_r[k],  th_r[k]])
        q_dot_ref  = np.array([xd[k],   yd[k],   thd[k]])
        q_ddot_ref = np.array([xdd[k],  ydd[k],  thdd[k]])

        state_ref      = state.copy()
        state_ref[:3]  = q_ref

        eta_ref, eta_dot_ref = compute_refs(robot, state_ref, q_dot_ref, q_ddot_ref)

        S     = robot.S(state)
        q = state[:3]
        e_q   = q_ref - q
        eta = state[3:3 + S.shape[1]]
        e_eta = eta_ref - eta

        if ct:
            bar_u = (robot.M_r(state) @ (eta_dot_ref + Kd @ e_eta + S.T @ (KP @ e_q))
                     + robot.C_r(state) @ eta
                     + robot.B_r(state) @ eta)
        else:
            bar_u = Kd @ e_eta + S.T @ (KP @ e_q)

        state     = robot.step(state, torque_alloc(robot.E_r(state), bar_u), dt)
        hist[k+1] = state

    return hist


def simulate_ackermann(robot, state0, ref, ct):
    x_r, y_r, th_r, xd, yd, thd, xdd, ydd, thdd = ref
    state  = state0.copy()
    hist   = np.zeros((n_steps, 5))
    hist[0] = state
    Kd = KD["ackermann"]

    for k in range(n_steps - 1):
        q_ref      = np.array([x_r[k],  y_r[k],  th_r[k]])
        q_dot_ref  = np.array([xd[k],   yd[k],   thd[k]])
        q_ddot_ref = np.array([xdd[k],  ydd[k],  thdd[k]])

        state_ref     = state.copy()
        state_ref[:3] = q_ref

        eta_ref, eta_dot_ref = compute_refs(
            robot, state_ref, q_dot_ref, q_ddot_ref, delta_dot=0.0)

        S     = robot.S(state)
        q = state[:3]
        e_q   = q_ref - q
        eta = state[3:4]
        e_eta = eta_ref - eta

        if ct:
            bar_u = (robot.M_r(state) @ (eta_dot_ref + Kd @ e_eta + S.T @ (KP @ e_q))
                     + robot.C_r(state) @ eta
                     + robot.B_r(state) @ eta)
        else:
            bar_u = Kd @ e_eta + S.T @ (KP @ e_q)

        F         = float(bar_u[0])
        state     = robot.step(state, np.array([F, 0.0]), dt)
        hist[k+1] = state

    return hist


# -- Main ---------------------------------------------------------------------------------------
def main():
    ref = reference_trajectory(t)
    x_r, y_r = ref[0], ref[1]

    W_ack    = PARAMS["ackermann"]["W"]
    delta_ss = np.arctan(W_ack * omega_r / V_ref)

    robots = {
        "differential":    DifferentialDynamics(**PARAMS["differential"],    **FRICTION["differential"]),
        "ackermann":       AckermannDynamics(**PARAMS["ackermann"],           **FRICTION["ackermann"]),
        "omnidirectional": OmnidirectionalDynamics(**PARAMS["omnidirectional"], **FRICTION["omnidirectional"]),
        "mecanum":         MecanumDynamics(**PARAMS["mecanum"],               **FRICTION["mecanum"]),
    }
    init = {
        "differential":    np.zeros(5),
        "ackermann":       np.array([0., 0., 0., 0., delta_ss]),
        "omnidirectional": np.zeros(6),
        "mecanum":         np.zeros(6),
    }

    # Run simulations
    hist = {}
    for ct_flag, label in [(True, "ct"), (False, "pd")]:
        for key in ["differential", "omnidirectional", "mecanum"]:
            hist[f"{key}_{label}"] = simulate(robots[key], init[key], ref, ct_flag, key)
        hist[f"ackermann_{label}"] = simulate_ackermann(
            robots["ackermann"], init["ackermann"], ref, ct_flag)

    # -- 2x2 trajectory plot ----------------------------------------------------------------
    KEYS   = ["differential", "ackermann", "omnidirectional", "mecanum"]
    TITLES = [
        "Differential drive",
        "Ackermann steering",
        "Three-wheel omnidirectional (3,0)",
        "Four-wheel Mecanum drive",
    ]
    COLOR_REF = "#000000"
    COLOR_CT  = "#1f77b4"
    COLOR_PD  = "#d62728"

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "lines.linewidth": 1.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, key, title in zip(axes.flat, KEYS, TITLES):
        h_ct = hist[f"{key}_ct"]
        h_pd = hist[f"{key}_pd"]
        ax.plot(x_r, y_r, "--", color=COLOR_REF, lw=1.4, label="Reference", zorder=3)
        ax.plot(h_ct[:, 0], h_ct[:, 1], "-",  color=COLOR_CT, lw=1.6, label="CT+PD", zorder=2)
        ax.plot(h_pd[:, 0], h_pd[:, 1], "-",  color=COLOR_PD, lw=1.4, label="PD",    zorder=2, alpha=0.85)
        ax.scatter(0, 0, color="green", s=25, zorder=5, label="Start")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$x\,\mathrm{(m)}$")
        ax.set_ylabel(r"$y\,\mathrm{(m)}$")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, lw=0.5, alpha=0.7)
        ax.legend(loc="lower right", framealpha=0.85)

    fig.suptitle("Circular trajectory tracking: CT+PD vs PD", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()