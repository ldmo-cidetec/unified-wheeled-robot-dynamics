from .mobile_robot_dynamics import MobileRobotDynamics
import numpy as np


class DifferentialDynamics(MobileRobotDynamics):
    """
    Euler-Lagrange dynamic model of the differential-drive robot
    (mobility class (2,0), non-holonomic).

    Derived via the unified eight-stage framework of:
        Pantoja-Garcia et al. (2026). A Unified Euler-Lagrange Framework for the
        Dynamic Modeling of Wheeled Mobile Robots: Holonomic and Non-Holonomic
        Architectures. Mathematics.

    Geometric notation
    ---------------------------------
    r : wheel radius (m)
    L : transverse distance between the two drive wheels (m)

    State vector  q_tilde = [x, y, theta, x_dot_m, omega]^T in R^5.
    Reduced velocity  eta = [x_dot_m, omega]^T in R^2.

    Low-level Jacobian and pseudo-inverse:
        J   = [[1, 0, -L/2],  in R^{2x3}
               [1, 0,  L/2]]
        J^+ = [[ 1/2,  1/2],  in R^{3x2}
               [   0,    0],
               [-1/L, 1/L ]]

    Reduced matrices -- all constant:
        M_tilde = diag(m, Iz)
        C_tilde = 0_{2x2}
        B_tilde = diag(b_v, b_omega)
        E_tilde = (1/r) [[1, 1], [-L/2, L/2]]

    Control-affine model:
        f = [x_dot_m cos(theta), x_dot_m sin(theta), omega,
             -b_v x_dot_m/m, -b_omega omega/Iz]^T
        g = [[0_{3x2}], [M_tilde^{-1} E_tilde]]

    Input  u = [tau_1, tau_2]^T  (wheel torques, N.m).

    Pseudo-inverse of J_u
    ---------------------------------
    J_u(theta) = [[cos(theta), 0],  in R^{3x2}
                  [sin(theta), 0],
                  [0,          1]]

    The columns are orthonormal (J_u^T J_u = I_2), so J_u^+ = J_u^T exactly:
        J_u^+ = [[cos(theta), sin(theta), 0],  in R^{2x3}
                 [0,          0,          1]]
        d/dt J_u^+ = omega * [[-sin(theta), cos(theta), 0],
                               [0,           0,          0]]

    where omega = state[4].
    """

    def __init__(self, m: float, Iz: float, r: float, L: float,
                 bv: float = 0.0, bw: float = 0.0):
        """
        Parameters
        ----------
        m  : float  Total chassis mass (kg).
        Iz : float  Total axial moment of inertia (kg.m^2).
        r  : float  Wheel radius (m).
        L  : float  Transverse distance between the two drive wheels (m).
        bv : float  Translational viscous damping coefficient (N.s/m).
        bw : float  Rotational viscous damping coefficient (N.m.s/rad).
        """
        super().__init__(m=m, Iz=Iz, bv=bv, bw=bw)
        if r <= 0:
            raise ValueError("Wheel radius r must be positive.")
        if L <= 0:
            raise ValueError("Track width L must be positive.")
        self.r = float(r)
        self.L = float(L)

        # Precomputed constant reduced matrices.
        self._M_r     = np.diag([self.m, self.Iz])
        self._E_r     = (1.0 / self.r) * np.array([[1.0, 1.0],[-self.L / 2.0, self.L / 2.0]])
        self._B_r     = np.diag([self.bv, self.bw])
        self._M_r_inv = np.diag([1.0 / self.m, 1.0 / self.Iz])

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def input_dim(self) -> int:
        return 2

    # ------------------------------------------------------------------
    # Reduced dynamic matrices
    # ------------------------------------------------------------------

    def S(self, state: np.ndarray) -> np.ndarray:
        """
        Null-space basis S(q) = J_u in R^{3x2}.
            S = [[cos(theta), 0],
                 [sin(theta), 0],
                 [0,          1]]

        Satisfies A(q) S(q) = 0  with  A(q) = [-sin(theta), cos(theta), 0].
        """
        theta = float(state[2])
        return np.array([[np.cos(theta), 0.0],
                         [np.sin(theta), 0.0],
                         [0.0,           1.0]])

    def M_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced inertia M_tilde = diag(m, Iz) in R^{2x2}. Constant."""
        return self._M_r.copy()

    def C_r(self, state: np.ndarray) -> np.ndarray:
        """
        Reduced Coriolis C_tilde = 0_{2x2}.

        Vanishes because the cross-terms in S^T M S_dot cancel identically:
        -sin(theta) cos(theta) + cos(theta) sin(theta) = 0.
        """
        return np.zeros((2, 2))

    def B_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced damping B_tilde = diag(b_v, b_omega) in R^{2x2}. Constant."""
        return self._B_r.copy()

    def E_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced input E_tilde = (1/r) [[1,1], [-L/2, L/2]] in R^{2x2}. Constant."""
        return self._E_r.copy()

    # ------------------------------------------------------------------
    # Pseudo-inverse of J_u
    # ------------------------------------------------------------------

    def J_u_pinv(self, state: np.ndarray) -> np.ndarray:
        """
        Moore-Penrose pseudo-inverse J_u^+ in R^{2x3}.

        Because J_u^T J_u = I_2, J_u^+ = J_u^T exactly:
            J_u^+ = [[cos(theta), sin(theta), 0],
                     [0,          0,          1]]

        Parameters
        ----------
        state : (5,) array_like
        """
        theta = float(state[2])
        return np.array([[np.cos(theta), np.sin(theta), 0.0],
                         [0.0,           0.0,           1.0]])

    def J_u_pinv_dot(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Time derivative d/dt J_u^+ in R^{2x3}.

        Since J_u^+ = J_u^T, differentiating with theta_dot = omega = state[4]:
            d/dt J_u^+ = omega * [[-sin(theta), cos(theta), 0],
                                   [0,           0,          0]]

        Parameters
        ----------
        state   : (5,) array_like
        **kwargs: unused; accepted for interface consistency
        """
        theta = float(state[2])
        omega = float(state[4])
        return omega * np.array([[-np.sin(theta), np.cos(theta), 0.0],
                                  [0.0,            0.0,           0.0]])

    # ------------------------------------------------------------------
    # Control-affine state-space model
    # ------------------------------------------------------------------

    def f(self, state: np.ndarray) -> np.ndarray:
        """
        Drift vector field with viscous friction:
            f = [x_dot_m cos(theta), x_dot_m sin(theta), omega,
                 -b_v x_dot_m/m, -b_omega omega/Iz]^T
        """
        state = np.asarray(state, dtype=float)
        theta, x_dot_m, omega = state[2], state[3], state[4]
        return np.array([x_dot_m * np.cos(theta),
                         x_dot_m * np.sin(theta),
                         omega,
                         -self.bv * x_dot_m     / self.m,
                         -self.bw * omega / self.Iz])

    def g(self, state: np.ndarray) -> np.ndarray:
        """
        Input matrix field in R^{5x2}:
            g = [[0_{3x2}            ],
                 [M_tilde^{-1} E_tilde]]

        Constant (M_tilde and E_tilde are pose-independent).
        """
        return np.vstack([np.zeros((3, 2)), self._M_r_inv @ self._E_r])

    def __str__(self) -> str:
        return "differential_dynamics"
