from .mobile_robot_dynamics import MobileRobotDynamics
import numpy as np


class MecanumDynamics(MobileRobotDynamics):
    """
    Euler-Lagrange dynamic model of the four-wheeled Mecanum-drive platform
    (mobility class (3,0), holonomic, redundantly actuated).

    Derived via the unified eight-stage framework of:
        Pantoja-Garcia et al. (2026). A Unified Euler-Lagrange Framework for the
        Dynamic Modeling of Wheeled Mobile Robots: Holonomic and Non-Holonomic
        Architectures. Mathematics.

    Geometric notation
    ------------------------------------------
    r : wheel radius (m)
    l : half-length of the chassis (m)
    w : half-width  of the chassis (m)

    Wheel layout and roller angles alpha_i:
        Wheel 1 (front-right):  alpha = +pi/4,  position [ w, -l]
        Wheel 2 (rear-right):   alpha = -pi/4,  position [-w, -l]
        Wheel 3 (front-left):   alpha = -pi/4,  position [ w,  l]
        Wheel 4 (rear-left):    alpha = +pi/4,  position [-w,  l]

    rank(J) = 3 with four wheels -> redundantly actuated (A = 0, lambda = 0, S = I_3).

    State vector  q_tilde = [x, y, theta, x_dot, y_dot, theta_dot]^T in R^6.
    With S = I_3, the reduced velocity eta = q_dot lives in the world frame.

    Low-level Jacobian and pseudo-inverse:
        J   = [[1, -1, -(l+w)],  in R^{4x3}
               [1,  1,  (l+w)],
               [1,  1, -(l+w)],
               [1, -1,  (l+w)]]
        J^+ = (1/4) [[ 1,  1,  1,  1         ],  in R^{3x4}
                      [-1,  1,  1, -1         ],
                      [-1/(l+w), 1/(l+w), -1/(l+w), 1/(l+w)]]

    Reduced matrices:
        M_tilde = diag(m, m, Iz)                        constant
        C_tilde = 0_{3x3}                               S = I_3 -> S_dot = 0
        B_tilde = diag(b_v, b_v, b_omega)               constant
        E_tilde = (1/r) H(theta) J^T                    theta-dependent

    E_tilde expanded (c = cos(theta), s = sin(theta)):
        row 1: (1/r) [c+s,    c-s,    c-s,    c+s  ]
        row 2: (1/r) [s-c,    s+c,    s+c,    s-c  ]
        row 3: (1/r) [-(l+w), (l+w), -(l+w), (l+w) ]

    Control-affine model:
        f = [x_dot, y_dot, theta_dot,
             -b_v x_dot/m, -b_v y_dot/m, -b_omega theta_dot/Iz]^T
        g = [[0_{3x4}], [M_tilde^{-1} E_tilde(theta)]]

    Input  u = [tau_1, tau_2, tau_3, tau_4]^T  (wheel torques, N.m).

    Pseudo-inverse of J_u
    ---------------------------------
    J_u = I_3  ->  J_u^+ = I_3,  d/dt J_u^+ = 0_{3x3}.
    """

    def __init__(self, m: float, Iz: float, r: float, l: float, w: float,
                 bv: float = 0.0, bw: float = 0.0):
        """
        Parameters
        ----------
        m  : float  Total chassis mass (kg).
        Iz : float  Total axial moment of inertia (kg.m^2).
        r  : float  Wheel radius (m).
        l  : float  Half-length of the chassis (m).
        w  : float  Half-width of the chassis (m).
        bv : float  Translational viscous damping coefficient (N.s/m).
        bw : float  Rotational viscous damping coefficient (N.m.s/rad).
        """
        super().__init__(m=m, Iz=Iz, bv=bv, bw=bw)
        if r <= 0:
            raise ValueError("Wheel radius r must be positive.")
        if l <= 0:
            raise ValueError("Half-length l must be positive.")
        if w <= 0:
            raise ValueError("Half-width w must be positive.")
        self.r = float(r)
        self.l = float(l)
        self.w = float(w)

        k = self.l + self.w
        # Low-level kinematic Jacobian J in R^{4x3} (Table 1).
        self._J = np.array([[1.0, -1.0, -k],
                             [1.0,  1.0,  k],
                             [1.0,  1.0, -k],
                             [1.0, -1.0,  k]], dtype=float)

        # Precomputed constant reduced matrices.
        self._M_r     = self.M.copy()
        self._B_r     = self.B_global.copy()
        self._M_r_inv = np.diag([1.0 / self.m, 1.0 / self.m, 1.0 / self.Iz])

    @staticmethod
    def _H(theta: float) -> np.ndarray:
        """Body-to-world rotation matrix H(theta) in R^{3x3}."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def input_dim(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Reduced dynamic matrices
    # ------------------------------------------------------------------

    def S(self, state: np.ndarray) -> np.ndarray:
        """S = I_3 (holonomic: no active constraint, A = 0)."""
        return np.eye(3)

    def M_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced inertia M_tilde = diag(m, m, Iz) in R^{3x3}. Constant."""
        return self._M_r.copy()

    def C_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced Coriolis C_tilde = 0_{3x3}. S = I_3 is constant, so S_dot = 0."""
        return np.zeros((3, 3))

    def B_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced damping B_tilde = diag(b_v, b_v, b_omega) in R^{3x3}. Constant."""
        return self._B_r.copy()

    def E_r(self, state: np.ndarray) -> np.ndarray:
        """
        Reduced input matrix  E_tilde = (1/r) H(theta) J^T  in R^{3x4}.

        Maps wheel torques to world-frame generalized forces.
        """
        return (1.0 / self.r) * self._H(float(state[2])) @ self._J.T

    # ------------------------------------------------------------------
    # Pseudo-inverse of J_u
    # ------------------------------------------------------------------

    def J_u_pinv(self, state: np.ndarray) -> np.ndarray:
        """
        J_u^+ = I_3.  J_u = I_3 for all holonomic platforms.

        Parameters
        ----------
        state : (6,) array_like  (unused)
        """
        return np.eye(3)

    def J_u_pinv_dot(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        d/dt J_u^+ = 0_{3x3}.  J_u^+ = I_3 is constant.

        Parameters
        ----------
        state   : (6,) array_like  (unused)
        **kwargs: unused; accepted for interface consistency
        """
        return np.zeros((3, 3))

    # ------------------------------------------------------------------
    # Control-affine state-space model
    # ------------------------------------------------------------------

    def f(self, state: np.ndarray) -> np.ndarray:
        """
        Drift vector field with viscous friction:
            f = [x_dot, y_dot, theta_dot,
                 -b_v x_dot/m, -b_v y_dot/m, -b_omega theta_dot/Iz]^T
        """
        state = np.asarray(state, dtype=float)
        xd, yd, thd = state[3], state[4], state[5]
        return np.array([xd, yd, thd,
                         -self.bv * xd  / self.m,
                         -self.bv * yd  / self.m,
                         -self.bw * thd / self.Iz])

    def g(self, state: np.ndarray) -> np.ndarray:
        """
        Input matrix field in R^{6x4}:
            g = [[0_{3x4}                    ],
                 [M_tilde^{-1} E_tilde(theta)]]
        """
        return np.vstack([np.zeros((3, 4)), self._M_r_inv @ self.E_r(state)])

    def __str__(self) -> str:
        return "mecanum_dynamics"
