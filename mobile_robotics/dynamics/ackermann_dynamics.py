from .mobile_robot_dynamics import MobileRobotDynamics
import numpy as np


class AckermannDynamics(MobileRobotDynamics):
    """
    Euler-Lagrange dynamic model of the Ackermann-steering vehicle
    (mobility class (1,1), non-holonomic).

    Derived via the unified eight-stage framework of:
        Pantoja-Garcia et al. (2026). A Unified Euler-Lagrange Framework for the
        Dynamic Modeling of Wheeled Mobile Robots: Holonomic and Non-Holonomic
        Architectures. Mathematics.

    Geometric notation
    ---------------------------------
    r : rear wheel radius (m)
    L : transverse distance between the two rear drive wheels (m)
    W : wheelbase -- rear axle to front steering axle (m)

    State vector  q_tilde = [x, y, theta, x_dot_m, delta]^T in R^5.
    Reduced velocity  eta = x_dot_m in R  (scalar).

    The steering angle delta is a kinematic input from an independent servo loop,
    not a generalized coordinate of the Lagrangian.

    Low-level Jacobian and pseudo-inverse (rear axle):
        J   = [[1, 0, -L/2],  in R^{2x3}
               [1, 0,  L/2]]
        J^+ = [[ 1/2,  1/2],  in R^{3x2}
               [   0,    0],
               [-1/L, 1/L ]]

    Reduced scalars -- all delta-dependent:
        M_tilde(delta)         = m + Iz tan^2(delta)/W^2
        C_tilde(delta,d_delta) = Iz tan(delta) sec^2(delta)/W^2 * delta_dot  [coefficient only in C_r]
        B_tilde(delta)         = b_v + b_omega tan^2(delta)/W^2
        E_tilde                = 1

    Note: C_r() returns the state-dependent coefficient without tau_delta = delta_dot.
    The full Coriolis term C_coeff * delta_dot * x_dot_m enters through g() as u[1].

    Control-affine model:
        f = [x_dot_m cos(theta), x_dot_m sin(theta), x_dot_m tan(delta)/W,
             -B_tilde/M_tilde * x_dot_m, 0]^T
        g = [[0,          0                        ],
             [0,          0                        ],
             [0,          0                        ],
             [1/M_tilde, -C_coeff * x_dot_m/M_tilde],
             [0,          1                        ]]

    Input  u = [F, tau_delta]^T  (traction force N; steering rate rad/s).

    Pseudo-inverse of J_u
    ---------------------------------
    J_u(theta, delta) = [cos(theta), sin(theta), tan(delta)/W]^T  in R^{3x1}

    Let  sigma^2 = ||J_u||^2 = 1 + tan^2(delta)/W^2.  Then:
        J_u^+ = (1/sigma^2) [cos(theta), sin(theta), tan(delta)/W]   in R^{1x3}

    Full time derivative:
        d/dt J_u^+ = -(2 tan sec^2/W^2 * delta_dot)/sigma^4 * [cos, sin, tan/W]
                   + (1/sigma^2) * [-sin*omega, cos*omega, sec^2/W * delta_dot]
    where  omega = x_dot_m * tan(delta)/W  (bicycle kinematic constraint)
    and    delta_dot is passed as keyword argument (it is the control input u[1]).
    """

    def __init__(self, m: float, Iz: float, r: float, L: float, W: float,
                 bv: float = 0.0, bw: float = 0.0):
        """
        Parameters
        ----------
        m  : float  Total chassis mass (kg).
        Iz : float  Total axial moment of inertia (kg.m^2).
        r  : float  Rear wheel radius (m).
        L  : float  Transverse distance between the two rear drive wheels (m).
        W  : float  Wheelbase (m).
        bv : float  Translational viscous damping coefficient (N.s/m).
        bw : float  Rotational viscous damping coefficient (N.m.s/rad).
        """
        super().__init__(m=m, Iz=Iz, bv=bv, bw=bw)
        if r <= 0:
            raise ValueError("Wheel radius r must be positive.")
        if L <= 0:
            raise ValueError("Rear track width L must be positive.")
        if W <= 0:
            raise ValueError("Wheelbase W must be positive.")
        self.r = float(r)
        self.L = float(L)
        self.W = float(W)

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
        Null-space basis S(q,delta) = J_u in R^{3x1}.
            S = [cos(theta), sin(theta), tan(delta)/W]^T

        Satisfies A(q) S = 0 with the 2x3 Pfaffian constraint matrix A(q).
        """
        theta, delta = float(state[2]), float(state[4])
        return np.array([[np.cos(theta)],
                         [np.sin(theta)],
                         [np.tan(delta) / self.W]])

    def M_r(self, state: np.ndarray) -> np.ndarray:
        """
        Reduced inertia scalar  M_tilde(delta) = m + Iz tan^2(delta)/W^2.

        Grows as |delta| -> pi/2. Returned as (1,1) for interface consistency.
        """
        delta = float(state[4])
        return np.array([[self.m + self.Iz * np.tan(delta)**2 / self.W**2]])

    def C_r(self, state: np.ndarray) -> np.ndarray:
        """
        Reduced Coriolis coefficient  C_coeff(delta) = Iz tan(delta) sec^2(delta)/W^2.

        Returns the state-dependent coefficient only (without delta_dot).
        The full Coriolis force  C_coeff * delta_dot * x_dot_m  enters through g()
        as u[1], so it is absent from f() to avoid double-counting.
        Returned as (1,1) for interface consistency.
        """
        delta = float(state[4])
        return np.array([[self.Iz * np.tan(delta) / (np.cos(delta)**2 * self.W**2)]])

    def B_r(self, state: np.ndarray) -> np.ndarray:
        """
        Reduced damping scalar  B_tilde(delta) = b_v + b_omega tan^2(delta)/W^2.

        Derivation: S^T diag(b_v, b_v, b_omega) S
            = b_v cos^2(theta) + b_v sin^2(theta) + b_omega tan^2(delta)/W^2
            = b_v + b_omega tan^2(delta)/W^2.
        Returned as (1,1) for interface consistency.
        """
        delta = float(state[4])
        return np.array([[self.bv + self.bw * np.tan(delta)**2 / self.W**2]])

    def E_r(self, state: np.ndarray) -> np.ndarray:
        """
        Reduced input scalar  E_tilde = S^T E(q) = 1.

        Derivation: E(q) = [cos(theta), sin(theta), 0]^T, so
        S^T E(q) = cos^2(theta) + sin^2(theta) + 0 = 1.
        Returned as (1,1) for interface consistency.
        """
        return np.array([[1.0]])

    # ------------------------------------------------------------------
    # Pseudo-inverse of J_u
    # ------------------------------------------------------------------

    def J_u_pinv(self, state: np.ndarray) -> np.ndarray:
        """
        Moore-Penrose pseudo-inverse J_u^+ in R^{1x3}.

        For a column vector J_u,  J_u^+ = J_u^T / ||J_u||^2  with
        sigma^2 = 1 + tan^2(delta)/W^2:
            J_u^+ = (1/sigma^2) [cos(theta), sin(theta), tan(delta)/W]

        Parameters
        ----------
        state : (5,) array_like
        """
        delta  = float(state[4])
        sigma2 = 1.0 + np.tan(delta)**2 / self.W**2
        return self.S(state).T / sigma2

    def J_u_pinv_dot(self, state: np.ndarray,
                     delta_dot: float = 0.0, **kwargs) -> np.ndarray:
        """
        Time derivative d/dt J_u^+ in R^{1x3}.

        Full expression:
            d/dt J_u^+ = -(2t sec2/W^2 * delta_dot)/sigma^4 * [c, s, t/W]
                       + (1/sigma^2) * [-s*omega, c*omega, sec2/W * delta_dot]
        where  c = cos(theta), s = sin(theta), t = tan(delta), sec2 = 1/cos^2(delta),
               omega = x_dot_m * tan(delta)/W  (bicycle constraint),
               sigma^2 = 1 + tan^2(delta)/W^2.

        Parameters
        ----------
        state     : (5,) array_like  [x, y, theta, x_dot_m, delta]
        delta_dot : float  Steering rate (rad/s), i.e., u[1]. Default 0.
        """
        theta  = float(state[2])
        x_dot_m = float(state[3])
        delta  = float(state[4])
        c, s   = np.cos(theta), np.sin(theta)
        t      = np.tan(delta)
        sec2   = 1.0 / np.cos(delta)**2
        omega  = x_dot_m * t / self.W                          # bicycle constraint
        sigma2 = 1.0 + t**2 / self.W**2

        term1 = -(2.0 * t * sec2 / self.W**2 * delta_dot) / sigma2**2 \
                * np.array([[c, s, t / self.W]])
        term2 = (1.0 / sigma2) * np.array([[-s * omega,
                                              c * omega,
                                              sec2 / self.W * delta_dot]])
        return term1 + term2

    # ------------------------------------------------------------------
    # Control-affine state-space model
    # ------------------------------------------------------------------

    def f(self, state: np.ndarray) -> np.ndarray:
        """
        Drift vector field with viscous friction:
            f = [x_dot_m cos(theta), x_dot_m sin(theta), x_dot_m tan(delta)/W,
                 -B_tilde/M_tilde * x_dot_m, 0]^T

        The Coriolis term -C_coeff * delta_dot * x_dot_m / M_tilde is absent
        because delta_dot = tau_delta enters as u[1] through the second column of g().
        """
        state = np.asarray(state, dtype=float)
        theta, x_dot_m, delta = state[2], state[3], state[4]
        M_val = float(self.M_r(state)[0, 0])
        B_val = float(self.B_r(state)[0, 0])
        return np.array([x_dot_m * np.cos(theta),
                         x_dot_m * np.sin(theta),
                         x_dot_m * np.tan(delta) / self.W,
                         -B_val * x_dot_m / M_val,
                         0.0])

    def g(self, state: np.ndarray) -> np.ndarray:
        """
        Input matrix field in R^{5x2} for u = [F, delta_dot]^T:
            g = [[0,          0                        ],
                 [0,          0                        ],
                 [0,          0                        ],
                 [1/M_tilde, -C_coeff * x_dot_m/M_tilde],
                 [0,          1                        ]]
        """
        state   = np.asarray(state, dtype=float)
        x_dot_m = state[3]
        M_val   = float(self.M_r(state)[0, 0])
        C_coeff = float(self.C_r(state)[0, 0])
        return np.array([[0.0,  0.0],
                         [0.0,  0.0],
                         [0.0,  0.0],
                         [1.0 / M_val, -C_coeff * x_dot_m / M_val],
                         [0.0,  1.0]])

    def __str__(self) -> str:
        return "ackermann_dynamics"
