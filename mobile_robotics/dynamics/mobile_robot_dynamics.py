from abc import ABC, abstractmethod
import numpy as np


class MobileRobotDynamics(ABC):
    """
    Abstract base class for Euler-Lagrange dynamic models of wheeled mobile robots (WMRs).

    Implements the unified eight-stage procedure described in:
        Pantoja-Garcia et al. (2026). A Unified Euler-Lagrange Framework for the
        Dynamic Modeling of Wheeled Mobile Robots: Holonomic and Non-Holonomic
        Architectures. Mathematics.

    Reduced dynamic model with viscous friction:
        M_r(q) eta_dot + (C_r(q,eta) + B_r(q)) eta = E_r(q) u

    Rewritten as a control-affine state-space system:
        q_tilde_dot = f(q_tilde) + g(q_tilde) u
    where q_tilde = [q^T, eta^T]^T.

    All quantities except J_u_pinv and J_u_pinv_dot are already provided by the
    dynamic matrices (M_r, C_r, B_r, E_r, S).

    Subclasses must implement
    -------------------------
    state_dim            : total state dimension n + k
    input_dim            : number of actuator inputs n_w
    S(state)             : null-space basis S(q) in R^{n x k} (the same as J_u)
    M_r(state)           : reduced inertia  S^T M S  in R^{k x k}
    C_r(state)           : reduced Coriolis S^T M S_dot  in R^{k x k}
    B_r(state)           : reduced damping  S^T B S  in R^{k x k}
    E_r(state)           : reduced input    S^T E(q)  in R^{k x n_w}
    J_u_pinv(state)      : pseudo-inverse J_u^+ in R^{k x n}
    J_u_pinv_dot(state)  : time derivative d/dt J_u^+ in R^{k x n}
    f(state)             : drift field (n+k,)
    g(state)             : input matrix (n+k, n_w)
    """

    def __init__(self, m: float, Iz: float, bv: float = 0.0, bw: float = 0.0):
        """
        Parameters
        ----------
        m  : float  Total chassis mass (kg).
        Iz : float  Total axial moment of inertia (kg.m^2).
        bv : float  Translational viscous damping coefficient (N.s/m).
        bw : float  Rotational viscous damping coefficient (N.m.s/rad).
        """
        if m <= 0:
            raise ValueError("Mass m must be positive.")
        if Iz <= 0:
            raise ValueError("Moment of inertia Iz must be positive.")
        if bv < 0:
            raise ValueError("Translational damping bv must be non-negative.")
        if bw < 0:
            raise ValueError("Rotational damping bw must be non-negative.")

        self.m  = float(m)
        self.Iz = float(Iz)
        self.bv = float(bv)
        self.bw = float(bw)

        self.M        = np.diag([self.m, self.m, self.Iz])
        self.B_global = np.diag([self.bv, self.bv, self.bw])

    # ------------------------------------------------------------------
    # Abstract interface -- dynamic matrices
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def S(self, state: np.ndarray) -> np.ndarray:
        """Null-space basis S(q) in R^{n x k} (the same as J_u). Satisfies A(q) S(q) = 0."""
        raise NotImplementedError

    @abstractmethod
    def M_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced inertia M_tilde = S^T M S in R^{k x k}."""
        raise NotImplementedError

    @abstractmethod
    def C_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced Coriolis C_tilde = S^T M S_dot in R^{k x k}."""
        raise NotImplementedError

    @abstractmethod
    def B_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced damping B_tilde = S^T B S in R^{k x k}."""
        raise NotImplementedError

    @abstractmethod
    def E_r(self, state: np.ndarray) -> np.ndarray:
        """Reduced input E_tilde = S^T E(q) in R^{k x n_w}."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Abstract interface -- pseudo-inverse of J_u
    # ------------------------------------------------------------------

    @abstractmethod
    def J_u_pinv(self, state: np.ndarray) -> np.ndarray:
        """
        Moore-Penrose pseudo-inverse J_u^+ in R^{k x n}.

        Used to compute the reference reduced velocity and to project the
        pose error into the admissible subspace:
            eta_ref = J_u^+(state) @ q_dot_ref

        Parameters
        ----------
        state : (n+k,) array_like

        Returns
        -------
        J_u_pinv : (k, n) ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def J_u_pinv_dot(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Time derivative d/dt J_u^+ in R^{k x n}.

        Used to compute the reference reduced acceleration:
            eta_dot_ref = J_u_pinv(state) @ q_ddot_ref
                        + J_u_pinv_dot(state, ...) @ q_dot_ref

        Architectures whose J_u^+ depends on a control input (Ackermann: delta_dot)
        accept it via keyword argument documented in the subclass.

        Parameters
        ----------
        state   : (n+k,) array_like
        **kwargs: architecture-specific (see subclass)

        Returns
        -------
        J_u_pinv_dot : (k, n) ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def f(self, state: np.ndarray) -> np.ndarray:
        """Drift vector field f(q_tilde) in R^{n+k}."""
        raise NotImplementedError

    @abstractmethod
    def g(self, state: np.ndarray) -> np.ndarray:
        """Input matrix g(q_tilde) in R^{(n+k) x n_w}."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared simulation step
    # ------------------------------------------------------------------

    def step(self, state: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance the state one step via explicit Euler integration.

        Parameters
        ----------
        state : (n+k,) array_like
        u     : (n_w,) array_like
        dt    : float  Integration step (s).

        Returns
        -------
        state_next : (n+k,) ndarray
        """
        if dt <= 0:
            raise ValueError("Time step dt must be positive.")
        state = np.asarray(state, dtype=float).ravel()
        u     = np.asarray(u,     dtype=float).ravel()
        if state.size != self.state_dim:
            raise ValueError(f"State size mismatch: expected {self.state_dim}, got {state.size}.")
        if u.size != self.input_dim:
            raise ValueError(f"Input size mismatch: expected {self.input_dim}, got {u.size}.")
        return state + (self.f(state) + self.g(state) @ u) * dt

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
