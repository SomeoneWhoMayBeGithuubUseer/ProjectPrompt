import numpy as np


# ==============================================================================
# 1. PARAMETERS & MODEL DYNAMICS
# ==============================================================================
PARAMS = {
    "R": 0.8,          # Armature resistance (Ohm)
    "L": 2.5e-3,       # Armature inductance (H)
    "J": 0.012,        # Rotor inertia (kg*m^2)
    "b": 0.006,        # Viscous friction (N*m*s/rad)
    "K": 0.15,         # Motor torque/back-EMF constant (N*m/A = V*s/rad)
    "V_max": 24.0,     # Supply voltage limit (V)
    "omega_ref": 100.0,# Speed reference (rad/s)
    "T_L0": 0.7,       # Step load torque (N*m)
    "t_L": 1.5,        # Load step time (s)
    "N": 500.0,        # Derivative filter pole (rad/s)
    "t_end": 3.0,      # Simulation horizon (s)
}

def sat(u: float, v_max: float) -> float:
    return float(np.clip(u, -v_max, v_max))


def rhs_nonlinear(t: float, x: np.ndarray, gains: tuple[float, float, float], p: dict = PARAMS) -> np.ndarray:
    """
    State x = [i, omega, x_I, z] with x(0) = 0
    """
    Kp, Ki, Kd = gains
    i, omega, x_I, z = x

    e = p["omega_ref"] - omega
    T_L = p["T_L0"] if t >= p["t_L"] else 0.0

    u = Kp * e + Ki * x_I + Kd * p["N"] * (e - z)
    v_a = sat(u, p["V_max"])

    # Clamping anti-windup: freeze integrator if saturated (u != v_a) and sign(e) == sign(u)
    saturated = not np.isclose(u, v_a, rtol=0.0, atol=1e-12)
    same_sign = (e * u) > 0.0
    dx_I = 0.0 if (saturated and same_sign) else e

    dz = p["N"] * (e - z)
    di = (-p["R"] * i - p["K"] * omega + v_a) / p["L"]
    domega = (p["K"] * i - p["b"] * omega - T_L) / p["J"]

    return np.array([di, domega, dx_I, dz], dtype=float)

# ==============================================================================
# 2. ODE INTEGRATORS: FIXED-STEP RK4 & BACKWARD EULER + NEWTON
# ==============================================================================
def integrate_rk4(gains, h: float = 1e-4, p: dict = PARAMS):
    n_steps = int(np.round(p["t_end"] / h))
    t_grid = np.linspace(0.0, n_steps * h, n_steps + 1)
    X = np.zeros((n_steps + 1, 4), dtype=float)

    for k in range(n_steps):
        tk = t_grid[k]
        xk = X[k]
        k1 = rhs_nonlinear(tk, xk, gains, p)
        k2 = rhs_nonlinear(tk + 0.5 * h, xk + 0.5 * h * k1, gains, p)
        k3 = rhs_nonlinear(tk + 0.5 * h, xk + 0.5 * h * k2, gains, p)
        k4 = rhs_nonlinear(tk + h, xk + h * k3, gains, p)
        X[k + 1] = xk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_grid, X

def numerical_jacobian_G(x_next: np.ndarray, x_prev: np.ndarray, t_next: float, h: float, gains, p: dict):
    """
    Jacobian of implicit residual G(x_next) = x_next - x_prev - h * f(t_next, x_next)
    via central finite differences (note piecewise kinks at saturation/load boundaries).
    """
    m = len(x_next)
    J_mat = np.zeros((m, m), dtype=float)
    eps = 1e-7
    for j in range(m):
        dx = np.zeros(m, dtype=float)
        step = eps * max(1.0, abs(x_next[j]))
        dx[j] = step
        g_plus = (x_next + dx) - x_prev - h * rhs_nonlinear(t_next, x_next + dx, gains, p)
        g_minus = (x_next - dx) - x_prev - h * rhs_nonlinear(t_next, x_next - dx, gains, p)
        J_mat[:, j] = (g_plus - g_minus) / (2.0 * step)
    return J_mat

def integrate_backward_euler(gains, h: float = 1e-3, p: dict = PARAMS, tol: float = 1e-10, max_iter: int = 25):
    n_steps = int(np.round(p["t_end"] / h))
    t_grid = np.linspace(0.0, n_steps * h, n_steps + 1)
    X = np.zeros((n_steps + 1, 4), dtype=float)

    for k in range(n_steps):
        t_next = t_grid[k + 1]
        x_prev = X[k]
        # Initial guess: explicit Euler predictor
        x_curr = x_prev + h * rhs_nonlinear(t_grid[k], x_prev, gains, p)

        for _ in range(max_iter):
            G = x_curr - x_prev - h * rhs_nonlinear(t_next, x_curr, gains, p)
            if np.linalg.norm(G, ord=np.inf) < tol:
                break
            JG = numerical_jacobian_G(x_curr, x_prev, t_next, h, gains, p)
            delta = solve_linear_pp(JG, -G)
            x_curr = x_curr + delta
            if np.linalg.norm(delta, ord=np.inf) < tol:
                break

        X[k + 1] = x_curr

    return t_grid, X

# ==============================================================================
# 3. GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING & CONDITION NUMBER
# ==============================================================================
def solve_linear_pp(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solves Ax = b using Gaussian elimination with partial pivoting (NumPy)."""
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    n = len(b)

    for col in range(n - 1):
        pivot_row = col + int(np.argmax(np.abs(A[col:, col])))
        if np.isclose(A[pivot_row, col], 0.0, atol=1e-15):
            raise ValueError("Matrix is singular or nearly singular.")
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            b[[col, pivot_row]] = b[[pivot_row, col]]

        factors = A[col + 1 :, col] / A[col, col]
        A[col + 1 :, col:] -= factors[:, None] * A[col, col:]
        b[col + 1 :] -= factors * b[col]

    if np.isclose(A[-1, -1], 0.0, atol=1e-15):
        raise ValueError("Matrix is singular or nearly singular.")

    x = np.zeros(n, dtype=float)
    for row in range(n - 1, -1, -1):
        x[row] = (b[row] - np.dot(A[row, row + 1 :], x[row + 1 :])) / A[row, row]
    return x


def cond_estimate_2norm(A: np.ndarray) -> float:
    """2-norm condition number via SVD (useful for comparing row/column scaling)."""
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1])
