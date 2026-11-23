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

# ==============================================================================
# 4. COMPLEX NEWTON-RAPHSON ROOT FINDER (DEFLATION + POLISHING)
# ==============================================================================
def eval_poly_and_deriv(coeffs: np.ndarray, z: complex):
    """
    coeffs = [c_n, c_{n-1}, ..., c_0] for P(z) = sum c_k z^k using Horner's scheme.
    """
    p_val = 0j
    dp_val = 0j
    deg = len(coeffs) - 1
    for idx, c in enumerate(coeffs):
        power = deg - idx
        if power > 0:
            dp_val = dp_val * z + p_val
        p_val = p_val * z + c
    return p_val, dp_val


def deflate_linear(coeffs: np.ndarray, root: complex) -> np.ndarray:
    """Synthetic division of P(z) by (z - root)."""
    out = np.zeros(len(coeffs) - 1, dtype=complex)
    acc = 0j
    for idx in range(len(coeffs) - 1):
        acc = coeffs[idx] + root * acc
        out[idx] = acc
    return out


def newton_single_root(coeffs: np.ndarray, z0: complex, tol: float = 1e-13, max_iter: int = 100):
    z = complex(z0)
    for it in range(1, max_iter + 1):
        pz, dpz = eval_poly_and_deriv(coeffs, z)
        if abs(pz) < tol:
            return z, it, True
        if abs(dpz) < 1e-16:
            z += 1e-4 * (1.0 + 1j)
            continue
        dz = pz / dpz
        z -= dz
        if abs(dz) < tol * max(1.0, abs(z)):
            return z, it, True
    return z, max_iter, False


def find_poly_roots_nr(coeffs: np.ndarray, seeds=None):
    """Finds all roots via deflation, then polishes each on the original polynomial."""
    orig = np.array(coeffs, dtype=complex)
    while len(orig) > 1 and abs(orig[0]) < 1e-16:
        orig = orig[1:]
    deg = len(orig) - 1
    if deg <= 0:
        return np.array([], dtype=complex), []

    if seeds is None:
        # Asymmetric complex seeds off the real axis to encourage complex conjugate pairs
        radius = 1.0 + float(np.max(np.abs(orig[1:])) / abs(orig[0]))
        seeds = [radius * np.exp(1j * (2.0 * np.pi * k / deg + 0.37)) for k in range(deg)]

    active = orig.copy()
    raw_roots = []
    logs = []

    for k in range(deg):
        seed = seeds[k % len(seeds)]
        r_defl, it_d, conv_d = newton_single_root(active, seed)
        # Polish on original un-deflated polynomial
        r_pol, it_p, conv_p = newton_single_root(orig, r_defl)
        raw_roots.append(r_pol)
        logs.append((k + 1, seed, r_pol, it_d, conv_d, it_p, conv_p))
        active = deflate_linear(active, r_defl)

    return np.array(raw_roots, dtype=complex), logs
