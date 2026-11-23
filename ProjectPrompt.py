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
