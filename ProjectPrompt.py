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
