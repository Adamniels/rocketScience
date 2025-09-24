import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants 
C = 0.05 # kg/m
KM = 700 # m/s
M0 = 8 # startmassa
TARGET = np.array([80, 40])
TURN_AFTER_20 = 20.0
Gvec = np.array([0.00, -9.81]) # gravity (only in y direction)


# External forces F 
def Fvec(t:float, v: np.ndarray) -> np.ndarray:
    mass = m(t)
    drag = - C * np.linalg.norm(v) * v
    gravity = mass * Gvec
    return gravity + drag

# Mass functions 
def m(t: float) -> float:
    if (t <= 10):
        return 8.0 - 0.4*t
    elif( t > 10):
        return 4
    else:
        raise ArithmeticError(f"t should be a positive number")

def mprim(t: float) -> float:
    if (t <= 10):
        return -0.4
    elif( t > 10):
        return 0
    else:
        raise ArithmeticError(f"t should be a positive number")

# Thrust
def u(t, state, theta):
    return KM * np.array([np.cos(theta), np.sin(theta)])

# ODE
def ode_rhs(t, state, theta):
    # state = [x, y, vx, vy] 
    x, y, vx, vy = state
    v = np.array([vx, vy], dtype=float)
    thrust = u(t, state, theta)
    acc = (Fvec(t, v) + mprim(t) * thrust) / m(t)       
    return np.array([vx, vy, acc[0], acc[1]])



