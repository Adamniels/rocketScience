import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------- Constants -----------
C = 0.05 # kg/m
KM = 700 # m/s
M0 = 8 # startmassa
TARGET = np.array([80, 60])
TURN_AFTER_20 = 20.0
Gvec = np.array([0.00, -9.81]) # gravity


# ---------- External forces -----------
def Fvec(t:float, v: np.ndarray) -> np.ndarray:
    mass = m(t)
    drag = - C * np.linalg.norm(v) * v
    gravity = mass * Gvec
    return gravity + drag

# ---------- Mass functions -----------
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

def u(t: float, state, theta_func ) -> np.ndarray:
    theta = theta_func(t, state)  
    return KM * np.array([np.cos(theta), np.sin(theta)])


# ---------- Bas styrning -----------
def theta_simple_solution(t: float, state: np.ndarray) -> float:
    x, y, vx, vy = state
    if y < TURN_AFTER_20:
        return -np.pi/2  
    dx, dy = TARGET[0] - x, TARGET[1] - y
    return np.arctan2(dy, dx) - np.pi   # rikta gasen mot motsatt håll av målet


# ---------- ODE: -----------
def ode_rhs(t: float, state: np.ndarray) -> np.ndarray:
    # state = [ux, uy, vx, vy] = [x, y, vx, vy]
    x, y, vx, vy = state
    v = np.array([vx, vy], dtype=float)
    thrust = u(t, state, theta_simple_solution)                   
    acc = (Fvec(t, v) + mprim(t) * thrust) / m(t)       
    return np.array([vx, vy, acc[0], acc[1]])


# ----------  initialvillkor ----------
y0 = np.array([0.0, 0.0, 0.0, 0.0])   # [x, y, vx, vy]
t_span = (0, 40)                       # simulera i 40 s
t_eval = np.arange(t_span[0], t_span[1], 0.1)  # tät tidsskala

# ---------- lös ODE:n ----------
sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval)

# ---------- plocka ut banan ---------
x = sol.y[0]
y = sol.y[1]

# ---------- plott ----------
plt.figure()
plt.plot(x, y, label="raketbana")
plt.plot(TARGET[0], TARGET[1], "ro", label="mål")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Raket med 'peka mot mål'-styrning")
plt.show()
