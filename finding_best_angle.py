import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants 
C = 0.05 # kg/m
KM = 700 # m/s
M0 = 8 # startmassa
TARGET = np.array([80, 15])
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
def u(t, state, theta, theta_func):
    theta_thrust = theta_func(state, theta)
    return KM * np.array([np.cos(theta_thrust), np.sin(theta_thrust)])

# ODE
def ode_rhs(t, state, theta):
    # state = [x, y, vx, vy] 
    x, y, vx, vy = state
    v = np.array([vx, vy], dtype=float)
    thrust = u(t, state, theta, theta_func)
    acc = (Fvec(t, v) + mprim(t) * thrust) / m(t)       
    return np.array([vx, vy, acc[0], acc[1]])

def theta_func(state, theta):
    x, y, vx, vy = state
    if y < TURN_AFTER_20: # TODO: detta ställer till det efter
        return -np.pi/2  
    return theta + np.pi

# Helper for getting closest distance
def simulate_one_theta(theta):
    sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, args=(theta,))
    
    traj = sol.y[:2].T                       
    dists = np.linalg.norm(traj - TARGET, axis=1)

    i_min = np.argmin(dists)
    t_min = sol.t[i_min]
    p_min = traj[i_min]
    d_min = dists[i_min]
    return i_min, t_min, p_min, d_min

# TODO: kolla mer på
def clamp_theta(th):
    # undvik exakt ±pi/2 för numerisk stabilitet
    eps = 1e-3
    return float(np.clip(th, -np.pi/2 + eps, np.pi/2 - eps))

y0 = np.array([0.0, 0.0, 0.0, 0.0])   # [x, y, vx, vy]
t_span = (0, 40)                       
t_eval = np.arange(t_span[0], t_span[1], 0.1)  


# Solving
low, high = -np.pi/2, np.pi/2

best_theta = 0.0
# använd simulate_one_theta för initialt bästa
i_min, t_min, p_min, best_d = simulate_one_theta(clamp_theta(best_theta))

for _ in range(20):  # 20 iterationer räcker långt
    mid = clamp_theta(0.5 * (low + high))
    right = clamp_theta(0.5 * (mid + high))

    i_m, t_m, p_m, d_m = simulate_one_theta(mid)
    i_r, t_r, p_r, d_r = simulate_one_theta(right)

    if d_r < d_m:
        # bättre åt höger → flytta intervallet höger
        low = mid
        cur_theta, cur_d = right, d_r
    else:
        # bättre på mitten → flytta intervallet vänster (mot mid)
        high = right
        cur_theta, cur_d = mid, d_m

    if cur_d < best_d:
        best_d = cur_d
        best_theta = cur_theta

    if (high - low) < 1e-4:
        break

theta = best_theta

sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, args=(theta,))

traj = sol.y[:2].T
dists = np.linalg.norm(traj - TARGET, axis=1)
best_i = int(np.argmin(dists))
p_min = traj[best_i]

print(f"Bästa vinkel θ = {theta:.6f} rad = {np.degrees(theta):.3f}°")
print(f"Minsta avstånd ≈ {best_d:.3f} m vid t = {sol.t[best_i]:.2f} s, punkt {p_min}")

# ---- Plot ----
import matplotlib.pyplot as plt
plt.figure()
plt.plot(sol.y[0], sol.y[1], label=f"θ={np.degrees(theta):.2f}°")
plt.plot(TARGET[0], TARGET[1], "ro", label="Mål")
plt.plot(p_min[0], p_min[1], "kx", ms=10, label="Närmast")
plt.axis("equal"); plt.grid(True); plt.legend()
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.show()
