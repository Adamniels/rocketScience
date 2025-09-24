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
def u(t: float, state, theta_func ) -> np.ndarray:
    theta = theta_func(t, state)  
    return KM * np.array([np.cos(theta), np.sin(theta)])

# ODE
def ode_rhs(t: float, state: np.ndarray, theta_func) -> np.ndarray:
    # state = [x, y, vx, vy] 
    x, y, vx, vy = state
    v = np.array([vx, vy], dtype=float)
    thrust = u(t, state, theta_func)
    acc = (Fvec(t, v) + mprim(t) * thrust) / m(t)       
    return np.array([vx, vy, acc[0], acc[1]])

# Base solution 
def theta_simple_solution(t: float, state: np.ndarray) -> float:
    x, y, vx, vy = state
    if y < TURN_AFTER_20:
        return -np.pi/2  
    dx, dy = TARGET[0] - x, TARGET[1] - y

    #Debug print
    # angle_to_target = np.arctan2(dy, dx)
    # angle_real_direction = np.arctan2(vy, vx)
    # if (y > 20 and y < 21):
    #     print(f"y: {y}, angle to target: {angle_to_target} and real angle {angle_real_direction}")
        
    return np.arctan2(dy, dx) - np.pi   # Change so direction i from the goal (thrust go other way)

# Better solution
# ta hänsyn till gravitation och hastighet
def shift_angle(a):
    return (a + np.pi) % (2*np.pi)

def theta_better_solution(t: float, state: np.ndarray) -> float:
    x, y, vx, vy = state
    if y < TURN_AFTER_20:
        return -np.pi/2  
    dx, dy = TARGET[0] - x, TARGET[1] - y
    angle_t = np.arctan2(dy, dx)   # target angle
    angle_v = np.arctan2(vy, vx)   # current v angle

    # account current v angle and mirror in in target angle
    if (x > TARGET[0] or vx < 0):
        thrust_angle = shift_angle(angle_t + np.pi)
        # theta = shift_angle(2*thrust_angle - angle_v)
        theta = angle_t - np.pi
    
    else:
        acc_angle = shift_angle(2*angle_t - angle_v)
        theta = shift_angle(acc_angle + np.pi)

    return theta


# Initials
y0 = np.array([0.0, 0.0, 0.0, 0.0])   # [x, y, vx, vy]
t_span = (0, 40)                       
t_eval = np.arange(t_span[0], t_span[1], 0.1)  


# Solving
sol_simple = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, args=(theta_simple_solution,))

sol_better = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, args=(theta_better_solution,))

# Find closest distance 
traj = sol_simple.y[:2].T                       
dists = np.linalg.norm(traj - TARGET, axis=1)

i_min = np.argmin(dists)
t_min = sol_simple.t[i_min]
p_min = traj[i_min]
d_min = dists[i_min]
print(f"Simple närmast: t={t_min:.3f}, p={p_min}, d={d_min:.3f} m")

traj_better = sol_better.y[:2].T                       
dists_better = np.linalg.norm(traj_better - TARGET, axis=1)

i_min_better = np.argmin(dists_better)
t_min_better = sol_better.t[i_min_better]
p_min_better = traj_better[i_min_better]
d_min_better = dists_better[i_min_better]
print(f"Better närmast: t={t_min_better:.3f}, p={p_min_better}, d={d_min_better:.3f} m")

# Draw graph 
plt.figure()
plt.plot(sol_simple.y[0], sol_simple.y[1], label="Rocket simple solution")
plt.plot(sol_better.y[0], sol_better.y[1], label="Rocket simple solution")
plt.plot(TARGET[0], TARGET[1], "ro", label="Goal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Rocket with basix solution stearing")
# Rita ut närmaste punkt
plt.plot(p_min[0], p_min[1], "kx", ms=10, label="Närmaste (Simple)")
plt.plot(p_min_better[0], p_min_better[1], "kx", ms=10, label="Närmaste (Simple)")
plt.legend()
plt.ylim(bottom=0)   # visa aldrig under y=0
plt.show()
