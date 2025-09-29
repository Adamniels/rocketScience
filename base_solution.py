import numpy as np
import matplotlib.pyplot as plt
from numpy._core.records import array
from scipy.integrate import solve_ivp

# Konstanter 
C = 0.05 # kg/m
KM = 700 # m/s
M0 = 8 # startmassa
TARGET = np.array([80, 60])
TURN_AFTER_20 = 20.0
Gvec = np.array([0.00, -9.81]) # gravitation (bara i y-led)


# Yttre krafter F vektor med krafter i x- och y-led 
def Fvec(t, v): # v är vektorn för hastigheten
    mass = m(t)
    drag = - C * np.linalg.norm(v) * v #luft motståndet
    gravity = mass * Gvec
    return gravity + drag

# Mass funktioner
def m(t: float) -> float:
    if (t >= 0 and t <= 10):
        return 8.0 - 0.4*t
    elif( t > 10):
        return 4
    else:
        raise ArithmeticError(f"t should be a positive number")

def mprim(t: float) -> float:
    if (t >= 0  and t <= 10):
        return -0.4
    elif( t > 10):
        return 0
    else:
        raise ArithmeticError(f"t should be a positive number")

# Motor kraft (thrust)
def u(t, state, theta_func ):
    theta = theta_func(t, state)  
    return KM * np.array([np.cos(theta), np.sin(theta)])

# ODE
def ode_rhs(t, state, theta_func):
    # state = [x, y, vx, vy] 
    x, y, vx, vy = state

    # För bättre print
    if(y < 0):
        return np.array([0,0,0,0])

    v = np.array([vx, vy], dtype=float)
    thrust = u(t, state, theta_func)
    acc = (Fvec(t, v) + mprim(t) * thrust) / m(t)       
    return np.array([vx, vy, acc[0], acc[1]])

# Bas lösning, vinklar bara mot målet i varje punkt 
def theta_simple_solution(t, state):
    x, y, vx, vy = state
    if y < TURN_AFTER_20:
        return -np.pi/2  
    dx, dy = TARGET[0] - x, TARGET[1] - y

        
    return np.arctan2(dy, dx) - np.pi   # Ändra så riktningen är bort från målet efter som motorn är riktad motsatt riktningen vi åker i

# Egen rungekutta lösare
def RK4(f, tspan, u0, dt, *args):
    t_vec = np.arange(tspan[0],tspan[1]+1.e-14,dt)
    dt_vec = dt*np.ones_like(t_vec)
    if t_vec[-1] < tspan[1]:
        t_vec = np.append(t_vec,tspan[1])
        dt_vec = np.append(dt_vec, t_vec[-1]-t_vec[-2])
    u = np.zeros((len(t_vec),len(u0)))
    u[0,:]= u0
    for i in range(len(t_vec)-1):
        h = dt_vec[i]
        k1 = f(t_vec[i], u[i,:], *args)
        k2 = f(t_vec[i]+0.5*h, u[i,:]+0.5*h*k1, *args)
        k3 = f(t_vec[i]+0.5*h, u[i,:]+0.5*h*k2, *args)
        k4 = f(t_vec[i+1], u[i,:]+h*k3, *args)
        u[i+1,:] = u[i,:] + h*(k1+ 2*k2 + 2*k3 + k4)/6
    return t_vec, u

# Startvärden
y0 = np.array([0.0, 0.0, 0.0, 0.0])   # [x, y, vx, vy]
t_span = (0, 40)                       
t_eval = np.arange(t_span[0], t_span[1], 0.1)  

# Lös
sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, args=(theta_simple_solution,))
sol_rk4_t, sol_rk4_y = RK4(ode_rhs, t_span, y0, 0.1, theta_simple_solution)

# Hitta närmsta distans
trajectory = sol.y[:2].T                       
distances = np.linalg.norm(trajectory - TARGET, axis=1)

index_min = np.argmin(distances)
t_min = sol.t[index_min]
cordinate_min = trajectory[index_min]
distance_min = distances[index_min]
print(f"Baslösning närmast: t={t_min:.3f}, p={cordinate_min}, d={distance_min:.3f} m")


# Plotta 
plt.figure()
plt.plot(sol.y[0], sol.y[1], label="Raket Baslösning")
plt.plot(sol_rk4_y[:,0], sol_rk4_y[:,1], label="Raket Baslösning (egen RK4)")
plt.plot(TARGET[0], TARGET[1], "ro", label="Mål")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Raket med baslösningsstyrning")

# Rita ut närmaste punkt
plt.plot(cordinate_min[0], cordinate_min[1], "kx", ms=10, label=f"Närmaste punkt: t={t_min:.3f}, p={cordinate_min}, d={distance_min:.3f} m")
plt.legend()
plt.show()
