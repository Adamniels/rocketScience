import numpy as np
import matplotlib.pyplot as plt
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
def u(t, state, theta_func, theta):
    '''I den här lösningen så skickar vi även in theta som är 
    den vinkeln som vi testar för tillfället så att vår theta 
    funktion kan skicka tillbaka den'''

    theta_thrust = theta_func(state, theta)
    return KM * np.array([np.cos(theta_thrust), np.sin(theta_thrust)])

# ODE
def ode_rhs(t, state, theta):
    # state = [x, y, vx, vy] 
    _, _, vx, vy = state
    v = np.array([vx, vy])
    thrust = u(t, state, theta_func, theta)
    acc = (Fvec(t, v) + mprim(t) * thrust) / m(t)       
    return np.array([vx, vy, acc[0], acc[1]])

def theta_func(state, theta):
    '''Theta funktion som returnera tillbaka en konstant så länge vi inte är under 20 i y-led'''
    _, y, _, _ = state
    if y < TURN_AFTER_20: 
        return -np.pi/2  
    return theta + np.pi

# Helper för att hitta closest distance i en körning av ett specielt theta
def simulate_one_theta(theta, ode, tspan, y0, teval):
    solution = solve_ivp(ode, tspan, y0, t_eval=teval, args=(theta,))
    
    trajectory = solution.y[:2].T # tar 2 första arrayerna som innehålller x och y och transposar dem så att vi får fram punkter                       
    distances = np.linalg.norm(trajectory - TARGET, axis=1) # avstånd med längd på vektorn, med punkerna från tidigare steg, axis = 1 för att vi vill beräkna längden för varje rad


    min_index = np.argmin(distances) # hämtar index på minsta distans
    min_distance = distances[min_index]
    return min_distance 


# Start värden
y0 = np.array([0.0, 0.0, 0.0, 0.0])   # [x, y, vx, vy]
t_span = (0, 40)                       
t_eval = np.arange(t_span[0], t_span[1], 0.1)  


# Löser med en variant av binärsökning
low, high = -np.pi/2, np.pi/2

best_theta = 0.0

# Använd simulate_one_theta för initialt bästa
best_distance = simulate_one_theta(best_theta, ode_rhs, t_span, y0, t_eval)

for _ in range(20):  # 20 iterationer räcker
    mid = (0.5 * (low + high))
    step = 0.1 * (high - low)               
    right = (mid + step) 

    min_distance_mid = simulate_one_theta(mid, ode_rhs, t_span, y0, t_eval)
    min_distance_right = simulate_one_theta(right, ode_rhs, t_span, y0, t_eval)

    if min_distance_right < min_distance_mid:
        # bättre åt höger -> flytta intervallet höger
        low = mid
        cur_theta, cur_distance = right, min_distance_right
    else:
        # bättre på mitten -> flytta intervallet vänster (mot mid)
        high = right
        cur_theta, cur_distance = mid, min_distance_mid

    if cur_distance < best_distance:
        best_distance = cur_distance
        best_theta = cur_theta

    if (high - low) < 1e-4:
        break

# Visa Resultat
theta = best_theta
best_solution = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval, args=(theta,))

trajectory = best_solution.y[:2].T
distances = np.linalg.norm(trajectory - TARGET, axis=1)
min_index = int(np.argmin(distances))
min_cordinate = trajectory[min_index]

print(f"Bästa vinkel θ = {theta:.6f} rad = {np.degrees(theta):.3f}°")
print(f"Minsta avstånd ≈ {best_distance:.3f} m vid t = {best_solution.t[min_index]:.2f} s, punkt {min_cordinate}")

# Plotta
plt.figure()
plt.plot(best_solution.y[0], best_solution.y[1], label=f"θ={np.degrees(theta):.2f}°")
plt.plot(TARGET[0], TARGET[1], "ro", label="Mål")
plt.plot(min_cordinate[0], min_cordinate[1], "kx", ms=10, label="Närmast")
plt.axis("equal"); plt.grid(True); plt.legend()
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.ylim(bottom=0)   
plt.show()
