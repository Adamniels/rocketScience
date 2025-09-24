import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Tyngdacceleration (m/s^2) nedåt
g = np.array([0.0, -9.81])

# Luftmotståndskoefficient
c = 0.05

# Avgashastighet
km = 700.0

# Målet
x_target, y_target = 80.0, 60.0


def mass_and_mprim(t):
    if t <= 10.0:     # Bränsleförbränning aktiv
        return 8.0 - 0.4*t, -0.4
    else:             # Efter 10 sek är bränslet slut
        return 4.0, 0.0


def theta_to_target(x,y):
    if y < 20.0:
        return np.pi / 2  # rakt uppåt i början

    # Vektor från raket till mål (global)
    dx = x_target - x
    dy = y_target - y

    return np.arctan2(dy, dx)

def acceleration(t, state, theta_func):
    x, y, vx, vy = state
    v = np.array([vx, vy])        # Hastighetsvektor

    # massa och masstapp
    m, mprim = mass_and_mprim(t)

    # Hämta riktning för raketens motor
    theta = theta_func(x,y)
    u = -km * np.array([np.cos(theta), np.sin(theta)])  # avgashastighetsvektor

    # Newton 2
    a = g - (c * np.linalg.norm(v) * v) / m + (mprim / m) * u

    return a


def rocket_ode(t, state):
    x, y, vx, vy = state
    if y < 0:
        return np.array([0,0,0,0])

    ax, ay = acceleration(t, state, theta_to_target)

    return [vx, vy, ax, ay]


h = 0.1
y0 = 0
y1 = 20

state0 = [0.0, 0.0, 0.0, 0.0]
t_span = (y0, y1)   
tt = np.arange(y0,y1,h)

sol = solve_ivp(rocket_ode, t_span, state0, t_eval=tt)

t = sol.t
x = sol.y[0]
y = sol.y[1]


plt.plot(x, y, label="Raketens bana")
plt.scatter(x_target, y_target, color="red", marker="x", s=100, label="Mål (80,60)")
plt.xlabel("x-position [m]")
plt.ylabel("y-position [m]")
plt.title("Raketstyrning")
plt.legend()
plt.grid(True)
plt.show()

# Pythagoras från sista positionen för att hitta vektorn till målet
x_final, y_final = x[-1], y[-1]
dist = np.sqrt((x_final - x_target)**2 + (y_final - y_target)**2)
print(f"Slutposition: ({x_final:.2f}, {y_final:.2f})")
print(f"Avstånd till målet: {dist:.2f} m")
