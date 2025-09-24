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

# ---------- NYTT: global variabel för låst vinkel (None tills vi låser) ----------
theta_const = None   # None = inte låst än. När y>=20 första gången sätts denna.

def mass_and_mprim(t):
    if t <= 10.0:     # Bränsleförbränning aktiv
        return 8.0 - 0.4*t, -0.4
    else:             # Efter 10 sek är bränslet slut
        return 4.0, 0.0

def theta_to_target(x,y):
    global theta_const
    if theta_const is None:
        # Om vi ännu inte passerat 20 m: håll rakt upp
        if y < 20.0:
            return np.pi / 2  # rakt uppåt i början

        # Första gången vi når eller passerar y >= 20: beräkna och lås vinkeln
        dx = x_target - x
        dy = y_target - y

        adjustment = np.radians(9.3225)  # sikta ish 9 grader lägre
        theta_const = np.arctan2(dy, dx) - adjustment

        print(f"Vinkel låst till {np.degrees(theta_const):.2f}° vid position ({x:.2f}, {y:.2f})")
        return theta_const
    else:
        # Om redan låst: återvänd samma konstanta vinkel
        return theta_const

def acceleration(t, state, theta_func):
    x, y, vx, vy = state
    v = np.array([vx, vy])        # Hastighetsvektor

    # massa och masstapp
    m, mprim = mass_and_mprim(t)

    # Hämta riktning för raketens motor (Samma som i din kod)
    theta = theta_func(x,y)
    u = -km * np.array([np.cos(theta), np.sin(theta)])  # avgashastighetsvektor

    # Newton 2 (samme uttryck som du använde)
    a = g - (c * np.linalg.norm(v) * v) / m + (mprim / m) * u

    return a

def rocket_ode(t, state):
    x, y, vx, vy = state
    if y < 0:
        return np.array([0,0,0,0])

    ax, ay = acceleration(t, state, theta_to_target)

    return [vx, vy, ax, ay]

# ----------------- simulering (samma mönster som du använde) -----------------
h = 0.1
# OBS: t_span är tidsintervallet (sekunder). Om du bara kör (0,20) kanske
# simuleringen slutar innan raketen hinner nå målet. Öka t_span om du vill
# följa banan längre än 20 s.
t0 = 0
t1 = 50   # rekommenderar 50 s för att se hela banan; ändra vid behov
state0 = [0.0, 0.0, 0.0, 0.0]
tt = np.arange(t0, t1 + h, h)

sol = solve_ivp(rocket_ode, (t0, t1), state0, t_eval=tt)

t = sol.t
x = sol.y[0]
y = sol.y[1]

plt.plot(x, y, label="Raketens bana")
plt.scatter(x_target, y_target, color="red", marker="x", s=100, label="Mål (80,60)")
plt.xlabel("x-position [m]")
plt.ylabel("y-position [m]")
plt.title("Raketstyrning — konstant vinkel låst vid y>=20")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# Pythagoras från sista positionen för att hitta vektorn till målet
x_final, y_final = x[-1], y[-1]
# ... (allt som du redan har ovanför är samma)

# Pythagoras från sista positionen för att hitta vektorn till målet
x_final, y_final = x[-1], y[-1]

# Alla avstånd längs banan
distances = np.sqrt((x - x_target)**2 + (y - y_target)**2)
closest_dist = np.min(distances)

print(f"Slutposition: ({x_final:.2f}, {y_final:.2f})")
print(f"Närmsta avstånd till målet: {closest_dist:.2f} m")
