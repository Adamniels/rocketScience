import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
import math
from enum import Enum, auto

'''

Implementerar två lösningar: 

Pythagoras: 
    Kollar vinkeln mot målet med pythagoras sats, och riktar oss dit,  eftersom att krafter ackumuleras
    och vi inte tänker på de yttre påverkande krafter, kommer vi alltid att missa
    
"Spök-mål":
    Vi räknar ut effekten på de yttre påverkande krafterna med formeln: s = vt + (1/2)a t^2
    där v är hastighet, och a är accelration, en känd formel som bygger på att man integrerar accelrationen, 
    och får ut denna formel. Detta innebär att vi siktar bakom målet, när vi kommer nära hämnar spök målet bakom oss, 
    eftersom att vi kommer efter t sekunder befinna oss på fel sida, vilket även gör att vi bromsar in innan träff
    vilket är en snygg konsekvens av formeln. 
'''



# =======================
#   Konstanter & mål
# =======================
g = 9.81
c = 0.05
km = 700.0            # avgashastighet [m/s]
m0 = 8.0
fuel_rate = 0.4       # kg/s
burn_time = (m0/2)/fuel_rate   # 10 s

TARGET = np.array([360.0, 80.0])   # mål (x*, y*)
H_LOCK = 20.0                     # rakt upp tills y >= H_LOCK
HIT_RADIUS = 2.0                  # träffradie [m]
TAU = 0.8                         # lead-tid [s] för avancerad styrning

# =======================
#   Hjälpfunktioner fysik
# =======================
def mass(t: float) -> float:
    return m0 - fuel_rate*t if t <= burn_time else m0 - fuel_rate*burn_time

# mass derivatan
def mdot(t: float) -> float:
    return -fuel_rate if t <= burn_time else 0.0

# External forces
def F_ext(vx: float, vy: float, m: float) -> np.ndarray:
    v = np.hypot(vx, vy)
    drag = -c * v * np.array([vx, vy])
    gravity = np.array([0.0, -m*g])
    return gravity + drag

def unit_from_to(x: float, y: float, gx: float, gy: float) -> tuple[float, float]:
    dx, dy = gx - x, gy - y         # riktningsvektor
    n = math.hypot(dx, dy)          # längd
    return (dx/n, dy/n) if n > 1e-12 else (0.0, 0.0)  # enhetsvektor eller 0

# =======================
#   Styrning (enum)
# =======================
class SteeringMode(Enum):
    SIMPLE = auto()
    LEAD = auto()
    CONST275 = auto()

# Pythagoras sats, för att sikta exakt på målet i varje stund
def theta_simple_solution(t: float, state: np.ndarray) -> float:
    x, y, vx, vy = state
    if y < H_LOCK:
        return -np.pi/2
    dx, dy = TARGET[0] - x, TARGET[1] - y
    return np.arctan2(dy, dx) - np.pi


# Se uträkningar, men bygger på att vi integrerar external accelration i x- & y-led
# då får vi meter, som vi flyttar målet med, vi siktar nu mot denna punkt.

def theta_lead(t: float, state: np.ndarray) -> float:
    x, y, vx, vy = state
    if y < H_LOCK:
        return -np.pi/2
    m = mass(t)
    ax_ext, ay_ext = F_ext(vx, vy, m) / m
    gx = TARGET[0] - (vx*TAU + 0.5*ax_ext*TAU*TAU)
    gy = TARGET[1] - (vy*TAU + 0.5*ay_ext*TAU*TAU)
    tx, ty = unit_from_to(x, y, gx, gy)   # önskad THRUST-riktning
    return math.atan2(-ty, -tx)

# Hämtar rätt funktion, baserat på enum
def make_theta(mode: SteeringMode):
    return {
        SteeringMode.SIMPLE:   theta_simple_solution,
        SteeringMode.LEAD:     theta_lead
    }[mode]

# ===========================
#   ODE-högerled
# ==========================
def make_rhs(theta_func):
    def rhs(t: float, s: np.ndarray) -> np.ndarray:
        x, y, vx, vy = s
        m = mass(t)
        Fx, Fy = F_ext(vx, vy, m) #De externa krafter som påverkar (Ej thrust)

        md = mdot(t) # Mass-derivatan, är den 0, så är bränslet slut
        if md != 0.0:
            # Funktionen som väljer riktning, (Pythagoras, eller mer avancerade)
            th = theta_func(t, s)
            # Thrust frammåt, vi vet att bränsle finns kvar
            u = km * np.array([math.cos(th), math.sin(th)])
            # Forward thrust
            Ft = md * u
        else:
            Ft = np.zeros(2)

        # Deriverade från formlerna, se uträkningar
        ax, ay = (Fx + Ft[0]) / m, (Fy + Ft[1]) / m
        return np.array([vx, vy, ax, ay])
    return rhs

# =======================
#   Event (stopp)
# =======================

# För att veta när vi träffar mark, kan hämtas ut via sol_t_events
def hit_ground(t, s):
    return s[1]
hit_ground.terminal = True
hit_ground.direction = -1

#Event för att lösaren ska veta om vi träffat, kan hämtas ut via sol.t_events efteråt
def hit_target_radius_factory(r: float):
    def _ev(t, s):
        return math.hypot(s[0]-TARGET[0], s[1]-TARGET[1]) - r
    _ev.terminal = True
    _ev.direction = -1
    return _ev

# =======================
#   Simulering   (Tar in antingen simple pythagoras mode, eller den mer avancerade framåt gissande lösningen)
# =======================
def simulate(mode: SteeringMode):
    theta_func = make_theta(mode)
    u0 = np.array([0.0, 0.0, 0.0, 0.0])


    # Kör under 120 sekunder, med max steget 0.05 (väldigt små steg)
    tspan = (0.0, 120.0)
    sol = solve_ivp(
        make_rhs(theta_func), tspan, u0,
        method="RK45", dense_output=True,
        events=[hit_ground, hit_target_radius_factory(HIT_RADIUS)],
        max_step=0.05
    )


    # Under är bara kod för uppritning, inget som behövs varken för fysiken, eller simuleringen

    # ========================== Uppritning =========================================================


    # I solven skickade vi med events att hålla koll på (mark_träff och mål_träff)
    # Vi lagrar nu dessa
    t_end_candidates = []
    if sol.t_events[0].size: t_end_candidates.append(sol.t_events[0][0])  # mark
    if sol.t_events[1].size: t_end_candidates.append(sol.t_events[1][0])  # träff
    # det event som faktiskt stoppade simuleringen, markträff eller en riktig träff
    t_end = min(t_end_candidates) if t_end_candidates else sol.t[-1]

    # skapar en finfördelad lösning, där man kan accessa vilken tidpunkt som helst,
    # och inte enbart diskreta tidställen
    tt = np.linspace(sol.t[0], t_end, 400)
    X, Y, *_ = sol.sol(tt)


    # ============= För utskrift i konsolen =================

    # bränsleslut position (För att kunna printa i konsolen)
    t_b = min(burn_time, t_end)
    xb, yb, *_ = sol.sol(t_b)

    # slutpunkt (För att kunna printa i konsolen )
    xf, yf, *_ = sol.sol(t_end)
    d_end = float(np.hypot(xf - TARGET[0], yf - TARGET[1]))
    hit = (sol.t_events[1].size and abs(sol.t_events[1][0] - t_end) < 1e-9)

    fig, ax = plt.subplots(figsize=(7,5))

    # Dictionary, där mode ger oss en label för plotten, helt onödigt men lite snyggt
    label = {
        SteeringMode.SIMPLE: "Styrning: SIMPLE",
        SteeringMode.LEAD:   f"Styrning: LEAD (TAU={TAU})"
    }[mode]


    ax.plot(X, Y, 'C0', label=label)
    ax.plot(*TARGET, 'r*', ms=12, label='Mål')
    circ = Circle(TARGET, HIT_RADIUS, fill=False, ls='--', lw=1.0, ec='r', alpha=0.8,
                  label=f'Träffradie {HIT_RADIUS:g} m')
    ax.add_patch(circ)
    ax.plot(xb, yb, 'ks', ms=6, label=f'Bränsle slut ({burn_time:.0f} s)')
    ax.plot(xf, yf, 'go' if hit else 'ko', ms=6, label=('Träffpunkt' if hit else 'Stoppunkt'))

    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title('Raketstyrning')
    ax.grid(True); ax.legend(); ax.axis('equal')
    plt.tight_layout(); plt.show()
    # ==========================================================================

    # Konsol printouts för att veta om det gick bra
    print(f"Mode: {mode.name} | Slut: {'HIT' if hit else 'STOP'} @ t = {t_end:.2f}s")
    print(f"Slutpunkt: (x,y)=({xf:.2f}, {yf:.2f})  dist till mål = {d_end:.3f} m")
    print(f"Bränsleslut vid t = {burn_time:.2f}s på (x,y) = ({xb:.2f}, {yb:.2f})")

if __name__ == "__main__":
    simulate(SteeringMode.LEAD)
    # simulate(SteeringMode.SIMPLE)







