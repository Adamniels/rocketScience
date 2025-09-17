c = 0.05 # mass
Km = 700 # m/s
m0 = 8 # startmassa
g = 9.81 # gravity

def Fvec(t):
    f1 = m(t)
    f2 = m(t)
    return 1

def m(t):
    if (t <= 10):
        return 8.0 - 0.4*t
    elif (t > 10):
        return 4

def mprim(t):
    if (t <= 10):
        return - 0.4
    elif (t > 10):
        return 0
