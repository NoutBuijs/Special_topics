import numpy as np
from scipy import optimize as opt
from scipy import integrate as int
from matplotlib.colors import hsv_to_rgb


assignment = r"4771958 L1 0.03 2008 JL3 2021 QM1"

trajectory = np.genfromtxt(r"data/emp_ast/2021_QM1_xyPlane.dat")[:-1]

Ms = 1.9891E30
mus = 1.32712440042E20
Me = 6.0477E24
au = 149597870.7E3
P = 4.55E-6
omega = 2*np.pi/(365*24*3600)

# Ms = 5.9722E24
# mus = 3.986004418E14
# Me = 0.07346E24
# au = 384400E3

mu = Me/(Me+Ms)

Sun = np.array([-mu, 0, 0, 0])
Earth = np.array([1-mu, 0, 0, 0])

def colors(n, colorshift=0, s = 1, v = 1):
    ret = []
    start = colorshift
    h = start
    step = (240 - start) / (n - 1)
    for i in range(n):
        A = hsv_to_rgb(np.array([h / 360, s, v]))
        ret.append(A)
        h += step

    return ret

def get_params():
    return Ms, mus, Me, au, P, mu, Sun, Earth, omega

def grad_x(x):

    r1 = mu + x
    r2 = 1 - mu - x

    U_x = x - (1-mu) * r1/np.abs(r1)**3 + mu * r2/np.abs(r2)**3

    return U_x

def der_grad_x(x):

    r1 = mu + x
    r2 = 1 - mu - x

    return 1 + (1-mu) * 2/abs(r1)**3 + mu * 2/abs(r2)**3

def get_lagrange(x_0):
    return opt.newton(grad_x, x_0, fprime=der_grad_x)

def beta(x):

    r = mu+x

    return r**2/(1-mu) * np.abs(grad_x(x))

def Area(b, m):
    return b*mus*m/(P*2*au**2)

def get_jacobian(x):

    r1 = np.sqrt((mu+x[0])**2 + x[1]**2 + x[2]**2)
    r2 = np.sqrt((1-mu-x[0])**2 + x[1]**2 + x[2]**2)

    U_xx = 1 - (1-mu) * (r1**2 - 3 * (mu+x[0])**2)/r1**5 + mu * (3 * (1-mu-x[0])**2 - r2**2)/r2**5
    U_xy = 3 * (1-mu) * x[1] * (mu+x[0])/r1**5 - 3 * mu * (1-mu-x[0]) * x[1]/r2**5
    U_yy = 1 - (1-mu) * (r1**2 - 3 * x[1]**2)/r1**5 - mu * (r2**2 - 3 * x[1]**2)/r2**5

    return np.array([[0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [U_xx, U_xy, 0, 2],
                    [U_xy, U_yy, -2, 0]])

def dxdt(t, x):

    r1 = np.sqrt((mu+x[0])**2 + x[1]**2 + x[2]**2)
    r2 = np.sqrt((1-mu-x[0])**2 + x[1]**2 + x[2]**2)

    xdot = np.ones_like(x)

    xdot[:3] = x[3:]

    xdot[3] = 2 * x[4] + x[0] - (1-mu) * (mu+x[0])/r1**3 + mu * (1-mu-x[0])/r2**3
    xdot[4] = -2 * x[3] + x[1] - (1-mu) * x[1]/r1**3 - mu * x[1]/r2**3
    xdot[5] = -(1-mu) * x[2]/r1**3 - mu * x[2]/r2**3

    return xdot

def dxdt_sol(t, x, angle, beta):

    r1 = np.sqrt((mu + x[0]) ** 2 + x[1] ** 2 + x[2] ** 2)
    r2 = np.sqrt((1 - mu - x[0]) ** 2 + x[1] ** 2 + x[2] ** 2)

    delta = np.arctan2(x[1],(mu+x[0]))
    R = np.array([[np.cos(delta), -np.sin(delta), 0],
                  [np.sin(delta), np.cos(delta), 0],
                  [0, 0, 1]])

    n_hat = np.array([np.cos(angle), np.sin(angle), 0])
    n_hat = R @ n_hat

    xdot = np.zeros(np.size(x))

    xdot[:3] = x[3:]

    xdot[3] = 2 * x[4] + x[0] - (1 - mu) * (mu + x[0]) / r1 ** 3 + mu * (1 - mu - x[0]) / r2 ** 3
    xdot[4] = -2 * x[3] + x[1] - (1 - mu) * x[1] / r1 ** 3 - mu * x[1] / r2 ** 3
    xdot[5] = -(1 - mu) * x[2] / r1 ** 3 - mu * x[2] / r2 ** 3

    xdot[3:] += beta * (1-mu) / r1 ** 2 * np.cos(angle)**2 * n_hat

    return xdot

def get_trajectory(x_init, t, dx = dxdt):
    return int.solve_ivp(dx, (0, t), x_init,
                         rtol = 10**-12, atol = 10**-12)