import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import utilities as util
from tqdm import tqdm
from scipy import interpolate
import pandas as pd

sns.set_theme()
sns.set_palette("rocket")

assignment = r"4771958 L1 0.03 2008 JL3 2021 QM1"

Ms, mus, Me, au, P, mu, Sun, Earth, omega = util.get_params()

trajectory = np.genfromtxt(r"data/emp_ast/2021_QM1_xyPlane.dat")[:-1]

tck, u = interpolate.splprep(trajectory[:,1:3].T, u=trajectory[:, 0], s = 0)
tckv, uv = interpolate.splprep(trajectory[:,3:].T, u=trajectory[:, 0], s = 0)
u = np.linspace(trajectory[0,0],trajectory[-1,0],np.size(trajectory[:,0])*5)
trajectory_int = interpolate.splev(u, tck)
v = interpolate.splev(u, tckv)
trajectory_int = np.array(trajectory_int).T
v = np.array(v).T

# fig, ax = plt.subplots(1,1)
# ax.scatter(trajectory[:,1], trajectory[:,2], c = "r", alpha = 0.5, label="Asteroid")
# ax.scatter(trajectory_int[:,0], trajectory_int[:,1])
# ax.scatter(Sun[0], Sun[1], marker = "X", c ="y", label = "Sun")
# ax.scatter(Earth[0], Earth[1], marker = "X", c ="royalblue", label = "Earth")
# ax.legend()

L1 = util.get_lagrange(0.8)
b = util.beta(L1 - 0.011)
A = util.Area(b, 14)

x = np.array([L1, 0, 0, 0, 0, 0])

J = util.get_jacobian(x)
eigenvals, eigenvecs= np.linalg.eig(J)
eigenvals = np.round(eigenvals,12)

eigenvals_unique = np.array([np.max(eigenvals.real), np.min(eigenvals.real)])
eigenvecs_unique = np.array([np.real(eigenvecs.T[np.where(eigenvals == eigenvals_unique[0])])[0],
                              np.real(eigenvecs.T[np.where(eigenvals == eigenvals_unique[1])])[0]])

pert = np.array([10**-5, -10**-5])

x_inits = np.zeros((np.size(eigenvals_unique)*np.size(pert), np.size(x)))

x_inits[:2, :2] = x[:2] + pert[0] * eigenvecs_unique[:,:2]
x_inits[:2, 3:5] = x[3:5] + pert[0] * eigenvecs_unique[:,2:]

x_inits[2:, :2] = x[:2] + pert[1] * eigenvecs_unique[:,:2]
x_inits[2:, 3:5] = x[3:5] + pert[1] * eigenvecs_unique[:,2:]

time = 6 * np.pi
manifolds = np.zeros(4, dtype=object)
eigenvals_unique = np.hstack((eigenvals_unique, eigenvals_unique))
c = np.zeros(4, dtype=object)
for i in range(np.size(manifolds)):

    if i >= 4:
        pert_rand = np.random.normal(0, 1E-6, 6)
        manifolds[i] = util.get_trajectory(x+pert_rand, time)
    elif eigenvals_unique[i] > 0:
        manifolds[i] = util.get_trajectory(x_inits[i], time)
        c[i] = "b"
    elif eigenvals_unique[i] <= 0:
        c[i] = "r"
        manifolds[i] = util.get_trajectory(x_inits[i], -time)

fig, ax = plt.subplots(1,1)
plt.axis("off")
ax.plot(trajectory_int[:,0], trajectory_int[:,1], c = "r")
ax.fill_between(trajectory_int[:,0],trajectory_int[:,1], trajectory_int[:,1]-150000E3/au,
                color = "r", alpha=0.6, label="Asteroid interception margin")

ax.set_facecolor("black")
fig.patch.set_facecolor("black")
stability = [False, True, False, True]
# for i, manifold in enumerate(manifolds):
#     if i < 4:
#         ax.plot(manifold.y[0], manifold.y[1], ls = "--", c = c[i],
#                 label=f"Manifold {i}")
#     if i == 4:
#         ax.plot(manifold.y[0], manifold.y[1], label=f"Perturbed state manifold", ls="--", c="w")
#
#     if i >= 4:
#         ax.plot(manifold.y[0], manifold.y[1], ls="--", c="w")

ax.scatter(Sun[0], Sun[1], marker = ".", s=1000, c ="y", label = "Sun")
ax.scatter(Earth[0], Earth[1], marker = ".", s=200, c ="royalblue", label = "Earth")
ax.scatter(L1, 0, marker = "*", s=100, c ="g", label = "L1 point")
# ax.set_ylim(-0.00785714, 0.0177922)
# ax.set_xlim(0.987688, 1.02584)
ax.set_aspect("equal")

# ------------------ Data arrays minimum distance maximum Dv ----------------------
dangle = np.radians(2.5)
cone_angles = np.arange(-np.pi/2, np.pi/2 + dangle, dangle)
manifolds_sol = np.zeros(np.size(cone_angles))
rmins = np.zeros(np.size(cone_angles))
dvs = np.copy(rmins)
departures = np.copy(rmins)
arrivals = np.copy(rmins)
e_z = np.array([0,0,1])

# ------------------ Data arrays NN training ---------------------------------------
# data_x = np.zeros(np.size(cone_angles), dtype=object)
# data_y = np.zeros(np.size(cone_angles), dtype=object)

fig1, ax1 = plt.subplots(1,1)
# c = util.colors(np.size(cone_angles))
# ------------------ Data generation ------------------------------------------------
for i,manifold_sol in enumerate(tqdm(manifolds_sol)):
    manifold_sol = util.get_trajectory(x_inits[2], time,
                                       dx = lambda t, x:util.dxdt_sol(t, x, angle=cone_angles[i],
                                                                      beta = 0.03))

    r_earth = np.linalg.norm(Earth[:2] - manifold_sol.y[:2,:].T, axis=1)*au/10**3
    if np.size(np.where(r_earth <= 6371+300)) > 0:
        manifold_sol.y , manifold_sol.t = manifold_sol.y[:,:np.where(r_earth <= 6371+300)[0][0]], \
                                          manifold_sol.t[:np.where(r_earth <= 6371+300)[0][0]]
    if i == 0:
        ax.plot(manifold_sol.y[0], manifold_sol.y[1], ls="--", c = "w", label="Solar sail propelled trajectory")
    else:
        ax.plot(manifold_sol.y[0], manifold_sol.y[1], ls="--", c = "w")
    ax1.plot(manifold_sol.t, np.linalg.norm(manifold_sol.y[3:5].T, axis=1)*au*omega/10**3)

    # data_x_cone = np.zeros(np.size(manifold_sol.y[0]), dtype=object)
    # data_y_cone = np.zeros(np.size(manifold_sol.y[0]), dtype=object)

    for j,x in enumerate(manifold_sol.y.T):

        r = np.linalg.norm(trajectory_int - x[:2], axis=1)
        dv = np.linalg.norm(v - x[3:5], axis=1)

        if np.size(np.where(r*au/10**3 <= 150000)) != 0:
            rmin = r[np.where(dv == np.max(dv[np.where(r*au/10**3 <= 150000)]))]
            dv = np.linalg.norm(v - x[3:5], axis=1)[np.where(r == rmin)]
            arrival = u[np.where(r == rmin)]
            departure = arrival - manifold_sol.t[j] * 1/omega / (24*3600)
        else:
            rmin = np.min(r)
            dv = 0
            arrival = None
            departure = None

        # step = 10
        # data_x_cone[j] = np.array([np.ones(np.size(r))*cone_angles[i],
        #                            np.ones(np.size(r))*manifold_sol.t[j],
        #                            u]).T[np.arange(0, np.size(r), step)]
        # data_y_cone[j] = r[np.arange(0, np.size(r), step)]

        if j == 0:
            departures[i] = departure
            arrivals[i] = arrival
            dvs[i] = dv
            rmins[i] = rmin

        else:
            if rmin*au/10**3 <= 150000 and dv > dvs[i]:
                departures[i] = departure
                arrivals[i] = arrival
                dvs[i] = dv
                rmins[i] = rmin

    # data_x[i] = data_x_cone
    # data_y[i] = data_y_cone

rmins = rmins*au/10**3
dvs = dvs*au*omega/10**3
ax.legend(fontsize=16)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.bar(cone_angles, rmins, color = "k", width = dangle)
ax1.plot(cone_angles, np.ones_like(cone_angles)*150000, ls="--", c="r", label="150000 [km] interception requirement")
ax1.set_xlim(np.min(cone_angles), np.max(cone_angles))
ax1.set_xlabel("Cone angles in radians")
ax1.set_ylabel(r"Distance maximum $\Delta$V is achieved [km]")
ax1.legend()
ax2.plot(cone_angles, dvs)
ax2.set_xlim(np.min(cone_angles), np.max(cone_angles))
ax2.set_xlabel("Cone angles in radians")
ax2.set_ylabel("Maximum $\Delta$V [km/s]")
#
# data_x = np.concatenate(np.concatenate(data_x)).reshape((-1,3))
# data_y = np.concatenate(np.concatenate(data_y)).reshape((-1,1))
# data = np.hstack((data_x, data_y))
# data = pd.DataFrame(data)

# data.to_csv(f"data/NN/data_999.csv")