import matplotlib.pyplot as plt
from casadi import *
import biorbd
from BiorbdViz import BiorbdViz
import scipy.interpolate
import scipy.integrate
import numpy as np

T = 3.2
N = 1000
dN = T/N

model = biorbd.Model("/home/lim/Documents/code/Models/V3/arm26.bioMod")
#kni

# Creation des fonctions integrals

NbQ = model.nbQ()
NbQd = model.nbQdot()
NbTor = model.nbGeneralizedTorque()

def func_ode(t, x, tau1, tau2):
    q = x[:NbQ]
    v = x[NbQ:]
    a = model.ForwardDynamics(q, v, np.vstack([tau1, tau2]).squeeze()).to_array()
    return np.hstack([v, a])

# tau0 = np.zeros((2,1))
# Xk0 = np.zeros((4,1)).squeeze()
# sol = scipy.integrate.solve_ivp(func_ode, [0, dN], Xk0, method='RK45', args = (tau0[0],tau0[1]))


def INT(Xk, tau):
    # V0 = Xk[1]
    # X0 = Xk[0]
    # Total = Xk[0] + Xk[1] + tau
    Total = Xk[0] + Xk[1]

    t = np.linspace(0, dN, 100)

    #sol_Xk = scipy.integrate.solve_ivp(func_ode, [0, dN], Total, method='RK45')
    sol_Xk = scipy.integrate.solve_ivp(func_ode, [0, dN], Xk, method='RK45', args=(tau[0], tau[1]))
    # sol_interp_Xk = scipy.interpolate.interp1d(sol_Xk.t, sol_Xk.y, kind='cubic')

    # print(sol_interp_V(t))
    # plt.plot(t, sol_interp_V(t).squeeze())
    # plt.show(block=True)

    return sol_Xk.y


# Definition du temps et de la position au articulation

tps = np.linspace(0, T, N)
Q1 = np.cos(tps)
Q2 = Q1/2.
Q = np.vstack([Q1, Q2])
V = np.gradient(Q, dN, edge_order = 1, axis=1)


# Definition des couples articulaires

Tart_opt = np.load('Couple_Opt_V3.npy')
T1 = Tart_opt[4::6]
T2 = Tart_opt[5::6]
Vbis = np.zeros((2, 1)).squeeze()
Qbis = np.array([1, 0.5])
Xk = []
Nbis = N
while Nbis != 0:
    Xk = INT(np.append(Qbis[N - Nbis:], Vbis[N - Nbis:]), [T1[N - Nbis], T2[N - Nbis]])
    Qbis = np.vstack([Qbis, Xk[:NbQ, len(Xk[NbQd-1])-1]])
    Vbis = np.vstack([Vbis, Xk[NbQ:, len(Xk[NbQd-1])-1]])
    Nbis -= 1

Vbis_Test = np.gradient(Qbis, dN, edge_order = 1, axis=0)          # Par curiosité


# Definition de la position des markers en array. argument : Pas de calcul, Numero Marker, Coordonne xyz

markers = [[model.markers(np.array(Qbis[k]))[j].to_array() for j in range(len(model.markers(np.array(Qbis[k]))))] for k in range(len(Q1))]

np.save('DataMarkeur-Couple.npy', markers)

NewMarkers = np.load('DataMarkeur-Couple.npy')
print(NewMarkers)
# print(Q)
print(markers)

V1 = np.gradient(Q1, tps, edge_order = 1)                      # Approximation plus vrai, edge_order peut être egal a 2, change le premier et dernier terme
Acc = np.gradient(V1, tps, edge_order = 1)

# Affichage creation plt

plt.plot(tps, Q.T, label = 'Position')
plt.plot(tps, V.T, label = 'Vitesse')
plt.title('Position / Vitesse Initial - Cos')
plt.legend(loc='best')
plt.show(block=True)
plt.figure()
# plt.title('Position / Vitesse Initial')
# plt.legend(loc='best')
# plt.ion()
# plt.plot(tps, Q1, label='Q_1')
# plt.plot(tps, Q2, label='Q_2')
# plt.title('Position Initial')
# plt.legend(loc='best')
# plt.figure()
# plt.plot(tps, V1, label = 'Qdot_1bis')
# plt.title('Vitesse Initial')
# plt.legend(loc='best')
# plt.show(block=True)
plt.plot(tps, Qbis[:-1, 0], label = 'Position 0 tan')
plt.plot(tps, Qbis[:-1, 1], label = 'Position 1 tan')
plt.plot(tps, Vbis[:-1, 0], label = 'Vitesse 0 tan')
plt.plot(tps, Vbis[:-1, 1], label = 'Vitesse 1 tan')
# plt.plot(tps, Vbis_Test[:-1, 0], label = 'Vitesse Test 0 tan')
# plt.plot(tps, Vbis_Test[:-1, 1], label = 'Vitesse Test 1 tan')
plt.title('Position Initial')
plt.legend(loc='best')
plt.show(block=True)
plt.figure()


# Affichage creation Model


qs = np.array([Qbis[:-1, 0], Qbis[:-1, 1]])                                           # argument = les different angle de ton model par DDL
np.save("visual", qs.T)                                                               # qs.T pour transposer de qs
b = BiorbdViz(model_path="/home/lim/Documents/code/Models/V3/arm26.bioMod")           # Va chercher ton modele
b.load_movement(qs.T)
b.exec()

# qs = np.array([Q1, Q2])                                                               # argument = les different angle de ton model par DDL
# np.save("visual", qs.T)                                                               # qs.T pour transposer de qs
# b = BiorbdViz(model_path="/home/lim/Documents/code/Models/V3/arm26.bioMod")           # Va chercher ton modele
# b.load_movement(qs.T)
# b.exec()