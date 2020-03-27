import matplotlib.pyplot as plt
from casadi import *
import biorbd
from BiorbdViz import BiorbdViz
import scipy.interpolate
import scipy.integrate
import numpy as np
import conf as conf

T = conf.T
N = conf.N
CM = conf.CM
dN = T/N

model = biorbd.Model("/home/lim/Documents/code/Models/V6/arm26.bioMod")
#coucou

# Creation of integrals functions


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
    t = np.linspace(0, dN, 100)

    sol_Xk = scipy.integrate.solve_ivp(func_ode, [0, dN], Xk, method='RK45', args=(tau[0], tau[1]))

    # not developed yet
    # sol_interp_Xk = scipy.interpolate.interp1d(sol_Xk.t, sol_Xk.y, kind='cubic')
    # print(sol_interp_V(t))
    # plt.plot(t, sol_interp_V(t).squeeze())
    # plt.show(block=True)

    return sol_Xk.y


# Definition of time and angles of articulations


tps = np.linspace(0, T, N)

# Creation base on the angle
if CM == 0:
    Q1 = np.cos(tps)
    # Q1 = conf.fct_CM(tps)
    Q2 = Q1/2.
    Q = np.vstack([Q1, Q2])
    V = np.gradient(Q, dN, edge_order = 1, axis=1)

# Creation base on the Torque
elif CM == 1:
    T1 = 10*np.tanh(tps)
    # T1 = 10 * conf.fct_CM(tps)
    T2 = 5*np.tanh(tps)
    # T2 = 5 * conf.fct_CM(tps)
    V = np.zeros((2, 1)).squeeze()
    Q = np.array([1, 0.5])
    Xk = []
    Nbis = N
    while Nbis != 0:
        Xk = INT(np.append(Q[N - Nbis:], V[N - Nbis:]), [T1[N - Nbis], T2[N - Nbis]])
        Q = np.vstack([Q, Xk[:NbQ, len(Xk[NbQd-1])-1]])
        V = np.vstack([V, Xk[NbQ:, len(Xk[NbQd-1])-1]])
        Nbis -= 1

    V_Test = np.gradient(Q, dN, edge_order = 1, axis=0)          # By CURIOSITY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Definition of markers positions in array type. Argument : Step calculation, Number of Marker, Coordinate xyz

if CM == 0 :
    markers = [[model.markers(np.array(Q.T[k]))[j].to_array() for j in range(len(model.markers(np.array(Q.T[k]))))] for
               k in range(len(Q1))]
    np.save('DataMarkeur.npy', markers)
    NewMarkers = np.load('DataMarkeur.npy')
elif CM == 1 :
    markers = [[model.markers(np.array(Q[k]))[j].to_array() for j in range(len(model.markers(np.array(Q[k]))))] for k in
               range(N)]
    np.save('DataMarkeur-Couple.npy', markers)
    NewMarkers = np.load('DataMarkeur-Couple.npy')

# print(NewMarkers)
# print(Q)
# print(markers)

# V1 = np.gradient(Q1, tps, edge_order = 1)                      # Approximation plus vrai, edge_order peut être egal a 2, change le premier et dernier terme
# Acc = np.gradient(V1, tps, edge_order = 1)


# Display : creation of plt

if CM == 0 :
    plt.plot(tps, Q.T, label = 'Position')
    plt.plot(tps, V.T, label = 'Vitesse')
    plt.title('Position / Vitesse Initial')
    plt.legend(loc='best')
    plt.show(block=True)
    plt.figure()
# plt.plot(tps, Q1, label='Q_1')
# plt.plot(tps, Q2, label='Q_2')
# plt.title('Position Initial')
# plt.legend(loc='best')
# plt.figure()
# plt.plot(tps, V1, label = 'Qdot_1bis')
# plt.title('Vitesse Initial')
# plt.legend(loc='best')
# plt.show(block=True)
elif CM == 1 :
    plt.plot(tps, Q[:-1, 0], label = 'Position 0 tan')
    plt.plot(tps, Q[:-1, 1], label = 'Position 1 tan')
    plt.plot(tps, V[:-1, 0], label = 'Vitesse 0 tan')
    plt.plot(tps, V[:-1, 1], label = 'Vitesse 1 tan')
    # plt.plot(tps, Vbis_Test[:-1, 0], label = 'Vitesse Test 0 tan')
    # plt.plot(tps, Vbis_Test[:-1, 1], label = 'Vitesse Test 1 tan')
    plt.title('Position Initial')
    plt.legend(loc='best')
    plt.show(block=True)
    plt.figure()
    plt.plot(tps, T1, label = 'Couple 0 tan')
    plt.plot(tps, T2, label = 'Couple 1 tan')
    plt.title('Initial Torque')
    plt.legend(loc='best')
    plt.show(block=True)


# Display : creation biorbd-Viz

if CM == 1:
    qs = np.array([Q[:-1, 0], Q[:-1, 1]])                                                   # argument = different angle of you model by DoF
    np.save("visual", qs.T)                                                                 # qs.T is the transposed matrix to the matrix qs
    b = BiorbdViz(model_path="/home/lim/Documents/code/Models/V6/arm26.bioMod")             # find your model
    b.load_movement(qs.T)
    b.exec()
elif CM == 0:
    qs = np.array([Q1, Q2])                                                               # argument = different angle of you model by DoF
    np.save("visual", qs.T)                                                               # qs.T is the transposed matrix to the matrix qs
    b = BiorbdViz(model_path="/home/lim/Documents/code/Models/V6/arm26.bioMod")           # find your model
    b.load_movement(qs.T)
    b.exec()