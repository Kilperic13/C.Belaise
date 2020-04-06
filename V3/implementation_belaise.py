import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from casadi import *
import time
import biorbd
from BiorbdViz import BiorbdViz
import conf as conf

T = conf.T
N = conf.N
CM = conf.CM
dN = T/N


model = biorbd.Model("/home/lim/Documents/code/Models/V3/arm26.bioMod")


# Creation des fonctions integral

#L = lambda tau: tau * tau
# f = lambda x, tau: vertcat(x[-model.nbQdot():], model.ForwardDynamics(x[:model.nbQ()], x[-model.nbQdot():], tau))
x = MX.sym('x', model.nbQ()*2, 1)
q = MX.sym('q', model.nbQ(), 1)
tau = MX.sym('tau', model.nbQ(), 1)
F = Function('F', [x, tau], [vertcat(x[-model.nbQdot():], model.ForwardDynamics(x[:model.nbQ()], x[-model.nbQdot():], tau).to_mx())]).expand()
Markers = Function('markers', [q], [model.markers(q)]).expand()
x = SX.sym('x', model.nbQ()*2, 1)
tau = SX.sym('tau', model.nbQ(), 1)
dae = {'x': x, 'p': tau, 'ode': F(x, tau)}
opts = {'tf': T / N, 'number_of_finite_elements': 5}
INT = integrator('INT', 'rk', dae, opts)


#Initialisation Valeur

w = []                                      #Etat
w0 = []                                     #Etat initial
ubw = []
lbw = []
J = 0                                       #A optimiser
g = []                                      #Contrainte
ubg = []
lbg = []
DataMarkeur = np.load('DataMarkeur-Couple2.npy')    #Data position Markeur Reel - CREER Virtuel
Ncmv = len(DataMarkeur)                     # Nombre de dataMarkeur, CMV : Creat
# Shapecmv = DataMarkeur.shape
# DataMarkeur += 0.1*np.random.randn(Shapecmv[0], Shapecmv[1], Shapecmv[2])
Nb_Markeur = model.nbMarkers()              #Nombre de Markeur Model
Nb_Torque = model.nbGeneralizedTorque()
wMa = conf.wMa
wMt = conf.wMt

# Intitialisation lineaire de Q

Initi_Debut = 0
Initi_Fin = 10
PI = [((Initi_Fin - Initi_Debut) * k * dN + Initi_Debut) for k in range(N+1)]

#Creation premier noeud

tps = [k * dN for k in range(N + 1)]
q1 = [cos(t) for t in tps]
q2 = [k / 2 for k in q1]

Q = np.array([[q1[k], q2[k]] for k in range(len(q1))])

Xk = MX.sym('X0', (model.nbQ() + model.nbQdot()))           #ici 4 car 2 DDL et leurs vitesses (Theta1 - Wtheta1 - Theta2 - Wtheta2)
w += [Xk]
ubw += [10] * (model.nbQ()) + [10] * (model.nbQdot())
lbw += [-10] * (model.nbQ()) + [-10] * (model.nbQdot())
w0 += [PI[0]] * (model.nbQ()) + [0] * (model.nbQdot())
# w0 += list(Q[0])
# w0 += list(np.zeros((2, 1)))

#Q = Xk[:model.nbQ()]
markers = Markers(Xk[:model.nbQ()])
J += fctBel.fcn_objective_markers_Lea(wMa, wMt, markers, DataMarkeur[0])

# Acck = MX.sym('A0', model.nbQddot())                        #car 2 DDL, acceleration angulaire (Acc_theta1, Acc_theta2)
# w += [Acck]
# ubw += [0] * (model.nbQddot())
# lbw += [0] * (model.nbQddot())
# w0 += [0] * (model.nbQddot())

# PMk = [[[0 * 3] * Nb_Markeur] * N]                          #Position Markeurs

t = time.time()


#Formulation du NLP

for k in range(N):
    UTarc = MX.sym('UTarc_' + str(k), Nb_Torque)
    w += [UTarc]
    ubw += [25] * Nb_Torque
    lbw += [-25] * Nb_Torque
    w0 += [0] * Nb_Torque

    # J += UTarc * UTarc
    # J += [UTarc[k] * UTarc[k] for k in range(Nb_Torque)]
    # J += sum([UTarc[k] * UTarc[k] for k in range(Nb_Torque)])       # U.T.mtimes(U)     Produit scalaire
    # J += 0.1*mtimes(UTarc.T, UTarc) + 0.1*mtimes(Xk.T, Xk)
    J += 1 * mtimes(UTarc.T, UTarc)

    # Calcul q' et q en k+1
    Fk = INT(x0 = Xk, p = UTarc)
    Xkend = Fk['xf']

    Xk = MX.sym('X_' + str(k + 1), (model.nbQ() + model.nbQdot()))
    w += [Xk]
    # if k == N-1:
    #     ubw += [1] * (model.nbQ()) + [0] * (model.nbQdot())
    #     lbw += [-1] * (model.nbQ()) + [0] * (model.nbQdot())
    #     w0 += [-1] * (model.nbQ()) + [0] * (model.nbQdot())
    # else:
    ubw += [10] * (model.nbQ() + model.nbQdot())
    lbw += [-10] * (model.nbQ() + model.nbQdot())
    w0 += [PI[k + 1]] * model.nbQ() + [0] * model.nbQdot()
    # w0 += list(Q[k+1])
    # w0 += list(np.zeros((2, 1)))

    g += [Xkend - Xk]
    ubg += [0] * (model.nbQ() + model.nbQdot())
    lbg += [0] * (model.nbQ() + model.nbQdot())

    # Q = Xk[:model.nbQdot()+model.nbQ():2]
    #Q = Xk[:model.nbQ()]
    markers = Markers(Xk[:model.nbQ()])
    J += fctBel.fcn_objective_markers_Lea(wMa, wMt, markers, DataMarkeur[int((k+1)*Ncmv/N - 1)])                  # int() et pas round() pour eviter de retomber sur quelque chose supperieur a Ncmv
    # J += fctBel.fcn_objective_markers_Lea(wMa, wMt, markers, DataMarkeur[k+1])                  # int() et pas round() pour eviter de retomber sur quelque chose supperieur a Ncmv

# UTarc = MX.sym('UTarc_' + str(N), Nb_Torque)        # Ajout pour avoir N+1 torque
# w += [UTarc]
# ubw += [100] * Nb_Torque
# lbw += [-100] * Nb_Torque
# w0 += [0] * Nb_Torque
obj = Function('obj', [vertcat(*w)], [J])
print('Value of obj at 0 :')
print(obj(w0))


print(f"Time to create the formulation of the NLP {time.time() - t}")


# Create NLP solver
t = time.time()
opts = {'ipopt.linear_solver': 'ma57', 'ipopt.tol': 1e-8, 'ipopt.constr_viol_tol': 1e-3,       #Option du solver
        'ipopt.hessian_approximation': 'exact'}
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}                                             #creation du probleme : f a minimiser, x se qu il peut modifier, g les contraintes a respecter
solver = nlpsol('solver', 'ipopt', prob, opts)
print(f"Time to create regular problem {time.time() - t}")

t = time.time()
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)                                         #Resolution, avec les etats&controle initiaux proposer, les limites de l etat, les limite du controle
print(f"Time to solve regular problem {time.time() - t}")


Tart_opt = sol['x'].full().flatten()                                                               #Bibliotheque de solution, contenant f, g, et x

np.save('Couple_Opt_V3-Couple2.npy', Tart_opt)

