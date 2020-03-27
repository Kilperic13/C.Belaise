import matplotlib.pyplot as plt
from casadi import *
import biorbd
from BiorbdViz import BiorbdViz
import numpy as np
import conf as conf

T          = conf.T
N          = conf.N
dN = T/N

model = biorbd.Model("/home/lim/Documents/code/Models/V6/arm26.bioMod")


# Definition du temps et de la position de mon bras


tps = [k * dN for k in range(N + 1)]
q1 = [cos(t) for t in tps]
q2 = [k / 2 for k in q1]

Q = np.array([[q1[k], q2[k]] for k in range(len(q1))])


# Definition de la position des markers en array. argument : Pas de calcul, Numero Marker, Coordonne xyz


# markers = model.markers(np.array(Q[1]))
# markers = [model.markers(np.array(Q[k])) for k in range(len(q1))]
markers = [[model.markers(np.array(Q[k]))[j].to_array() for j in range(len(model.markers(np.array(Q[k]))))] for k in range(len(q1))]

np.save('DataMarkeur.npy', markers)

NewMarkers = np.load('DataMarkeur.npy')
print(NewMarkers)
print(Q)
print(markers)

V = [-sin(t) for t in tps]


# Affichage creation plt


plt.ion()
plt.plot(tps, q1, label='Q_1')
plt.plot(tps, q2, label='Q_2')
plt.plot(tps, V, label='Qdot_1')
# plt.plot(tps, A)
plt.title('Position / Vitesse Initial')
plt.legend(loc='best')
plt.show(block=True)


# Affichage creation Model


qs = np.array([q1, q2])                                                       # argument = les different angle de ton model par DDL
np.save("visual", qs.T)                                                               # qs.T pour transposer de qs
b = BiorbdViz(model_path="/home/lim/Documents/code/Models/V6/arm26.bioMod")              # Va chercher ton modele
b.load_movement(qs.T)
b.exec()