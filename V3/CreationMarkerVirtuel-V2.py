import matplotlib.pyplot as plt
from casadi import *
import biorbd
import numpy as np
import math
from BiorbdViz import BiorbdViz

T = 3.2
N = 1510
dN = T/N

model = biorbd.Model("/home/lim/Documents/code/Models/V3/arm26.bioMod")


# Definition du temps et de la position de mon bras


tps = [k * dN for k in range(N + 1)]
q1 = [cos(t) for t in tps]
q2 = [k / 2 for k in q1]

Q = np.array([[q1[k], q2[k]] for k in range(len(q1))])


# Definition de la position des markers en array. argument : Pas de calcul, Numero Marker, Coordonne xyz


# markers = model.markers(np.array(Q[1]))
# markers = [model.markers(np.array(Q[k])) for k in range(len(q1))]
markers = [[model.markers(np.array(Q[k]))[j].to_array() for j in range(len(model.markers(np.array(Q[k]))))] for k in range(len(q1))]

# from tempfile import TemporaryFile
# DataMarkeur = TemporaryFile()
np.save('DataMarkeur.npy', markers)

NewMarkers = np.load('DataMarkeur.npy')
print(NewMarkers)
print(Q)
print(markers)

V = [-sin(t) for t in tps]
Vbis = [(q1[k+1] - q1[k])/dN for k in range(len(q1) - 1)]           # Approximation Euler ordre 1
Vbis.append((q1[N] - q1[N-1])/dN)
Vbisbis = np.gradient(q1, tps, edge_order = 1)                      # Approximation plus vrai, edge_order peut être egal a 2, change le premier et dernier terme
Acc = np.gradient(Vbisbis, tps, edge_order = 1)

# Affichage creation plt

plt.title('Position / Vitesse Initial')
plt.legend(loc='best')
plt.ion()
plt.plot(tps, q1, label='Q_1')
plt.plot(tps, q2, label='Q_2')
plt.title('Position Initial')
plt.legend(loc='best')
plt.figure()
plt.plot(tps, V, label='Qdot_1')
plt.plot(tps, Vbis, label = 'Qdot_1bis')
plt.plot(tps, Vbisbis, label = 'Qdot_1bisbis')
plt.title('Vitesse Initial')
plt.legend(loc='best')
plt.show(block=True)


# Affichage creation Model


qs = np.array([q1, q2])                                                       # argument = les different angle de ton model par DDL
np.save("visual", qs.T)                                                               # qs.T pour transposer de qs
b = BiorbdViz(model_path="/home/lim/Documents/code/Models/V3/arm26.bioMod")              # Va chercher ton modele
b.load_movement(qs.T)
b.exec()